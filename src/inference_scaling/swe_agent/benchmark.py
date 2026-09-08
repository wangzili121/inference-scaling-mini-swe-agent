"""Send a fixed burst workload to one or more Conditional IS services."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from inference_scaling.shared.rng import SeedStream


@dataclass(frozen=True, slots=True)
class RequestMeasurement:
    request_id: str
    endpoint: str
    success: bool
    seconds: float
    error: str | None
    diagnostics: dict[str, Any] | None
    started_at: float
    finished_at: float


class EndpointRouter:
    """Thread-safe whole-job routing with observable per-instance pressure."""

    def __init__(self, endpoints: Sequence[str], mode: str) -> None:
        if mode not in {"round_robin", "least_outstanding"}:
            raise ValueError(f"unknown routing mode: {mode}")
        self.endpoints = tuple(endpoints)
        self.mode = mode
        self._lock = threading.Lock()
        self._cursor = 0
        self._outstanding = {endpoint: 0 for endpoint in endpoints}
        self._assigned = {endpoint: 0 for endpoint in endpoints}
        self._maximum = {endpoint: 0 for endpoint in endpoints}

    def acquire(self) -> str:
        with self._lock:
            if self.mode == "round_robin":
                index = self._cursor % len(self.endpoints)
            else:
                minimum = min(self._outstanding.values())
                eligible = {
                    endpoint
                    for endpoint, count in self._outstanding.items()
                    if count == minimum
                }
                index = next(
                    offset % len(self.endpoints)
                    for offset in range(
                        self._cursor, self._cursor + len(self.endpoints)
                    )
                    if self.endpoints[offset % len(self.endpoints)] in eligible
                )
            endpoint = self.endpoints[index]
            self._cursor = index + 1
            self._outstanding[endpoint] += 1
            self._assigned[endpoint] += 1
            self._maximum[endpoint] = max(
                self._maximum[endpoint], self._outstanding[endpoint]
            )
            return endpoint

    def release(self, endpoint: str) -> None:
        with self._lock:
            self._outstanding[endpoint] -= 1

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                "assigned": dict(self._assigned),
                "maximum_outstanding": dict(self._maximum),
                "final_outstanding": dict(self._outstanding),
            }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _post(endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/query",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("service returned a non-object response")
    return value


def _backend_snapshot(endpoint: str, timeout: float = 10.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(
            endpoint.rstrip("/") + "/v1/diagnostics", timeout=timeout
        ) as response:
            value = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    backend = value.get("backend") if isinstance(value, dict) else None
    return dict(backend) if isinstance(backend, dict) else None


def _snapshot_delta(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> dict[str, float]:
    if before is None or after is None:
        return {}
    return {
        key: float(value) - float(before[key])
        for key, value in after.items()
        if key in before
        and isinstance(value, (int, float))
        and isinstance(before[key], (int, float))
    }


def run_burst(
    records: Sequence[dict[str, Any]],
    endpoints: Sequence[str],
    *,
    workers: int,
    timeout: float = 7200.0,
    seed: int = 20260908,
    conditional_overrides: dict[str, int] | None = None,
    run_namespace: str | None = None,
    routing: str = "round_robin",
    after_release: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if not records or not endpoints:
        raise ValueError("burst requires records and endpoints")
    if workers <= 0:
        raise ValueError("workers must be positive")
    run_namespace = run_namespace or f"burst:{time.time_ns()}"
    release = threading.Event()
    router = EndpointRouter(endpoints, routing)
    before_snapshots = {endpoint: _backend_snapshot(endpoint) for endpoint in endpoints}

    def execute(index: int, record: dict[str, Any]) -> RequestMeasurement:
        request_id = f"{run_namespace}:{index}:{record['request_id']}"
        payload = {
            "messages": record["messages"],
            "request_id": request_id,
            "seed": SeedStream(seed).derive("burst", index, record["request_id"]),
        }
        if conditional_overrides:
            payload["conditional_is"] = dict(conditional_overrides)
        release.wait()
        endpoint = router.acquire()
        started = time.perf_counter()
        started_at = time.time()
        try:
            response = _post(endpoint, payload, timeout)
            actions = response.get("message", {}).get("extra", {}).get("actions", [])
            if not actions:
                raise RuntimeError("selected completion has no valid action")
            return RequestMeasurement(
                request_id,
                endpoint,
                True,
                time.perf_counter() - started,
                None,
                response.get("diagnostics"),
                started_at,
                time.time(),
            )
        except Exception as error:
            return RequestMeasurement(
                request_id,
                endpoint,
                False,
                time.perf_counter() - started,
                f"{type(error).__name__}: {error}",
                None,
                started_at,
                time.time(),
            )
        finally:
            router.release(endpoint)

    wall_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(execute, index, record)
            for index, record in enumerate(records)
        ]
        release.set()
        if after_release is not None:
            after_release()
        measurements = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - wall_started
    after_snapshots = {endpoint: _backend_snapshot(endpoint) for endpoint in endpoints}
    endpoint_backend_delta = {
        endpoint: _snapshot_delta(before_snapshots[endpoint], after_snapshots[endpoint])
        for endpoint in endpoints
    }
    backend_keys = {key for delta in endpoint_backend_delta.values() for key in delta}
    backend_delta = {
        key: sum(delta.get(key, 0.0) for delta in endpoint_backend_delta.values())
        for key in backend_keys
    }
    latencies = [item.seconds for item in measurements]
    successes = sum(item.success for item in measurements)
    return {
        "schema_version": 1,
        "requests": len(measurements),
        "workers": workers,
        "endpoints": list(endpoints),
        "conditional_is": dict(conditional_overrides or {}),
        "run_namespace": run_namespace,
        "routing": router.diagnostics(),
        "backend_delta": backend_delta,
        "endpoint_backend_delta": endpoint_backend_delta,
        "wall_seconds": wall_seconds,
        "jobs_per_second": successes / wall_seconds if wall_seconds else 0.0,
        "successes": successes,
        "success_rate": successes / len(measurements),
        "latency_seconds": {
            "mean": statistics.fmean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
        },
        "measurements": [asdict(item) for item in measurements],
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        help="run only the first N workload records after loading the fixed manifest",
    )
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--rollout-count", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument(
        "--routing",
        choices=("round_robin", "least_outstanding"),
        default="round_robin",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    conditional_overrides = {
        key: value
        for key, value in {
            "candidate_count": args.candidate_count,
            "rollout_count": args.rollout_count,
            "block_size": args.block_size,
        }.items()
        if value is not None
    }
    records = _load_records(Path(args.workload))
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        records = records[: args.limit]
    result = run_burst(
        records,
        args.endpoint,
        workers=args.workers,
        timeout=args.timeout,
        seed=args.seed,
        conditional_overrides=conditional_overrides,
        routing=args.routing,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "measurements"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
