"""Send a fixed burst workload to one or more Conditional IS services."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import statistics
import threading
import time
import urllib.error
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
    action_valid: bool | None
    seconds: float
    error: str | None
    diagnostics: dict[str, Any] | None
    started_at: float
    finished_at: float
    transport_retries: int = 0


ROUTING_MODES = (
    "round_robin",
    "least_outstanding",
    "least_cis_work",
    "cis_work_balanced",
)


def estimate_cis_attention_work(
    record: dict[str, Any],
    conditional: dict[str, int] | None,
    *,
    default_total_length: int = 512,
) -> float:
    """Estimate the dense attention work of one complete CIS request.

    This deliberately models the algorithmic branch tree rather than a top-level
    request count.  It is an upper-bound proxy: EOS can reduce real work, but the
    ordering remains useful before any branch has started.
    """

    diagnostics = record.get("diagnostics")
    prompt_tokens = (
        diagnostics.get("prompt_tokens")
        if isinstance(diagnostics, dict)
        else None
    )
    if not isinstance(prompt_tokens, (int, float)) or prompt_tokens <= 0:
        raise ValueError(
            "cis_work_balanced routing requires diagnostics.prompt_tokens"
        )
    values = conditional or {}
    candidates = int(values.get("candidate_count", 1))
    rollouts = int(values.get("rollout_count", 1))
    block = int(values.get("block_size", 1))
    total = int(values.get("total_length", default_total_length))
    if min(candidates, rollouts, block, total) <= 0:
        raise ValueError("Conditional IS work parameters must be positive")

    work = 0.0
    generated = 0
    while generated < total:
        candidate_tokens = min(block, total - generated)
        candidate_prefix = float(prompt_tokens + generated)
        # Sum the KV length read by each autoregressive token in this block.
        work += candidates * (
            candidate_tokens * candidate_prefix
            + candidate_tokens * (candidate_tokens + 1) / 2
        )
        rollout_tokens = total - generated - candidate_tokens
        rollout_prefix = candidate_prefix + candidate_tokens
        work += candidates * rollouts * (
            rollout_tokens * rollout_prefix
            + rollout_tokens * (rollout_tokens + 1) / 2
        )
        generated += candidate_tokens
    return work


def balanced_cis_assignments(
    records: Sequence[dict[str, Any]],
    endpoints: Sequence[str],
    conditional: dict[str, int] | None,
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Assign whole CIS trees with deterministic longest-processing-time first."""

    loads = {endpoint: 0.0 for endpoint in endpoints}
    assignments: list[str | None] = [None] * len(records)
    estimates = [estimate_cis_attention_work(record, conditional) for record in records]
    for index in sorted(range(len(records)), key=lambda item: (-estimates[item], item)):
        endpoint = min(endpoints, key=lambda item: (loads[item], endpoints.index(item)))
        assignments[index] = endpoint
        loads[endpoint] += estimates[index]
    return tuple(value for value in assignments if value is not None), loads


class EndpointRouter:
    """Thread-safe whole-job routing with observable per-instance pressure."""

    def __init__(
        self,
        endpoints: Sequence[str],
        mode: str,
        *,
        assignments: Sequence[str] | None = None,
        estimated_loads: dict[str, float] | None = None,
        work_estimates: Sequence[float] | None = None,
    ) -> None:
        if mode not in ROUTING_MODES:
            raise ValueError(f"unknown routing mode: {mode}")
        if mode == "cis_work_balanced":
            if assignments is None:
                raise ValueError("cis_work_balanced routing requires assignments")
            if any(endpoint not in endpoints for endpoint in assignments):
                raise ValueError("routing assignment references an unknown endpoint")
        if mode == "least_cis_work":
            if work_estimates is None:
                raise ValueError("least_cis_work routing requires work estimates")
            if any(value <= 0 for value in work_estimates):
                raise ValueError("CIS work estimates must be positive")
        self.endpoints = tuple(endpoints)
        self.mode = mode
        self._assignments = tuple(assignments or ())
        self._estimated_loads = dict(estimated_loads or {})
        self._work_estimates = tuple(float(value) for value in (work_estimates or ()))
        self._lock = threading.Lock()
        self._cursor = 0
        self._outstanding = {endpoint: 0 for endpoint in endpoints}
        self._outstanding_work = {endpoint: 0.0 for endpoint in endpoints}
        self._assigned = {endpoint: 0 for endpoint in endpoints}
        self._assigned_work = {endpoint: 0.0 for endpoint in endpoints}
        self._maximum = {endpoint: 0 for endpoint in endpoints}
        self._maximum_work = {endpoint: 0.0 for endpoint in endpoints}

    def acquire(self, request_index: int) -> str:
        with self._lock:
            if self.mode == "cis_work_balanced":
                endpoint = self._assignments[request_index]
                index = self.endpoints.index(endpoint)
            elif self.mode == "round_robin":
                index = self._cursor % len(self.endpoints)
            elif self.mode == "least_cis_work":
                minimum = min(self._outstanding_work.values())
                eligible = {
                    endpoint
                    for endpoint, work in self._outstanding_work.items()
                    if work == minimum
                }
                index = next(
                    offset % len(self.endpoints)
                    for offset in range(
                        self._cursor, self._cursor + len(self.endpoints)
                    )
                    if self.endpoints[offset % len(self.endpoints)] in eligible
                )
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
            if self.mode == "least_cis_work":
                work = self._work_estimates[request_index]
                self._outstanding_work[endpoint] += work
                self._assigned_work[endpoint] += work
                self._maximum_work[endpoint] = max(
                    self._maximum_work[endpoint], self._outstanding_work[endpoint]
                )
            self._maximum[endpoint] = max(
                self._maximum[endpoint], self._outstanding[endpoint]
            )
            return endpoint

    def release(self, endpoint: str, request_index: int | None = None) -> None:
        with self._lock:
            self._outstanding[endpoint] -= 1
            if self.mode == "least_cis_work":
                if request_index is None:
                    raise ValueError(
                        "least_cis_work release requires the request index"
                    )
                self._outstanding_work[endpoint] = max(
                    0.0,
                    self._outstanding_work[endpoint]
                    - self._work_estimates[request_index],
                )

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self.mode,
                "assigned": dict(self._assigned),
                "maximum_outstanding": dict(self._maximum),
                "final_outstanding": dict(self._outstanding),
                "estimated_attention_work": dict(
                    self._estimated_loads or self._assigned_work
                ),
                "maximum_outstanding_attention_work": dict(self._maximum_work),
                "final_outstanding_attention_work": dict(self._outstanding_work),
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


def _post_with_retry(
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
    retries: int,
) -> tuple[dict[str, Any], int]:
    completed_retries = 0
    while True:
        try:
            return _post(endpoint, payload, timeout), completed_retries
        except urllib.error.HTTPError as error:
            if error.code not in {502, 503, 504} or completed_retries >= retries:
                raise
        except (OSError, TimeoutError, http.client.HTTPException):
            if completed_retries >= retries:
                raise
        completed_retries += 1
        time.sleep(min(1.0, 0.25 * (2 ** (completed_retries - 1))))


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
    require_action: bool = False,
    transport_retries: int = 2,
) -> dict[str, Any]:
    if not records or not endpoints:
        raise ValueError("burst requires records and endpoints")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if transport_retries < 0:
        raise ValueError("transport retries must be non-negative")
    run_namespace = run_namespace or f"burst:{time.time_ns()}"
    release = threading.Event()
    assignments = None
    estimated_loads = None
    work_estimates = None
    if routing == "cis_work_balanced":
        assignments, estimated_loads = balanced_cis_assignments(
            records, endpoints, conditional_overrides
        )
    elif routing == "least_cis_work":
        work_estimates = tuple(
            estimate_cis_attention_work(record, conditional_overrides)
            for record in records
        )
    router = EndpointRouter(
        endpoints,
        routing,
        assignments=assignments,
        estimated_loads=estimated_loads,
        work_estimates=work_estimates,
    )
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
        endpoint = router.acquire(index)
        started = time.perf_counter()
        started_at = time.time()
        completed_retries = 0
        try:
            response, completed_retries = _post_with_retry(
                endpoint, payload, timeout, transport_retries
            )
            actions = response.get("message", {}).get("extra", {}).get("actions", [])
            action_valid = bool(actions)
            if require_action and not action_valid:
                return RequestMeasurement(
                    request_id,
                    endpoint,
                    False,
                    False,
                    time.perf_counter() - started,
                    "selected completion has no valid action",
                    response.get("diagnostics"),
                    started_at,
                    time.time(),
                    completed_retries,
                )
            return RequestMeasurement(
                request_id,
                endpoint,
                True,
                action_valid,
                time.perf_counter() - started,
                None,
                response.get("diagnostics"),
                started_at,
                time.time(),
                completed_retries,
            )
        except Exception as error:
            return RequestMeasurement(
                request_id,
                endpoint,
                False,
                None,
                time.perf_counter() - started,
                f"{type(error).__name__}: {error}",
                None,
                started_at,
                time.time(),
                completed_retries,
            )
        finally:
            router.release(endpoint, index)

    wall_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(execute, index, record)
            for index, record in enumerate(records)
        ]
        released_at = time.time()
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
    observed_actions = [
        item.action_valid for item in measurements if item.action_valid is not None
    ]
    throughput = {
        "jobs_per_second": successes / wall_seconds if wall_seconds else 0.0,
        "generated_tokens_per_second": (
            float(backend_delta.get("generated_tokens", 0.0)) / wall_seconds
            if wall_seconds
            else 0.0
        ),
        "prefill_tokens_per_second": (
            float(backend_delta.get("prefill_tokens", 0.0)) / wall_seconds
            if wall_seconds
            else 0.0
        ),
        "generation_forward_token_slots_per_second": (
            float(backend_delta.get("generation_forward_token_slots", 0.0))
            / wall_seconds
            if wall_seconds
            else 0.0
        ),
    }
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
        "released_at": released_at,
        "jobs_per_second": throughput["jobs_per_second"],
        "throughput": throughput,
        "successes": successes,
        "success_rate": successes / len(measurements),
        "transport_retries": sum(item.transport_retries for item in measurements),
        "require_action": require_action,
        "action_valid_rate": (
            sum(observed_actions) / len(observed_actions) if observed_actions else None
        ),
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
    parser.add_argument(
        "--require-action",
        action="store_true",
        help="treat a completed CIS response without a bash action as failed",
    )
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--rollout-count", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument(
        "--routing",
        choices=ROUTING_MODES,
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
        require_action=args.require_action,
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
