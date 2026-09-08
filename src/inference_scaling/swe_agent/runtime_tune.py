"""Successive-halving tuner for the two-card Conditional IS service."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from inference_scaling.swe_agent.benchmark import _load_records, run_burst


@dataclass(frozen=True, slots=True)
class EngineConfig:
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float

    @property
    def config_id(self) -> str:
        memory = str(self.gpu_memory_utilization).replace(".", "p")
        return f"mns{self.max_num_seqs}-mbt{self.max_num_batched_tokens}-mem{memory}"

    def overrides(self) -> tuple[str, ...]:
        return (
            f"vllm.max_num_seqs={self.max_num_seqs}",
            f"vllm.max_num_batched_tokens={self.max_num_batched_tokens}",
            f"vllm.gpu_memory_utilization={self.gpu_memory_utilization}",
        )


def engine_grid(
    max_num_seqs: Sequence[int] = (128, 192, 256, 384),
    max_num_batched_tokens: Sequence[int] = (16384, 32768, 65536),
    gpu_memory_utilization: Sequence[float] = (0.90, 0.92, 0.94),
) -> tuple[EngineConfig, ...]:
    return tuple(
        EngineConfig(int(mns), int(mbt), float(memory))
        for mns, mbt, memory in itertools.product(
            max_num_seqs, max_num_batched_tokens, gpu_memory_utilization
        )
    )


def _preemptions(result: dict[str, Any]) -> int:
    total = 0
    for measurement in result.get("measurements", ()):
        counters = (measurement.get("diagnostics") or {}).get("backend_delta") or {}
        total += sum(
            int(value)
            for key, value in counters.items()
            if "preempt" in str(key).lower() and isinstance(value, (int, float))
        )
    return total


def select_arms(
    results: Sequence[dict[str, Any]],
    *,
    keep: int,
    p95_factor: float = 1.25,
) -> list[dict[str, Any]]:
    if keep <= 0 or p95_factor < 1:
        raise ValueError("invalid successive-halving gate")
    valid = [
        result
        for result in results
        if result.get("success_rate") == 1.0
        and _preemptions(result) == 0
        and not any(
            "oom" in str(item.get("error", "")).lower()
            for item in result.get("measurements", ())
        )
    ]
    if not valid:
        return []
    minimum_p95 = min(result["latency_seconds"]["p95"] for result in valid)
    gated = [
        result
        for result in valid
        if result["latency_seconds"]["p95"] <= minimum_p95 * p95_factor
    ]
    gated.sort(
        key=lambda result: (
            -float(result["jobs_per_second"]),
            float(result["latency_seconds"]["p95"]),
            str(result["arm_id"]),
        )
    )
    return gated[:keep]


def _wait_for_health(endpoint: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"service exited during startup with {process.returncode}"
            )
        try:
            with urllib.request.urlopen(endpoint + "/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as current:
            error = current
        time.sleep(1)
    raise TimeoutError(f"service did not become healthy: {error}")


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _run_arm(
    *,
    config: EngineConfig,
    workers: int,
    records: Sequence[dict[str, Any]],
    warmup_records: Sequence[dict[str, Any]],
    service_config: Path,
    endpoint: str,
    port: int,
    devices: str,
    startup_timeout: float,
    request_timeout: float,
    seed: int,
    log_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "inference_scaling.swe_agent.server",
        "--config",
        str(service_config),
        "--port",
        str(port),
    ]
    for override in config.overrides():
        command.extend(("--set", override))
    environment = dict(os.environ)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = devices
    environment["CIS_INSTANCE_ID"] = config.config_id
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            _wait_for_health(endpoint, process, startup_timeout)
            if warmup_records:
                warmup = run_burst(
                    warmup_records,
                    (endpoint,),
                    workers=min(workers, len(warmup_records)),
                    timeout=request_timeout,
                    seed=seed,
                )
                if warmup["success_rate"] != 1.0:
                    raise RuntimeError("warmup workload failed")
            result = run_burst(
                records,
                (endpoint,),
                workers=workers,
                timeout=request_timeout,
                seed=seed,
            )
        except Exception as error:
            result = {
                "schema_version": 1,
                "requests": len(records),
                "workers": workers,
                "success_rate": 0.0,
                "jobs_per_second": 0.0,
                "latency_seconds": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0},
                "measurements": [{"error": f"{type(error).__name__}: {error}"}],
            }
        finally:
            _stop(process)
    result.update(
        {
            "arm_id": f"{config.config_id}-w{workers}",
            "engine": asdict(config),
            "workers": workers,
            "service_log": str(log_path),
            "started_at": started,
        }
    )
    return result


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--warmup-workload")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--worker", type=int, action="append")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = engine_grid()
    workers = tuple(args.worker or (8, 16, 32, 64))
    records = _load_records(Path(args.workload))
    if len(records) < 64:
        raise ValueError("runtime tuning requires at least 64 workload records")
    warmup_records = (
        _load_records(Path(args.warmup_workload)) if args.warmup_workload else []
    )
    plan = {
        "schema_version": 1,
        "engine_configs": [asdict(config) for config in configs],
        "workers": list(workers),
        "rounds": [16, 32, 64],
        "devices": args.devices,
        "seed": args.seed,
    }
    output = Path(args.output_directory)
    _write_checkpoint(output / "plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    endpoint = f"http://{args.host}:{args.port}"
    service_config = Path(args.config).resolve()
    history: list[dict[str, Any]] = []

    baseline = EngineConfig(128, 32768, 0.90)
    worker_results = [
        _run_arm(
            config=baseline,
            workers=worker,
            records=records[:16],
            warmup_records=warmup_records,
            service_config=service_config,
            endpoint=endpoint,
            port=args.port,
            devices=args.devices,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
            seed=args.seed,
            log_path=output / "logs" / f"worker-{worker}.log",
        )
        for worker in workers
    ]
    history.extend(worker_results)
    selected_worker_results = select_arms(worker_results, keep=1)
    if not selected_worker_results:
        raise RuntimeError("no worker baseline passed the hard gates")
    selected_worker = int(selected_worker_results[0]["workers"])

    survivors = list(configs)
    for round_index, (request_count, keep) in enumerate(
        ((16, 12), (32, 5), (64, 3)), 1
    ):
        round_results = [
            _run_arm(
                config=config,
                workers=selected_worker,
                records=records[:request_count],
                warmup_records=warmup_records,
                service_config=service_config,
                endpoint=endpoint,
                port=args.port,
                devices=args.devices,
                startup_timeout=args.startup_timeout,
                request_timeout=args.request_timeout,
                seed=args.seed,
                log_path=output
                / "logs"
                / f"round-{round_index}-{config.config_id}.log",
            )
            for config in survivors
        ]
        history.extend(round_results)
        selected = select_arms(round_results, keep=min(keep, len(round_results)))
        survivors = [EngineConfig(**result["engine"]) for result in selected]
        _write_checkpoint(
            output / f"round-{round_index}.json",
            {
                "request_count": request_count,
                "results": round_results,
                "selected": selected,
            },
        )
        if not survivors:
            raise RuntimeError(f"no configuration survived round {round_index}")

    final_results = [
        _run_arm(
            config=config,
            workers=worker,
            records=records[:64],
            warmup_records=warmup_records,
            service_config=service_config,
            endpoint=endpoint,
            port=args.port,
            devices=args.devices,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
            seed=args.seed,
            log_path=output / "logs" / f"final-{config.config_id}-w{worker}.log",
        )
        for config in survivors
        for worker in workers
    ]
    history.extend(final_results)
    winner = select_arms(final_results, keep=1)
    result = {**plan, "history": history, "winner": winner[0] if winner else None}
    _write_checkpoint(output / "result.json", result)
    print(json.dumps({"winner": result["winner"]}, indent=2))


if __name__ == "__main__":
    main()
