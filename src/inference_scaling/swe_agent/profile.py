"""Capture a measured Conditional IS burst with vLLM-Ascend worker profiling."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from inference_scaling.swe_agent.benchmark import _load_records, run_burst
from inference_scaling.swe_agent.runtime_tune import _stop, _wait_for_health


def _post_control(
    endpoint: str, operation: str, payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + f"/v1/profile/{operation}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("profile control returned a non-object response")
    return value


def _conditional_overrides(args: argparse.Namespace) -> dict[str, int]:
    return {
        key: value
        for key, value in {
            "candidate_count": args.candidate_count,
            "rollout_count": args.rollout_count,
            "block_size": args.block_size,
        }.items()
        if value is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--warmup-workload")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--profile-stop-timeout", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--profile-prefix", default="conditional-is")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-stack", action="store_true")
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--rollout-count", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    output = Path(args.output_directory).resolve()
    profile_directory = output / "npu-profile"
    trace_path = output / "algorithm-trace.jsonl"
    service_log_path = output / "service.log"
    output.mkdir(parents=True, exist_ok=True)
    profile_directory.mkdir(parents=True, exist_ok=True)
    records = _load_records(Path(args.workload))[: args.limit]
    if not records:
        raise ValueError("profiling requires at least one workload record")
    warmup = (
        _load_records(Path(args.warmup_workload))
        if args.warmup_workload
        else []
    )
    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": str(profile_directory),
        "torch_profiler_with_stack": args.profile_stack,
        "torch_profiler_with_memory": args.profile_memory,
        "ignore_frontend": False,
    }
    command = [
        sys.executable,
        "-m",
        "inference_scaling.swe_agent.server",
        "--config",
        str(Path(args.config).resolve()),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--set",
        "vllm.engine_kwargs.profiler_config="
        + json.dumps(profiler_config, separators=(",", ":")),
        "--set",
        f"service.trace_path={json.dumps(str(trace_path))}",
    ]
    for override in args.overrides:
        command.extend(("--set", override))
    environment = dict(os.environ)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
    endpoint = f"http://{args.host}:{args.port}"
    started_at = time.time()
    profiling_started = False
    with service_log_path.open("w", encoding="utf-8") as service_log:
        process = subprocess.Popen(
            command,
            stdout=service_log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            _wait_for_health(endpoint, process, args.startup_timeout)
            if warmup:
                warmup_result = run_burst(
                    warmup,
                    (endpoint,),
                    workers=min(args.workers, len(warmup)),
                    timeout=args.request_timeout,
                    seed=args.seed,
                    conditional_overrides=_conditional_overrides(args),
                )
                if warmup_result["success_rate"] != 1.0:
                    raise RuntimeError("profiling warmup failed")
            _post_control(
                endpoint,
                "start",
                {"prefix": args.profile_prefix},
                args.profile_stop_timeout,
            )
            profiling_started = True
            result = run_burst(
                records,
                (endpoint,),
                workers=args.workers,
                timeout=args.request_timeout,
                seed=args.seed,
                conditional_overrides=_conditional_overrides(args),
            )
        finally:
            try:
                if profiling_started:
                    _post_control(
                        endpoint, "stop", {}, args.profile_stop_timeout
                    )
            finally:
                _stop(process)

    result.update(
        {
            "profile": {
                "kind": "vllm-ascend-torch",
                "directory": str(profile_directory),
                "prefix": args.profile_prefix,
                "memory": args.profile_memory,
                "stack": args.profile_stack,
            },
            "devices": args.devices,
            "service_log": str(service_log_path),
            "algorithm_trace": str(trace_path),
            "service_command": command,
            "started_at": started_at,
        }
    )
    (output / "benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "success_rate": result["success_rate"],
                "jobs_per_second": result["jobs_per_second"],
                "p95": result["latency_seconds"]["p95"],
                "profile_directory": str(profile_directory),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
