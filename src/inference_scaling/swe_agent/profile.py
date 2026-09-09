"""Capture stable-window service or NPU profiles for saturated CIS bursts."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TextIO

from inference_scaling.swe_agent.benchmark import (
    _backend_snapshot,
    _load_records,
    run_burst,
)
from inference_scaling.swe_agent.artifacts import sha256_file, write_artifact_manifest
from inference_scaling.swe_agent.runtime_tune import _stop, _wait_for_health


SERVICE_PROFILER_VERSION = "1.2.2"
ENVIRONMENT_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(slots=True)
class ProfileService:
    instance_id: str
    devices: str
    endpoint: str
    process: subprocess.Popen
    log: TextIO
    trace_path: Path
    profile_directory: Path
    service_profile_config: Path | None


def _verify_writable_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".profile-preflight-{os.getpid()}"
    try:
        probe.write_text("ok\n", encoding="utf-8")
    finally:
        probe.unlink(missing_ok=True)


def _profile_preflight(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    """Reject incomplete profiler environments before loading the model."""

    required_files = {
        "config": Path(args.config).resolve(),
        "workload": Path(args.workload).resolve(),
    }
    if args.warmup_workload:
        required_files["warmup workload"] = Path(args.warmup_workload).resolve()
    if args.profiling_symbols:
        required_files["profiling symbols"] = Path(args.profiling_symbols).resolve()
    missing = [name for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "profiling preflight missing " + ", ".join(sorted(missing))
        )
    child_environment = dict(os.environ)
    for assignment in args.environment:
        if "=" not in assignment:
            raise ValueError(f"environment override requires KEY=VALUE: {assignment}")
        key, value = assignment.split("=", 1)
        if value == "__UNSET__":
            child_environment.pop(key, None)
        else:
            child_environment[key] = value
    references = set(
        ENVIRONMENT_REFERENCE.findall(
            required_files["config"].read_text(encoding="utf-8")
        )
    )
    missing_environment = sorted(
        name for name in references if not child_environment.get(name)
    )
    if missing_environment:
        raise RuntimeError(
            "profiling preflight missing environment: "
            + ", ".join(missing_environment)
        )
    _verify_writable_directory(output)

    dependencies: dict[str, Any] = {"output_writable": True}
    if args.profiler == "service":
        try:
            version = importlib.metadata.version("msserviceprofiler")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "service profiling requires msserviceprofiler=="
                f"{SERVICE_PROFILER_VERSION} before model startup"
            ) from error
        if version != SERVICE_PROFILER_VERSION:
            raise RuntimeError(
                "service profiling requires msserviceprofiler=="
                f"{SERVICE_PROFILER_VERSION}, found {version}"
            )
        executable = shutil.which("msserviceprofiler")
        if executable is None:
            raise RuntimeError("msserviceprofiler CLI is not on PATH")
        completed = subprocess.run(
            (executable, "analyze", "--help"),
            text=True,
            capture_output=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "msserviceprofiler analyze preflight failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        dependencies.update(
            {
                "msserviceprofiler": version,
                "msserviceprofiler_cli": executable,
            }
        )
    elif args.profiler == "torch":
        try:
            profiler = importlib.import_module("torch_npu.profiler.profiler")
        except ImportError as error:
            raise RuntimeError(
                "torch profiling requires torch_npu.profiler before model startup"
            ) from error
        if not callable(getattr(profiler, "analyse", None)):
            raise RuntimeError("torch_npu.profiler.profiler.analyse is unavailable")
        dependencies["torch_npu_analyse"] = True
    return dependencies


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


def _write_service_config(path: Path, profile_directory: Path, duration: float) -> None:
    payload = {
        "enable": 0,
        "prof_dir": str(profile_directory),
        "profiler_level": "INFO",
        "host_system_usage_freq": -1,
        "npu_memory_usage_freq": -1,
        "acl_task_time": 0,
        "acl_prof_task_time_level": "",
        "timelimit": max(1, int(duration) + 2),
        "domain": "Request;KVCache;ModelExecute;BatchSchedule;Communication",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _set_service_profile_enabled(path: Path, enabled: bool) -> None:
    payload = json.loads(path.read_text())
    payload["enable"] = int(enabled)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _start_services(
    args: argparse.Namespace,
    output: Path,
) -> list[ProfileService]:
    services: list[ProfileService] = []
    for index, devices in enumerate(args.devices):
        instance_id = f"profile-{index}"
        port = args.port + index
        profile_directory = output / args.profiler / instance_id
        trace_path = output / "algorithm-traces" / f"{instance_id}.jsonl"
        log_path = output / "logs" / f"{instance_id}.log"
        for path in (profile_directory, trace_path.parent, log_path.parent):
            path.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "inference_scaling.swe_agent.server",
            "--config",
            str(Path(args.config).resolve()),
            "--host",
            args.host,
            "--port",
            str(port),
            "--set",
            f"vllm.tensor_parallel_size={args.tensor_parallel_size}",
            "--set",
            f"vllm.pipeline_parallel_size={args.pipeline_parallel_size}",
            "--set",
            f"service.trace_path={json.dumps(str(trace_path))}",
        ]
        service_config = None
        if args.profiler == "torch":
            profiler_config = {
                "profiler": "torch",
                "torch_profiler_dir": str(profile_directory),
                "torch_profiler_with_stack": args.profile_stack,
                "torch_profiler_with_memory": args.profile_memory,
                "ignore_frontend": False,
            }
            command.extend(
                (
                    "--set",
                    "vllm.engine_kwargs.profiler_config="
                    + json.dumps(profiler_config, separators=(",", ":")),
                )
            )
        elif args.profiler == "service":
            service_config = output / "service-configs" / f"{instance_id}.json"
            _write_service_config(
                service_config, profile_directory, args.profile_seconds
            )
        for override in args.overrides:
            command.extend(("--set", override))
        environment = dict(os.environ)
        environment["ASCEND_RT_VISIBLE_DEVICES"] = devices
        environment["CIS_INSTANCE_ID"] = instance_id
        for assignment in args.environment:
            if "=" not in assignment:
                raise ValueError(
                    f"environment override requires KEY=VALUE: {assignment}"
                )
            key, value = assignment.split("=", 1)
            if value == "__UNSET__":
                environment.pop(key, None)
            else:
                environment[key] = value
        if service_config is not None:
            environment["SERVICE_PROF_CONFIG_PATH"] = str(service_config)
            if args.profiling_symbols:
                environment["PROFILING_SYMBOLS_PATH"] = str(
                    Path(args.profiling_symbols).resolve()
                )
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        services.append(
            ProfileService(
                instance_id,
                devices,
                f"http://127.0.0.1:{port}",
                process,
                log,
                trace_path,
                profile_directory,
                service_config,
            )
        )
    return services


def _stop_services(services: Sequence[ProfileService]) -> None:
    for service in services:
        _stop(service.process)
        service.log.close()


def _profile_lifecycle(
    args: argparse.Namespace,
    services: Sequence[ProfileService],
    telemetry: list[dict[str, Any]],
    window: dict[str, Any],
) -> None:
    time.sleep(args.ramp_up_seconds)
    if args.profiler == "torch":
        for service in services:
            _post_control(
                service.endpoint,
                "start",
                {"prefix": f"{args.profile_prefix}-{service.instance_id}"},
                args.profile_stop_timeout,
            )
    elif args.profiler == "service":
        for service in services:
            assert service.service_profile_config is not None
            _set_service_profile_enabled(service.service_profile_config, True)
    window["started_at"] = time.time()
    deadline = time.monotonic() + args.profile_seconds
    while time.monotonic() < deadline:
        sample = {
            "timestamp": time.time(),
            "instances": {
                service.instance_id: _backend_snapshot(service.endpoint)
                for service in services
            },
        }
        telemetry.append(sample)
        time.sleep(min(args.telemetry_interval, max(0.0, deadline - time.monotonic())))
    window["finished_at"] = time.time()
    if args.profiler == "torch":
        for service in services:
            _post_control(
                service.endpoint,
                "stop",
                {},
                args.profile_stop_timeout,
            )
    elif args.profiler == "service":
        for service in services:
            assert service.service_profile_config is not None
            _set_service_profile_enabled(service.service_profile_config, False)


def _analyze_service_profiles(
    services: Sequence[ProfileService],
) -> list[dict[str, Any]]:
    analyses = []
    executable = "msserviceprofiler"
    for service in services:
        analyzed = service.profile_directory / "analyzed"
        command = [
            executable,
            "analyze",
            f"--input-path={service.profile_directory}",
            f"--output-path={analyzed}",
        ]
        try:
            completed = subprocess.run(
                command, text=True, capture_output=True, timeout=1800
            )
            analyses.append(
                {
                    "instance_id": service.instance_id,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "output": str(analyzed),
                }
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            analyses.append(
                {
                    "instance_id": service.instance_id,
                    "command": command,
                    "error": f"{type(error).__name__}: {error}",
                    "output": str(analyzed),
                }
            )
    return analyses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--warmup-workload")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--devices", action="append")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument(
        "--routing", choices=("round_robin", "least_outstanding"), default="round_robin"
    )
    parser.add_argument(
        "--profiler", choices=("none", "torch", "service"), required=True
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--profile-stop-timeout", type=float, default=1800.0)
    parser.add_argument("--ramp-up-seconds", type=float, default=10.0)
    parser.add_argument("--profile-seconds", type=float, default=30.0)
    parser.add_argument("--telemetry-interval", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--profile-prefix", default="conditional-is")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-stack", action="store_true")
    parser.add_argument("--profiling-symbols")
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--rollout-count", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--env", dest="environment", action="append", default=[])
    args = parser.parse_args()
    args.devices = args.devices or ["0,1"]
    if args.ramp_up_seconds < 0 or args.profile_seconds <= 0:
        raise ValueError("profiling windows require ramp >= 0 and duration > 0")

    output = Path(args.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiler_dependencies = _profile_preflight(args, output)
    records = _load_records(Path(args.workload))[: args.limit]
    if not records:
        raise ValueError("profiling requires at least one workload record")
    if len(records) < args.workers:
        raise ValueError(
            "profile workload must contain at least one request per worker"
        )
    warmup = _load_records(Path(args.warmup_workload)) if args.warmup_workload else []
    services = _start_services(args, output)
    endpoints = tuple(service.endpoint for service in services)
    started_at = time.time()
    telemetry: list[dict[str, Any]] = []
    window: dict[str, Any] = {
        "ramp_up_seconds": args.ramp_up_seconds,
        "requested_seconds": args.profile_seconds,
    }
    try:
        for service in services:
            _wait_for_health(service.endpoint, service.process, args.startup_timeout)
        if warmup:
            warmup_result = run_burst(
                warmup,
                endpoints,
                workers=min(args.workers, len(warmup)),
                timeout=args.request_timeout,
                seed=args.seed,
                conditional_overrides=_conditional_overrides(args),
                routing=args.routing,
            )
            if warmup_result["success_rate"] != 1.0:
                raise RuntimeError("profiling warmup failed")
        lifecycle = (
            None
            if args.profiler == "none"
            else lambda: _profile_lifecycle(args, services, telemetry, window)
        )
        result = run_burst(
            records,
            endpoints,
            workers=args.workers,
            timeout=args.request_timeout,
            seed=args.seed,
            conditional_overrides=_conditional_overrides(args),
            routing=args.routing,
            after_release=lifecycle,
        )
    finally:
        _stop_services(services)

    service_analysis = (
        _analyze_service_profiles(services) if args.profiler == "service" else []
    )
    result.update(
        {
            "profile": {
                "kind": args.profiler,
                "window": window,
                "memory": args.profile_memory,
                "stack": args.profile_stack,
                "services": [
                    {
                        "instance_id": service.instance_id,
                        "devices": service.devices,
                        "endpoint": service.endpoint,
                        "profile_directory": str(service.profile_directory),
                        "algorithm_trace": str(service.trace_path),
                    }
                    for service in services
                ],
                "service_analysis": service_analysis,
                "dependencies": profiler_dependencies,
            },
            "started_at": started_at,
            "topology": {
                "instances": len(services),
                "tensor_parallel_size": args.tensor_parallel_size,
                "pipeline_parallel_size": args.pipeline_parallel_size,
                "routing": args.routing,
            },
            "overrides": list(args.overrides),
            "environment": list(args.environment),
        }
    )
    (output / "benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output / "telemetry.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in telemetry),
        encoding="utf-8",
    )
    write_artifact_manifest(
        output,
        repository=Path(__file__).resolve().parents[3],
        command=sys.argv,
        metadata={
            "profiler": args.profiler,
            "profiler_dependencies": profiler_dependencies,
            "profile_window": window,
            "devices": args.devices,
            "topology": result["topology"],
            "algorithm": _conditional_overrides(args),
            "config": {
                "path": str(Path(args.config).resolve()),
                "sha256": sha256_file(Path(args.config).resolve()),
            },
            "workload": {
                "path": str(Path(args.workload).resolve()),
                "sha256": sha256_file(Path(args.workload).resolve()),
                "records": len(records),
            },
            "seed": args.seed,
        },
    )
    print(
        json.dumps(
            {
                "success_rate": result["success_rate"],
                "jobs_per_second": result["jobs_per_second"],
                "p95": result["latency_seconds"]["p95"],
                "profile": args.profiler,
                "window": window,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
