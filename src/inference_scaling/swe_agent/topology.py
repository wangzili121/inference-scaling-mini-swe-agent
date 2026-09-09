"""Compare the specified four-card, two-instance CIS deployments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence, TextIO

from inference_scaling.swe_agent.benchmark import _load_records, run_burst
from inference_scaling.swe_agent.runtime_tune import _stop, _wait_for_health


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    instance_id: str
    devices: str
    port: int
    tensor_parallel_size: int
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1

    def overrides(self, trace_path: str) -> tuple[str, ...]:
        return (
            f"vllm.tensor_parallel_size={self.tensor_parallel_size}",
            f"vllm.pipeline_parallel_size={self.pipeline_parallel_size}",
            f"vllm.data_parallel_size={self.data_parallel_size}",
            f"service.trace_path={json.dumps(trace_path)}",
        )


@dataclass(frozen=True, slots=True)
class TopologySpec:
    topology_id: str
    services: tuple[ServiceSpec, ...]
    routing: str
    verification_status: str = "pending_capability_smoke"


@dataclass(slots=True)
class RunningService:
    process: subprocess.Popen
    endpoint: str
    log: TextIO


def _paired_services(
    base_port: int,
    *,
    topology: str,
    device_pairs: Sequence[str] = ("0,1", "2,3"),
) -> tuple[ServiceSpec, ServiceSpec]:
    if len(device_pairs) != 2:
        raise ValueError("four-card comparison requires exactly two device pairs")
    if topology == "tp2":
        parallel = {"tensor_parallel_size": 2, "pipeline_parallel_size": 1}
    elif topology == "pp2":
        parallel = {"tensor_parallel_size": 1, "pipeline_parallel_size": 2}
    else:
        raise ValueError(f"unsupported paired topology: {topology}")
    return (
        ServiceSpec(f"{topology}-a", str(device_pairs[0]), base_port, **parallel),
        ServiceSpec(f"{topology}-b", str(device_pairs[1]), base_port + 1, **parallel),
    )


def native_topologies(
    base_port: int = 18123,
    device_pairs: Sequence[str] = ("0,1", "2,3"),
) -> tuple[TopologySpec, ...]:
    return tuple(
        TopologySpec(
            f"2x{parallel}-{routing.replace('_', '-')}",
            _paired_services(base_port, topology=parallel, device_pairs=device_pairs),
            routing,
            "ready_for_runtime_smoke",
        )
        for parallel in ("tp2", "pp2")
        for routing in ("round_robin", "least_outstanding")
    )


def capability_matrix(
    base_port: int = 18123,
    device_pairs: Sequence[str] = ("0,1", "2,3"),
) -> dict[str, Any]:
    return {
        "specified_two_instance": [
            asdict(spec) for spec in native_topologies(base_port, device_pairs)
        ],
        "excluded_from_primary_comparison": {
            "tp4": "single instance; outside the requested four-card comparison",
            "tp2-pp2": "single instance; outside the requested four-card comparison",
            "dp2-shared-queue": "not two independently observable service instances",
        },
        "gated": {
            "pd-2-plus-2": {
                "status": "requires_v018_ascend_kv_connector_capability_smoke",
                "reason": "P/D is an intra-generation KV-transfer topology, not two ordinary CIS endpoints.",
            },
            "candidate-rollout-stage-pipeline": {
                "status": "requires_algorithm_stage_rpc_backend",
                "reason": "Candidate and rollout batches must be routed independently while preserving one CIS job.",
            },
        },
    }


def _start_services(
    topology: TopologySpec,
    *,
    config: Path,
    output: Path,
    host: str,
    overrides: Sequence[str] = (),
    environment_overrides: Sequence[str] = (),
) -> list[RunningService]:
    processes = []
    for service in topology.services:
        trace_path = str(
            output / "traces" / topology.topology_id / f"{service.instance_id}.jsonl"
        )
        command = [
            sys.executable,
            "-m",
            "inference_scaling.swe_agent.server",
            "--config",
            str(config),
            "--host",
            host,
            "--port",
            str(service.port),
        ]
        for override in service.overrides(trace_path):
            command.extend(("--set", override))
        for override in overrides:
            command.extend(("--set", override))
        environment = dict(os.environ)
        environment["ASCEND_RT_VISIBLE_DEVICES"] = service.devices
        environment["CIS_INSTANCE_ID"] = service.instance_id
        for assignment in environment_overrides:
            if "=" not in assignment:
                raise ValueError(
                    f"environment override requires KEY=VALUE: {assignment}"
                )
            key, value = assignment.split("=", 1)
            if value == "__UNSET__":
                environment.pop(key, None)
            else:
                environment[key] = value
        log_path = output / "logs" / topology.topology_id / f"{service.instance_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        processes.append(
            RunningService(process, f"http://127.0.0.1:{service.port}", log)
        )
    return processes


def _stop_services(processes: Sequence[RunningService]) -> None:
    for service in processes:
        _stop(service.process)
        service.log.close()


def run_topology(
    topology: TopologySpec,
    records: Sequence[dict[str, Any]],
    *,
    warmup_records: Sequence[dict[str, Any]],
    config: Path,
    output: Path,
    workers: int,
    startup_timeout: float,
    request_timeout: float,
    seed: int,
    host: str = "0.0.0.0",
    overrides: Sequence[str] = (),
    environment_overrides: Sequence[str] = (),
    conditional_overrides: dict[str, int] | None = None,
) -> dict[str, Any]:
    processes = _start_services(
        topology,
        config=config,
        output=output,
        host=host,
        overrides=overrides,
        environment_overrides=environment_overrides,
    )
    try:
        for service in processes:
            _wait_for_health(service.endpoint, service.process, startup_timeout)
        endpoints = tuple(service.endpoint for service in processes)
        if warmup_records:
            warmup = run_burst(
                warmup_records,
                endpoints,
                workers=min(workers, len(warmup_records)),
                timeout=request_timeout,
                seed=seed,
                routing=topology.routing,
                conditional_overrides=conditional_overrides,
            )
            if warmup["success_rate"] != 1.0:
                raise RuntimeError("topology warmup workload failed")
        result = run_burst(
            records,
            endpoints,
            workers=workers,
            timeout=request_timeout,
            seed=seed,
            routing=topology.routing,
            conditional_overrides=conditional_overrides,
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
        _stop_services(processes)
    result.update(
        {
            "topology_id": topology.topology_id,
            "routing": topology.routing,
            "services": [asdict(service) for service in topology.services],
        }
    )
    return result


def run_routing_comparison(
    topologies: Sequence[TopologySpec],
    records: Sequence[dict[str, Any]],
    *,
    warmup_records: Sequence[dict[str, Any]],
    config: Path,
    output: Path,
    workers: int,
    startup_timeout: float,
    request_timeout: float,
    seed: int,
    host: str = "0.0.0.0",
    overrides: Sequence[str] = (),
    environment_overrides: Sequence[str] = (),
    conditional_overrides: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Compare routing policies while reusing the same initialized services."""

    if not topologies:
        raise ValueError("routing comparison requires at least one topology")
    services = topologies[0].services
    if any(topology.services != services for topology in topologies[1:]):
        raise ValueError("routing comparison requires identical service layouts")
    processes = _start_services(
        topologies[0],
        config=config,
        output=output,
        host=host,
        overrides=overrides,
        environment_overrides=environment_overrides,
    )
    results: list[dict[str, Any]] = []
    try:
        for service in processes:
            _wait_for_health(service.endpoint, service.process, startup_timeout)
        endpoints = tuple(service.endpoint for service in processes)
        if warmup_records:
            warmup = run_burst(
                warmup_records,
                endpoints,
                workers=min(workers, len(warmup_records)),
                timeout=request_timeout,
                seed=seed,
                routing=topologies[0].routing,
                conditional_overrides=conditional_overrides,
            )
            if warmup["success_rate"] != 1.0:
                raise RuntimeError("topology warmup workload failed")
        for topology in topologies:
            result = run_burst(
                records,
                endpoints,
                workers=workers,
                timeout=request_timeout,
                seed=seed,
                routing=topology.routing,
                conditional_overrides=conditional_overrides,
            )
            result.update(
                {
                    "topology_id": topology.topology_id,
                    "routing": topology.routing,
                    "services": [asdict(service) for service in topology.services],
                    "shared_service_lifecycle": True,
                }
            )
            results.append(result)
    finally:
        _stop_services(processes)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--warmup-workload")
    parser.add_argument("--topology", action="append")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--base-port", type=int, default=18123)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--capabilities-only", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--env", dest="environment", action="append", default=[])
    parser.add_argument("--instance-devices", action="append")
    parser.add_argument("--candidate-count", type=int, default=15)
    parser.add_argument("--rollout-count", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    device_pairs = args.instance_devices or ["0,1", "2,3"]
    capabilities = capability_matrix(args.base_port, device_pairs)
    (output / "capabilities.json").write_text(
        json.dumps(capabilities, indent=2) + "\n", encoding="utf-8"
    )
    if args.capabilities_only:
        print(json.dumps(capabilities, indent=2))
        return
    records = _load_records(Path(args.workload))[: args.limit]
    warmup_records = (
        _load_records(Path(args.warmup_workload)) if args.warmup_workload else []
    )
    requested = set(args.topology or ())
    topologies = [
        topology
        for topology in native_topologies(args.base_port, device_pairs)
        if not requested or topology.topology_id in requested
    ]
    if requested - {topology.topology_id for topology in topologies}:
        raise ValueError("requested topology is not natively executable")
    shared_layout = bool(topologies) and all(
        topology.services == topologies[0].services for topology in topologies[1:]
    )
    common = {
        "warmup_records": warmup_records,
        "config": Path(args.config).resolve(),
        "output": output,
        "workers": args.workers,
        "startup_timeout": args.startup_timeout,
        "request_timeout": args.request_timeout,
        "seed": args.seed,
        "overrides": args.overrides,
        "environment_overrides": args.environment,
        "conditional_overrides": {
            "candidate_count": args.candidate_count,
            "rollout_count": args.rollout_count,
            "block_size": args.block_size,
        },
    }
    if shared_layout:
        results = run_routing_comparison(topologies, records, **common)
    else:
        results = [
            run_topology(topology, records, **common) for topology in topologies
        ]
    payload = {"schema_version": 1, "capabilities": capabilities, "results": results}
    (output / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            [
                {
                    "topology_id": result["topology_id"],
                    "success_rate": result["success_rate"],
                    "jobs_per_second": result["jobs_per_second"],
                    "p95": result["latency_seconds"]["p95"],
                }
                for result in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
