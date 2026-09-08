"""Compare native four-card vLLM topologies on one fixed CIS workload."""

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


def native_topologies(base_port: int = 18123) -> tuple[TopologySpec, ...]:
    return (
        TopologySpec(
            "2xtp2-whole-job",
            (
                ServiceSpec("tp2-a", "0,1", base_port, 2),
                ServiceSpec("tp2-b", "2,3", base_port + 1, 2),
            ),
            "round_robin_whole_job",
            "ready_for_runtime_smoke",
        ),
        TopologySpec(
            "tp2-dp2-shared-queue",
            (ServiceSpec("tp2-dp2", "0,1,2,3", base_port, 2, data_parallel_size=2),),
            "vllm_data_parallel_shared_request_queue",
        ),
        TopologySpec(
            "tp4",
            (ServiceSpec("tp4", "0,1,2,3", base_port, 4),),
            "single_engine",
        ),
        TopologySpec(
            "tp2-pp2",
            (
                ServiceSpec(
                    "tp2-pp2", "0,1,2,3", base_port, 2, pipeline_parallel_size=2
                ),
            ),
            "single_engine",
        ),
    )


def capability_matrix(base_port: int = 18123) -> dict[str, Any]:
    return {
        "configurable": [asdict(spec) for spec in native_topologies(base_port)],
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
        environment = dict(os.environ)
        environment["ASCEND_RT_VISIBLE_DEVICES"] = service.devices
        environment["CIS_INSTANCE_ID"] = service.instance_id
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
) -> dict[str, Any]:
    processes = _start_services(topology, config=config, output=output, host=host)
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
            )
            if warmup["success_rate"] != 1.0:
                raise RuntimeError("topology warmup workload failed")
        result = run_burst(
            records,
            endpoints,
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
        _stop_services(processes)
    result.update(
        {
            "topology_id": topology.topology_id,
            "routing": topology.routing,
            "services": [asdict(service) for service in topology.services],
        }
    )
    return result


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
    args = parser.parse_args()

    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    capabilities = capability_matrix(args.base_port)
    (output / "capabilities.json").write_text(
        json.dumps(capabilities, indent=2) + "\n", encoding="utf-8"
    )
    if args.capabilities_only:
        print(json.dumps(capabilities, indent=2))
        return
    records = _load_records(Path(args.workload))[: args.limit]
    warmup_records = (
        _load_records(Path(args.warmup_workload))
        if args.warmup_workload
        else []
    )
    requested = set(args.topology or ())
    topologies = [
        topology
        for topology in native_topologies(args.base_port)
        if not requested or topology.topology_id in requested
    ]
    if requested - {topology.topology_id for topology in topologies}:
        raise ValueError("requested topology is not natively executable")
    results = [
        run_topology(
            topology,
            records,
            warmup_records=warmup_records,
            config=Path(args.config).resolve(),
            output=output,
            workers=args.workers,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
            seed=args.seed,
        )
        for topology in topologies
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
