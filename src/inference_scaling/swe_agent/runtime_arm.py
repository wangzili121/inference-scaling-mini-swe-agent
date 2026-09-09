"""Run one reproducible Conditional IS runtime-tuning arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inference_scaling.swe_agent.artifacts import (
    sha256_file,
    source_revision,
    write_artifact_manifest,
)
from inference_scaling.swe_agent.benchmark import _load_records
from inference_scaling.swe_agent.runtime_tune import (
    BASE_FEATURES,
    EngineConfig,
    _cached_arm,
    select_max_model_len,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--warmup-workload")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--phase", default="focused")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--topology", choices=("tp2", "pp2"), default="tp2")
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-num-partial-prefills", type=int, default=1)
    parser.add_argument("--max-long-partial-prefills", type=int, default=1)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--candidate-count", type=int, default=15)
    parser.add_argument("--rollout-count", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[3]
    config_path = Path(args.config).resolve()
    workload_path = Path(args.workload).resolve()
    warmup_path = (
        Path(args.warmup_workload).resolve() if args.warmup_workload else None
    )
    records = _load_records(workload_path)
    if args.requests <= 0 or args.requests > len(records):
        raise ValueError("requests must be within the available workload records")
    selected_records = records[: args.requests]
    warmup_records = _load_records(warmup_path) if warmup_path else []
    max_model_len = args.max_model_len or select_max_model_len(records)
    engine = EngineConfig(
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        topology=args.topology,
        max_num_partial_prefills=args.max_num_partial_prefills,
        max_long_partial_prefills=args.max_long_partial_prefills,
        max_model_len=max_model_len,
    )
    output = Path(args.output_directory).resolve()
    result = _cached_arm(
        output,
        args.phase,
        resume=args.resume,
        config=engine,
        feature=BASE_FEATURES,
        workers=args.workers,
        records=selected_records,
        warmup_records=warmup_records,
        service_config=config_path,
        endpoint=f"http://{args.host}:{args.port}",
        port=args.port,
        devices=args.devices,
        startup_timeout=args.startup_timeout,
        request_timeout=args.request_timeout,
        seed=args.seed,
        conditional_overrides={
            "candidate_count": args.candidate_count,
            "rollout_count": args.rollout_count,
            "block_size": args.block_size,
        },
    )
    metadata = {
        "kind": "two-card-runtime-arm",
        "devices": args.devices,
        "source_revision": source_revision(repository),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "workload": {
            "path": str(workload_path),
            "sha256": sha256_file(workload_path),
            "records": args.requests,
        },
        "warmup_workload": (
            {"path": str(warmup_path), "sha256": sha256_file(warmup_path)}
            if warmup_path
            else None
        ),
        "arm": result["arm_id"],
    }
    write_artifact_manifest(
        output,
        repository=repository,
        command=sys.argv,
        metadata=metadata,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
