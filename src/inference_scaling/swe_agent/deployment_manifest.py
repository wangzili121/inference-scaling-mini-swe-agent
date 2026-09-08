"""Materialize selected tuner output as an immutable profiling deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from inference_scaling.swe_agent.artifacts import sha256_file
from inference_scaling.swe_agent.runtime_tune import EngineConfig, FeatureConfig


def build_deployment_manifest(
    tuning_result: str | Path,
    *,
    deployment_id: str,
    devices: Sequence[str],
    routing: str,
    workers: int | None = None,
    limit: int = 64,
) -> dict[str, Any]:
    source = Path(tuning_result).resolve()
    payload = json.loads(source.read_text())
    winner = payload.get("winner")
    if not isinstance(winner, dict):
        raise ValueError("tuning result has no winner")
    engine = EngineConfig(**winner["engine"])
    feature = FeatureConfig(**winner["feature"])
    if routing not in {"round_robin", "least_outstanding"}:
        raise ValueError(f"unknown routing: {routing}")
    if len(devices) not in {1, 2}:
        raise ValueError("profiling deployment requires one or two instances")
    tp, pp = (2, 1) if engine.topology == "tp2" else (1, 2)
    environment = [
        f"{key}={'__UNSET__' if value is None else value}"
        for key, value in feature.environment
    ]
    return {
        "schema_version": 1,
        "id": deployment_id,
        "source_tuning_result": str(source),
        "source_tuning_result_sha256": sha256_file(source),
        "devices": list(devices),
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "routing": routing,
        "workers": int(workers or winner["workers"]),
        "limit": int(limit),
        "overrides": list(engine.overrides()) + list(feature.overrides),
        "environment": environment,
        "selected_engine": winner["engine"],
        "selected_features": winner["feature"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-result", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--devices", action="append", required=True)
    parser.add_argument(
        "--routing",
        choices=("round_robin", "least_outstanding"),
        default="round_robin",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_deployment_manifest(
        args.tuning_result,
        deployment_id=args.id,
        devices=args.devices,
        routing=args.routing,
        workers=args.workers,
        limit=args.limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
