"""Tune two copied two-card instances without changing CIS stage ownership."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from inference_scaling.swe_agent.benchmark import _load_records
from inference_scaling.swe_agent.runtime_tune import (
    EngineConfig,
    FeatureConfig,
    _valid,
    select_arms,
)
from inference_scaling.swe_agent.topology import (
    TopologySpec,
    native_topologies,
    run_topology,
)


def _environment(feature: FeatureConfig) -> tuple[str, ...]:
    return tuple(
        f"{key}={'__UNSET__' if value is None else value}"
        for key, value in feature.environment
    )


def _best(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected = select_arms(results, keep=1)
    if not selected:
        raise RuntimeError("no four-card configuration passed the hard gates")
    return selected[0]


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cached_run(
    output: Path,
    phase: str,
    *,
    topology: TopologySpec,
    engine: EngineConfig,
    feature: FeatureConfig,
    workers: int,
    records: Sequence[dict[str, Any]],
    warmup_records: Sequence[dict[str, Any]],
    config: Path,
    startup_timeout: float,
    request_timeout: float,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    stem = f"{topology.topology_id}-{engine.config_id}-w{workers}-n{len(records)}"
    path = output / "arms" / phase / f"{stem}.json"
    if resume and path.exists():
        return json.loads(path.read_text())
    result = run_topology(
        topology,
        records,
        warmup_records=warmup_records,
        config=config,
        output=output / "runs" / phase / stem,
        workers=workers,
        startup_timeout=startup_timeout,
        request_timeout=request_timeout,
        seed=seed,
        overrides=(*engine.overrides(), *feature.overrides),
        environment_overrides=_environment(feature),
        conditional_overrides={
            "candidate_count": 15,
            "rollout_count": 3,
            "block_size": 128,
        },
    )
    result.update(
        {
            "arm_id": stem,
            "engine": asdict(engine),
            "feature": asdict(feature),
            "workers": workers,
        }
    )
    _write(path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--two-card-result", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--holdout-workload")
    parser.add_argument("--warmup-workload")
    parser.add_argument("--instance-devices", action="append", required=True)
    parser.add_argument("--base-port", type=int, default=18123)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if len(args.instance_devices) != 2:
        parser.error("provide exactly two --instance-devices pairs")

    source = json.loads(Path(args.two_card_result).read_text())
    two_winner = source.get("winner")
    if not isinstance(two_winner, dict):
        raise ValueError("two-card result has no winner")
    engine = EngineConfig(**two_winner["engine"])
    feature = FeatureConfig(**two_winner["feature"])
    records = _load_records(Path(args.workload))
    if len(records) < 64:
        raise ValueError("four-card tuning requires at least 64 unique calls")
    holdout = (
        _load_records(Path(args.holdout_workload)) if args.holdout_workload else []
    )
    warmup = _load_records(Path(args.warmup_workload)) if args.warmup_workload else []
    output = Path(args.output_directory)
    config = Path(args.config).resolve()
    all_topologies = {
        item.topology_id: item
        for item in native_topologies(args.base_port, args.instance_devices)
        if item.topology_id.startswith(f"2x{engine.topology}-")
    }
    common = {
        "feature": feature,
        "warmup_records": warmup,
        "config": config,
        "startup_timeout": args.startup_timeout,
        "request_timeout": args.request_timeout,
        "seed": args.seed,
        "resume": args.resume,
    }
    history: list[dict[str, Any]] = []

    initial_workers = min(64, max(8, int(two_winner["workers"]) * 2))
    route_results = [
        _cached_run(
            output,
            "routing",
            topology=topology,
            engine=engine,
            workers=initial_workers,
            records=records[:64],
            **common,
        )
        for topology in all_topologies.values()
    ]
    history.extend(route_results)
    selected = _best(route_results)
    topology = all_topologies[selected["topology_id"]]

    worker_values = [8, 16, 32, 64]
    worker_values.extend(value for value in (96, 128, 256) if len(records) >= value)
    worker_results = [
        _cached_run(
            output,
            "workers",
            topology=topology,
            engine=engine,
            workers=worker,
            records=records[: max(64, worker)],
            **common,
        )
        for worker in worker_values
    ]
    history.extend(worker_results)
    selected = _best(worker_results)
    workers = int(selected["workers"])

    mns_values = sorted(
        {
            max(64, engine.max_num_seqs // 2),
            engine.max_num_seqs,
            min(2048, engine.max_num_seqs * 3 // 2),
            min(2048, engine.max_num_seqs * 2),
        }
    )
    mbt_values = sorted(
        {
            max(8192, engine.max_num_batched_tokens // 2),
            engine.max_num_batched_tokens,
            min(524288, engine.max_num_batched_tokens * 2),
        }
    )
    engine_results = []
    for value in mns_values:
        engine_results.append(
            _cached_run(
                output,
                "mns",
                topology=topology,
                engine=replace(engine, max_num_seqs=value),
                workers=workers,
                records=records[: max(64, workers)],
                **common,
            )
        )
    engine = EngineConfig(**_best(engine_results)["engine"])
    for value in mbt_values:
        engine_results.append(
            _cached_run(
                output,
                "mbt",
                topology=topology,
                engine=replace(engine, max_num_batched_tokens=value),
                workers=workers,
                records=records[: max(64, workers)],
                **common,
            )
        )
    history.extend(engine_results)
    engine = EngineConfig(**_best(engine_results)["engine"])

    partial_results = [
        _cached_run(
            output,
            "partial-prefill",
            topology=topology,
            engine=replace(
                engine,
                max_num_partial_prefills=partial,
                max_long_partial_prefills=long_partial,
            ),
            workers=workers,
            records=records[: max(64, workers)],
            **common,
        )
        for partial, long_partial in (
            (1, 1),
            (2, 1),
            (2, 2),
            (4, 2),
            (4, 4),
            (8, 4),
            (8, 8),
        )
    ]
    history.extend(partial_results)
    finalist = _best(partial_results)

    holdout_result = None
    if holdout:
        holdout_result = _cached_run(
            output,
            "holdout",
            topology=topology,
            engine=EngineConfig(**finalist["engine"]),
            feature=feature,
            workers=workers,
            records=holdout[:64],
            warmup_records=warmup,
            config=config,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
            seed=args.seed,
            resume=args.resume,
        )
        if not _valid(holdout_result):
            raise RuntimeError("four-card winner failed holdout hard gates")
        history.append(holdout_result)

    winner = holdout_result or finalist
    winner_engine = EngineConfig(**winner["engine"])
    tp, pp = (2, 1) if winner_engine.topology == "tp2" else (1, 2)
    deployment = {
        "schema_version": 1,
        "id": "four-card-best",
        "devices": list(args.instance_devices),
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "routing": topology.routing,
        "workers": workers,
        "limit": max(64, workers),
        "overrides": list(winner_engine.overrides()) + list(feature.overrides),
        "environment": list(_environment(feature)),
        "selected_engine": winner["engine"],
        "selected_features": winner["feature"],
    }
    result = {
        "schema_version": 1,
        "source_two_card_result": str(Path(args.two_card_result).resolve()),
        "history": history,
        "winner": winner,
        "deployment": deployment,
    }
    _write(output / "result.json", result)
    _write(output / "deployment.json", deployment)
    print(json.dumps({"winner": winner, "deployment": deployment}, indent=2))


if __name__ == "__main__":
    main()
