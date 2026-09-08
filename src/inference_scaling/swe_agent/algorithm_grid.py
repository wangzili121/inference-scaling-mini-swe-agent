"""Benchmark the fixed Conditional IS C/R grid without reloading the model."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from inference_scaling.swe_agent.benchmark import _load_records, run_burst


def _forward_tokens(result: dict[str, Any]) -> int:
    counters = result.get("backend_delta") or {}
    return int(counters.get("generation_forward_token_slots", 0)) + int(
        counters.get("score_forward_token_slots", 0)
    )


def _median_ess_ratio(result: dict[str, Any], candidate_count: int) -> float:
    values = []
    for measurement in result["measurements"]:
        diagnostics = measurement.get("diagnostics") or {}
        values.extend(
            float(value) / candidate_count
            for value in diagnostics.get("candidate_ess", ())
        )
    return float(statistics.median(values)) if values else 0.0


def _pareto(results: Sequence[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    for candidate in results:
        dominated = False
        for other in results:
            if other is candidate:
                continue
            weakly_better = (
                other["jobs_per_second"] >= candidate["jobs_per_second"]
                and other["latency_seconds"]["p95"]
                <= candidate["latency_seconds"]["p95"]
                and other["forward_tokens"] <= candidate["forward_tokens"]
            )
            strictly_better = (
                other["jobs_per_second"] > candidate["jobs_per_second"]
                or other["latency_seconds"]["p95"] < candidate["latency_seconds"]["p95"]
                or other["forward_tokens"] < candidate["forward_tokens"]
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            selected.append(str(candidate["arm_id"]))
    return selected


def run_algorithm_grid(
    records: Sequence[dict[str, Any]],
    endpoints: Sequence[str],
    *,
    workers: int,
    candidate_counts: Sequence[int] = (4, 8, 15),
    rollout_counts: Sequence[int] = (2, 3),
    block_size: int = 128,
    seed: int = 20260908,
    timeout: float = 7200.0,
) -> dict[str, Any]:
    if not records:
        raise ValueError("algorithm grid requires at least one record")
    results = []
    for candidate_count in candidate_counts:
        for rollout_count in rollout_counts:
            result = run_burst(
                records,
                endpoints,
                workers=workers,
                timeout=timeout,
                seed=seed,
                conditional_overrides={
                    "candidate_count": int(candidate_count),
                    "rollout_count": int(rollout_count),
                    "block_size": int(block_size),
                },
            )
            result.update(
                {
                    "arm_id": f"C{candidate_count}-R{rollout_count}-B{block_size}",
                    "forward_tokens": _forward_tokens(result),
                    "median_ess_ratio": _median_ess_ratio(result, int(candidate_count)),
                }
            )
            results.append(result)
    valid = [result for result in results if result["success_rate"] == 1.0]
    return {
        "schema_version": 1,
        "requests": len(records),
        "workers": workers,
        "seed": seed,
        "arms": results,
        "pareto_arm_ids": _pareto(valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = _load_records(Path(args.workload))[: args.limit]
    result = run_algorithm_grid(
        records,
        args.endpoint,
        workers=args.workers,
        block_size=args.block_size,
        seed=args.seed,
        timeout=args.timeout,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pareto_arm_ids": result["pareto_arm_ids"]}, indent=2))


if __name__ == "__main__":
    main()
