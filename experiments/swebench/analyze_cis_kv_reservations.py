#!/usr/bin/env python3
"""Compare conservative CIS KV reservations with realized branch lengths."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = list(values)
    return {
        "count": len(samples),
        "mean": statistics.fmean(samples) if samples else None,
        "p50": _percentile(samples, 0.50),
        "p90": _percentile(samples, 0.90),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
        "max": max(samples, default=None),
    }


def _allocated(tokens: int, block_size: int) -> int:
    return math.ceil(tokens / block_size) * block_size if tokens > 0 else 0


def analyze_trace(
    path: Path,
    *,
    total_length: int,
    rollout_count: int,
    kv_block_size: int,
    include_warmup: bool = False,
) -> dict[str, Any]:
    conservative: list[float] = []
    realized: list[float] = []
    ratios: list[float] = []
    candidate_lengths: list[float] = []
    rollout_lengths: list[float] = []
    terminal_fractions: list[float] = []
    requests = 0

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            request_id = str(record.get("request_id", ""))
            if not include_warmup and ":warmup:" in request_id:
                continue
            requests += 1
            candidate_events = {
                int(event["step"]): event
                for event in record.get("stage_events", ())
                if event.get("name") == "candidate" and "step" in event
            }
            for step in record.get("conditional_steps", ()):
                candidates = step.get("candidates", ())
                if not candidates:
                    continue
                step_index = int(step.get("block_id", 0))
                event = candidate_events.get(step_index)
                if event is None:
                    continue
                prefix_tokens = int(event.get("prefix_tokens", 0))
                generated_before = int(step.get("generated_tokens_before", 0))
                remaining = max(0, total_length - generated_before)
                candidate_max = min(
                    int(event.get("block_length", remaining)), remaining
                )
                rollout_max = max(0, remaining - candidate_max)
                candidate_count = len(candidates)
                worst_case = (
                    _allocated(prefix_tokens, kv_block_size)
                    + candidate_count * _allocated(candidate_max, kv_block_size)
                    + candidate_count
                    * rollout_count
                    * _allocated(rollout_max, kv_block_size)
                )

                candidate_peak = _allocated(prefix_tokens, kv_block_size)
                terminal = 0
                for candidate in candidates:
                    output_tokens = int(candidate.get("output_tokens", 0))
                    candidate_lengths.append(output_tokens)
                    candidate_peak += _allocated(output_tokens, kv_block_size)
                    terminal += bool(candidate.get("terminal"))
                rollout_peak = candidate_peak
                for candidate in candidates:
                    for rollout in candidate.get("rollouts", ()):
                        output_tokens = int(rollout.get("output_tokens", 0))
                        rollout_lengths.append(output_tokens)
                        rollout_peak += _allocated(output_tokens, kv_block_size)

                actual = max(candidate_peak, rollout_peak)
                conservative.append(worst_case)
                realized.append(actual)
                ratios.append(worst_case / actual if actual else 1.0)
                terminal_fractions.append(terminal / candidate_count)

    return {
        "schema_version": 1,
        "trace": str(path),
        "parameters": {
            "total_length": total_length,
            "rollout_count": rollout_count,
            "kv_block_size": kv_block_size,
            "include_warmup": include_warmup,
        },
        "requests": requests,
        "steps": len(conservative),
        "conservative_tokens": _distribution(conservative),
        "realized_peak_tokens": _distribution(realized),
        "overreservation_ratio": _distribution(ratios),
        "terminal_candidate_fraction": _distribution(terminal_fractions),
        "candidate_output_tokens": _distribution(candidate_lengths),
        "rollout_output_tokens": _distribution(rollout_lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--total-length", type=int, default=512)
    parser.add_argument("--rollout-count", type=int, default=3)
    parser.add_argument("--kv-block-size", type=int, default=128)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("total_length", "rollout_count", "kv_block_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    payload = {
        path.parent.parent.name: analyze_trace(
            path,
            total_length=args.total_length,
            rollout_count=args.rollout_count,
            kv_block_size=args.kv_block_size,
            include_warmup=args.include_warmup,
        )
        for path in args.traces
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
