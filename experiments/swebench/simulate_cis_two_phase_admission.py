#!/usr/bin/env python3
"""Model the safe admission headroom of two-phase CIS tree reservations.

This is a capacity model, not a throughput simulator. It answers whether a
candidate-phase tree can be admitted while retaining a safe completion order
for every admitted tree's worst-case rollout expansion. It also reports the
candidate-branch limit imposed by max_num_seqs so theoretical KV headroom is
not mistaken for executable concurrency.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class StepClaim:
    claim_id: str
    candidate_count: int
    candidate_tokens: int
    maximum_tokens: int
    realized_candidate_tokens: int
    post_candidate_max_tokens: int
    realized_peak_tokens: int


def _allocated(tokens: int, block_size: int) -> int:
    return math.ceil(tokens / block_size) * block_size if tokens > 0 else 0


def _percentile(values: Sequence[float], quantile: float) -> float | None:
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
        "p95": _percentile(samples, 0.95),
        "min": min(samples, default=None),
        "max": max(samples, default=None),
    }


def extract_step_claims(
    path: Path,
    *,
    total_length: int,
    rollout_count: int,
    kv_block_size: int,
    include_warmup: bool = False,
) -> list[StepClaim]:
    claims: list[StepClaim] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            request_id = str(record.get("request_id", ""))
            if not include_warmup and ":warmup:" in request_id:
                continue
            candidate_events = {
                int(event["step"]): event
                for event in record.get("stage_events", ())
                if event.get("name") == "candidate" and "step" in event
            }
            for step in record.get("conditional_steps", ()):
                candidates = step.get("candidates", ())
                step_index = int(step.get("block_id", 0))
                event = candidate_events.get(step_index)
                if not candidates or event is None:
                    continue
                generated_before = int(step.get("generated_tokens_before", 0))
                remaining = max(0, total_length - generated_before)
                candidate_max = min(
                    int(event.get("block_length", remaining)), remaining
                )
                rollout_max = max(0, remaining - candidate_max)
                prefix_tokens = int(event.get("prefix_tokens", 0))
                candidate_count = len(candidates)
                candidate_tokens = _allocated(prefix_tokens, kv_block_size)
                candidate_tokens += candidate_count * _allocated(
                    candidate_max, kv_block_size
                )
                maximum_tokens = candidate_tokens
                maximum_tokens += (
                    candidate_count
                    * rollout_count
                    * _allocated(rollout_max, kv_block_size)
                )

                realized_candidate_tokens = _allocated(
                    prefix_tokens, kv_block_size
                ) + sum(
                    _allocated(int(candidate.get("output_tokens", 0)), kv_block_size)
                    for candidate in candidates
                )
                rollouts = [
                    rollout
                    for candidate in candidates
                    for rollout in candidate.get("rollouts", ())
                ]
                post_candidate_max_tokens = realized_candidate_tokens + len(
                    rollouts
                ) * _allocated(rollout_max, kv_block_size)
                realized_peak_tokens = realized_candidate_tokens + sum(
                    _allocated(int(rollout.get("output_tokens", 0)), kv_block_size)
                    for rollout in rollouts
                )
                claims.append(
                    StepClaim(
                        claim_id=f"{request_id}:step:{step_index}",
                        candidate_count=candidate_count,
                        candidate_tokens=candidate_tokens,
                        maximum_tokens=maximum_tokens,
                        realized_candidate_tokens=realized_candidate_tokens,
                        post_candidate_max_tokens=post_candidate_max_tokens,
                        realized_peak_tokens=max(
                            realized_candidate_tokens, realized_peak_tokens
                        ),
                    )
                )
    return claims


def safe_completion_order(
    claims: Sequence[StepClaim], capacity_tokens: int
) -> list[str] | None:
    """Return a Banker-style completion order, or None for an unsafe state."""

    available = capacity_tokens - sum(claim.candidate_tokens for claim in claims)
    if available < 0:
        return None
    unfinished = list(claims)
    order: list[str] = []
    while unfinished:
        selected = next(
            (
                claim
                for claim in unfinished
                if claim.maximum_tokens - claim.candidate_tokens <= available
            ),
            None,
        )
        if selected is None:
            return None
        # The tree temporarily receives its remaining claim, completes, then
        # releases the entire maximum. Net available grows by its allocation.
        available += selected.candidate_tokens
        unfinished.remove(selected)
        order.append(selected.claim_id)
    return order


def admitted_frontier(
    claims: Sequence[StepClaim],
    *,
    capacity_tokens: int,
    mode: str,
    max_num_seqs: int | None = None,
) -> list[StepClaim]:
    admitted: list[StepClaim] = []
    candidate_sequences = 0
    for claim in claims:
        if (
            max_num_seqs is not None
            and candidate_sequences + claim.candidate_count > max_num_seqs
        ):
            break
        trial = [*admitted, claim]
        if mode == "full_reservation":
            fits = sum(item.maximum_tokens for item in trial) <= capacity_tokens
        elif mode == "two_phase_safe":
            fits = safe_completion_order(trial, capacity_tokens) is not None
        elif mode == "candidate_only":
            fits = sum(item.candidate_tokens for item in trial) <= capacity_tokens
        else:
            raise ValueError(f"unknown admission mode: {mode}")
        if not fits:
            break
        admitted.append(claim)
        candidate_sequences += claim.candidate_count
    return admitted


def _frontier_summary(
    claims: Sequence[StepClaim], capacity_tokens: int
) -> dict[str, Any]:
    return {
        "steps": len(claims),
        "candidate_sequences": sum(claim.candidate_count for claim in claims),
        "candidate_tokens": sum(claim.candidate_tokens for claim in claims),
        "maximum_tokens": sum(claim.maximum_tokens for claim in claims),
        "post_candidate_max_tokens": sum(
            claim.post_candidate_max_tokens for claim in claims
        ),
        "realized_peak_tokens": sum(claim.realized_peak_tokens for claim in claims),
        "maximum_overcommit_ratio": (
            sum(claim.maximum_tokens for claim in claims) / capacity_tokens
            if capacity_tokens
            else None
        ),
        "post_candidate_pressure_ratio": (
            sum(claim.post_candidate_max_tokens for claim in claims) / capacity_tokens
            if capacity_tokens
            else None
        ),
    }


def simulate_trace(
    claims: Sequence[StepClaim],
    *,
    capacity_tokens: int,
    max_num_seqs: int,
    shuffle_trials: int = 200,
    seed: int = 20260914,
) -> dict[str, Any]:
    if capacity_tokens <= 0 or max_num_seqs <= 0:
        raise ValueError("capacity and max_num_seqs must be positive")
    if shuffle_trials < 0:
        raise ValueError("shuffle_trials must not be negative")
    modes = ("full_reservation", "two_phase_safe", "candidate_only")
    fixed: dict[str, Any] = {}
    for mode in modes:
        kv_only = admitted_frontier(claims, capacity_tokens=capacity_tokens, mode=mode)
        execution_bound = admitted_frontier(
            claims,
            capacity_tokens=capacity_tokens,
            mode=mode,
            max_num_seqs=max_num_seqs,
        )
        fixed[mode] = {
            "kv_only": _frontier_summary(kv_only, capacity_tokens),
            "mns_bound": _frontier_summary(execution_bound, capacity_tokens),
        }

    rng = random.Random(seed)
    trial_counts: dict[str, list[float]] = {
        f"{mode}:{bound}": [] for mode in modes for bound in ("kv_only", "mns_bound")
    }
    for _ in range(shuffle_trials):
        shuffled = list(claims)
        rng.shuffle(shuffled)
        for mode in modes:
            trial_counts[f"{mode}:kv_only"].append(
                len(
                    admitted_frontier(
                        shuffled, capacity_tokens=capacity_tokens, mode=mode
                    )
                )
            )
            trial_counts[f"{mode}:mns_bound"].append(
                len(
                    admitted_frontier(
                        shuffled,
                        capacity_tokens=capacity_tokens,
                        mode=mode,
                        max_num_seqs=max_num_seqs,
                    )
                )
            )

    return {
        "schema_version": 1,
        "capacity_tokens": capacity_tokens,
        "max_num_seqs": max_num_seqs,
        "claim_count": len(claims),
        "fixed_order": fixed,
        "order_sensitivity": {
            key: _distribution(values) for key, values in trial_counts.items()
        },
        "interpretation": {
            "full_reservation": "reserve every tree's worst rollout peak",
            "two_phase_safe": (
                "admit candidate allocations only when a worst-case safe "
                "completion order still exists"
            ),
            "candidate_only": (
                "unsafe upper bound that ignores future rollout expansion"
            ),
            "mns_bound": (
                "also require all admitted candidate branches to fit "
                "max_num_seqs; this is an execution headroom model"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--total-length", type=int, default=512)
    parser.add_argument("--rollout-count", type=int, default=3)
    parser.add_argument("--kv-block-size", type=int, default=128)
    parser.add_argument("--capacity-tokens", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--shuffle-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.labels is not None and len(args.labels) != len(args.traces):
        parser.error("--labels must contain one value per trace")
    labels = args.labels or [path.parent.parent.name for path in args.traces]
    payload = {
        label: simulate_trace(
            extract_step_claims(
                path,
                total_length=args.total_length,
                rollout_count=args.rollout_count,
                kv_block_size=args.kv_block_size,
                include_warmup=args.include_warmup,
            ),
            capacity_tokens=args.capacity_tokens,
            max_num_seqs=args.max_num_seqs,
            shuffle_trials=args.shuffle_trials,
            seed=args.seed,
        )
        for label, path in zip(labels, args.traces, strict=True)
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
