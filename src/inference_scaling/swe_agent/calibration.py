"""Offline reward calibration for shared Conditional IS rollout pools."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from inference_scaling.shared.metrics import importance_effective_sample_size


RewardBlocks = Sequence[Sequence[Sequence[float]]]


def _logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("each candidate requires at least one rollout reward")
    maximum = max(values)
    return maximum + math.log(
        sum(math.exp(value - maximum) for value in values) / len(values)
    )


def block_ess_ratios(
    reward_blocks: RewardBlocks,
    *,
    scale: float = 1.0,
    temperature: float = 1.0,
) -> tuple[float, ...]:
    if scale < 0 or not math.isfinite(scale):
        raise ValueError("reward scale must be finite and non-negative")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("reward temperature must be finite and positive")
    ratios: list[float] = []
    for block in reward_blocks:
        if not block:
            raise ValueError("a reward block must contain candidates")
        candidate_weights = [
            _logmeanexp([scale * float(value) / temperature for value in rewards])
            for rewards in block
        ]
        ratios.append(
            importance_effective_sample_size(candidate_weights) / len(candidate_weights)
        )
    return tuple(ratios)


def _median_ess(
    reward_blocks: RewardBlocks,
    *,
    scale: float,
    temperature: float,
) -> float:
    ratios = block_ess_ratios(
        reward_blocks,
        scale=scale,
        temperature=temperature,
    )
    if not ratios:
        raise ValueError("calibration requires at least one reward block")
    return float(statistics.median(ratios))


def _bisect_decreasing(
    objective: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    target: float,
    iterations: int = 48,
) -> float:
    lower_value = objective(lower)
    upper_value = objective(upper)
    if target >= lower_value:
        return lower
    if target <= upper_value:
        return upper
    for _ in range(iterations):
        middle = (lower + upper) / 2
        if objective(middle) > target:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    parameter: float
    median_ess_ratio: float
    block_ess_ratios: tuple[float, ...]
    target_ess_ratio: float
    bounded: bool


def calibrate_logprob_alpha(
    logprob_blocks: RewardBlocks,
    *,
    target_ess_ratio: float = 0.6,
    alpha_bounds: tuple[float, float] = (1.0, 4.0),
) -> CalibrationResult:
    lower, upper = alpha_bounds
    if not 0 < target_ess_ratio <= 1:
        raise ValueError("target_ess_ratio must be in (0, 1]")
    if lower < 1 or upper < lower:
        raise ValueError("alpha bounds must be ordered and at least one")
    alpha = _bisect_decreasing(
        lambda value: _median_ess(
            logprob_blocks,
            scale=value - 1.0,
            temperature=1.0,
        ),
        lower,
        upper,
        target=target_ess_ratio,
    )
    ratios = block_ess_ratios(logprob_blocks, scale=alpha - 1.0)
    return CalibrationResult(
        parameter=alpha,
        median_ess_ratio=float(statistics.median(ratios)),
        block_ess_ratios=ratios,
        target_ess_ratio=target_ess_ratio,
        bounded=math.isclose(alpha, lower) or math.isclose(alpha, upper),
    )


def calibrate_reward_temperature(
    reward_blocks: RewardBlocks,
    *,
    target_ess_ratio: float = 0.6,
    temperature_bounds: tuple[float, float] = (1e-3, 100.0),
) -> CalibrationResult:
    lower, upper = temperature_bounds
    if not 0 < target_ess_ratio <= 1:
        raise ValueError("target_ess_ratio must be in (0, 1]")
    if lower <= 0 or upper < lower:
        raise ValueError("temperature bounds must be positive and ordered")

    # ESS rises with temperature, so bisect inverse temperature, which is decreasing.
    inverse_lower = 1.0 / upper
    inverse_upper = 1.0 / lower
    inverse = _bisect_decreasing(
        lambda value: _median_ess(
            reward_blocks,
            scale=value,
            temperature=1.0,
        ),
        inverse_lower,
        inverse_upper,
        target=target_ess_ratio,
    )
    temperature = 1.0 / inverse
    ratios = block_ess_ratios(reward_blocks, temperature=temperature)
    return CalibrationResult(
        parameter=temperature,
        median_ess_ratio=float(statistics.median(ratios)),
        block_ess_ratios=ratios,
        target_ess_ratio=target_ess_ratio,
        bounded=math.isclose(temperature, lower) or math.isclose(temperature, upper),
    )


__all__ = [
    "CalibrationResult",
    "block_ess_ratios",
    "calibrate_logprob_alpha",
    "calibrate_reward_temperature",
]
