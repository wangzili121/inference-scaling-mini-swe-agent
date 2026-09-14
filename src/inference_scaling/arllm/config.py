"""Validated configuration objects shared by all algorithms."""

from __future__ import annotations

from dataclasses import dataclass

from inference_scaling.shared.config import (
    RuntimeConfig as RuntimeConfig,
    SMCForestConfig as SMCForestConfig,
    canonical_float,
    require_finite,
    require_positive,
    require_probability,
)


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """The complete stochastic policy used for one autoregressive request."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        require_positive("temperature", self.temperature)
        require_probability("top_p", self.top_p, include_zero=False)
        if self.top_k is not None:
            require_positive("top_k", self.top_k)
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative")

    @property
    def policy_id(self) -> str:
        return (
            f"temperature={canonical_float(self.temperature)};"
            f"top_p={canonical_float(self.top_p)};"
            f"top_k={self.top_k};eos={self.eos_token_id}"
        )


@dataclass(frozen=True, slots=True)
class MHConfig:
    alpha: float = 4.0
    total_length: int = 192
    block_size: int = 32
    steps_per_block: int = 10
    chains: int = 1
    suffix_schedule: str = "uniform"

    def __post_init__(self) -> None:
        require_finite("alpha", self.alpha)
        if self.alpha < 1:
            raise ValueError("alpha must be at least one")
        for name in ("total_length", "block_size", "steps_per_block", "chains"):
            require_positive(name, getattr(self, name))
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.suffix_schedule not in {"uniform", "inverse_length", "multiscale"}:
            raise ValueError("unknown MH suffix_schedule")


@dataclass(frozen=True, slots=True)
class RewardMHConfig:
    """Full-sequence MH budget for a base-times-exponentiated-reward target."""

    total_length: int = 192
    block_size: int = 32
    steps_per_block: int = 10
    reward_temperature: float = 0.1
    suffix_schedule: str = "uniform"

    def __post_init__(self) -> None:
        for name in ("total_length", "block_size", "steps_per_block"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.suffix_schedule not in {"uniform", "inverse_length", "multiscale"}:
            raise ValueError("unknown MH suffix_schedule")

    @property
    def updates(self) -> int:
        blocks = (self.total_length + self.block_size - 1) // self.block_size
        return blocks * self.steps_per_block


@dataclass(frozen=True, slots=True)
class ConditionalISConfig:
    candidate_count: int = 4
    rollout_count: int = 4
    block_size: int = 16
    total_length: int = 128
    reward_temperature: float = 1.0
    importance_log_ratio_clip: float | None = None
    apply_importance_correction: bool = True
    rollout_design: str = "iid"
    exact_rollout_early_stop: bool = False
    rollout_log_weight_bounds: tuple[float, float] | None = None
    rollout_evaluation_batch_size: int = 1
    rollout_submission_batch_size: int | None = None
    rollout_subtree_max_active_batches: int | None = None
    fused_candidate_rollout_paths: bool = False
    engine_fork_candidate_rollouts: bool = False
    engine_fork_release_remaining_candidates: int | None = None
    engine_fork_adaptive_release: bool = False
    engine_fork_adaptive_runnable_fraction: float = 0.5
    stream_candidate_rollouts: bool = False
    rollout_stream_candidate_batch_size: int = 5
    rollout_stream_max_batches: int = 2
    rollout_frontier_capacity: int | None = None
    rollout_frontier_batch_size: int = 15
    active_step_limit: int | None = None
    active_step_admission: str = "fixed"
    active_step_max_limit: int | None = None
    active_step_token_budget: int | None = None
    active_step_kv_capacity_fraction: float = 0.9
    active_step_reference_window: int = 32
    active_step_queue_policy: str = "fifo"
    active_step_coalesce_seconds: float = 0.0
    active_step_borrow_limit: int | None = None
    active_step_borrow_below_requests: int | None = None

    def __post_init__(self) -> None:
        for name in ("candidate_count", "rollout_count", "block_size", "total_length"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.importance_log_ratio_clip is not None:
            require_positive(
                "importance_log_ratio_clip",
                self.importance_log_ratio_clip,
            )
        if (
            not self.apply_importance_correction
            and self.importance_log_ratio_clip is not None
        ):
            raise ValueError(
                "importance_log_ratio_clip requires apply_importance_correction=True"
            )
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.rollout_design not in {
            "iid",
            "scrambled_sobol",
            "arithmetic_lattice",
        }:
            raise ValueError("unknown rollout_design")
        require_positive(
            "rollout_evaluation_batch_size",
            self.rollout_evaluation_batch_size,
        )
        if self.rollout_submission_batch_size is not None:
            require_positive(
                "rollout_submission_batch_size",
                self.rollout_submission_batch_size,
            )
        if self.rollout_subtree_max_active_batches is not None:
            require_positive(
                "rollout_subtree_max_active_batches",
                self.rollout_subtree_max_active_batches,
            )
        require_positive(
            "rollout_stream_candidate_batch_size",
            self.rollout_stream_candidate_batch_size,
        )
        require_positive(
            "rollout_stream_max_batches",
            self.rollout_stream_max_batches,
        )
        if self.rollout_frontier_capacity is not None:
            require_positive(
                "rollout_frontier_capacity",
                self.rollout_frontier_capacity,
            )
        require_positive(
            "rollout_frontier_batch_size",
            self.rollout_frontier_batch_size,
        )
        if self.active_step_limit is not None:
            require_positive("active_step_limit", self.active_step_limit)
        if self.active_step_admission not in {
            "fixed",
            "peak_token_budget",
            "runtime_kv_budget",
        }:
            raise ValueError("unknown active_step_admission")
        if (
            self.active_step_admission == "peak_token_budget"
            and self.active_step_limit is None
        ):
            raise ValueError("rolling peak-token admission requires active_step_limit")
        if self.active_step_token_budget is not None:
            require_positive("active_step_token_budget", self.active_step_token_budget)
        if not 0 < self.active_step_kv_capacity_fraction <= 1:
            raise ValueError("active_step_kv_capacity_fraction must be in (0, 1]")
        if self.active_step_max_limit is not None:
            require_positive("active_step_max_limit", self.active_step_max_limit)
            if (
                self.active_step_limit is None
                and self.active_step_admission != "runtime_kv_budget"
            ):
                raise ValueError("active_step_max_limit requires active_step_limit")
            if (
                self.active_step_limit is not None
                and self.active_step_max_limit < self.active_step_limit
            ):
                raise ValueError(
                    "active_step_max_limit must not be below active_step_limit"
                )
        if (
            self.active_step_admission == "runtime_kv_budget"
            and self.active_step_max_limit is None
        ):
            raise ValueError("runtime KV admission requires active_step_max_limit")
        require_positive(
            "active_step_reference_window", self.active_step_reference_window
        )
        if self.active_step_queue_policy not in {
            "fifo",
            "largest_fit",
            "balanced_fit",
        }:
            raise ValueError("unknown active_step_queue_policy")
        if self.active_step_coalesce_seconds < 0:
            raise ValueError("active_step_coalesce_seconds must not be negative")
        if self.active_step_borrow_limit is not None:
            require_positive("active_step_borrow_limit", self.active_step_borrow_limit)
            if self.active_step_limit is None:
                raise ValueError("active_step_borrow_limit requires active_step_limit")
            if self.active_step_borrow_limit <= self.active_step_limit:
                raise ValueError(
                    "active_step_borrow_limit must exceed active_step_limit"
                )
        if self.active_step_borrow_below_requests is not None:
            require_positive(
                "active_step_borrow_below_requests",
                self.active_step_borrow_below_requests,
            )
            if self.active_step_borrow_limit is None:
                raise ValueError(
                    "active_step_borrow_below_requests requires "
                    "active_step_borrow_limit"
                )
        if (
            self.rollout_frontier_capacity is not None
            and self.rollout_frontier_batch_size > self.rollout_frontier_capacity
        ):
            raise ValueError(
                "rollout_frontier_batch_size cannot exceed "
                "rollout_frontier_capacity"
            )
        if self.stream_candidate_rollouts and self.rollout_submission_batch_size:
            raise ValueError(
                "stream_candidate_rollouts and rollout_submission_batch_size "
                "are mutually exclusive"
            )
        if self.rollout_subtree_max_active_batches is not None and (
            self.stream_candidate_rollouts
            or self.rollout_submission_batch_size is not None
            or self.rollout_frontier_capacity is not None
        ):
            raise ValueError(
                "rollout_subtree_max_active_batches is mutually exclusive with "
                "other rollout admission controls"
            )
        if self.fused_candidate_rollout_paths and (
            self.stream_candidate_rollouts
            or self.rollout_submission_batch_size is not None
            or self.rollout_subtree_max_active_batches is not None
            or self.rollout_frontier_capacity is not None
            or self.exact_rollout_early_stop
        ):
            raise ValueError(
                "fused_candidate_rollout_paths is mutually exclusive with staged "
                "rollout admission and early stopping"
            )
        if self.fused_candidate_rollout_paths and self.rollout_design != "iid":
            raise ValueError("fused_candidate_rollout_paths currently requires iid rollouts")
        if self.engine_fork_candidate_rollouts and (
            self.fused_candidate_rollout_paths
            or self.stream_candidate_rollouts
            or self.rollout_submission_batch_size is not None
            or self.rollout_subtree_max_active_batches is not None
            or self.rollout_frontier_capacity is not None
            or self.exact_rollout_early_stop
        ):
            raise ValueError(
                "engine_fork_candidate_rollouts is mutually exclusive with other "
                "rollout execution modes"
            )
        if self.engine_fork_candidate_rollouts and self.rollout_design != "iid":
            raise ValueError("engine_fork_candidate_rollouts currently requires iid rollouts")
        if self.engine_fork_release_remaining_candidates is not None:
            if not self.engine_fork_candidate_rollouts:
                raise ValueError(
                    "engine_fork_release_remaining_candidates requires "
                    "engine_fork_candidate_rollouts"
                )
            if not 0 <= self.engine_fork_release_remaining_candidates < self.candidate_count:
                raise ValueError(
                    "engine_fork_release_remaining_candidates must be in "
                    "[0, candidate_count)"
                )
        if self.engine_fork_adaptive_release and not self.engine_fork_candidate_rollouts:
            raise ValueError(
                "engine_fork_adaptive_release requires "
                "engine_fork_candidate_rollouts"
            )
        if not 0.0 < self.engine_fork_adaptive_runnable_fraction <= 1.0:
            raise ValueError(
                "engine_fork_adaptive_runnable_fraction must be in (0, 1]"
            )
        if self.rollout_frontier_capacity is not None and (
            self.stream_candidate_rollouts
            or self.rollout_submission_batch_size is not None
        ):
            raise ValueError(
                "rollout_frontier_capacity is mutually exclusive with per-job "
                "rollout submission controls"
            )
        if self.rollout_frontier_capacity is not None and self.exact_rollout_early_stop:
            raise ValueError(
                "rollout_frontier_capacity does not support exact rollout early stop"
            )
        if self.stream_candidate_rollouts and self.exact_rollout_early_stop:
            raise ValueError(
                "stream_candidate_rollouts and exact_rollout_early_stop "
                "are mutually exclusive"
            )
        if self.stream_candidate_rollouts and self.rollout_design != "iid":
            raise ValueError(
                "stream_candidate_rollouts currently requires iid rollouts"
            )
        if self.rollout_log_weight_bounds is not None:
            if len(self.rollout_log_weight_bounds) != 2:
                raise ValueError("rollout_log_weight_bounds requires two values")
            lower, upper = self.rollout_log_weight_bounds
            require_finite("rollout_log_weight_lower_bound", lower)
            require_finite("rollout_log_weight_upper_bound", upper)
            if lower > upper:
                raise ValueError("rollout log-weight bounds must be ordered")
        if self.exact_rollout_early_stop:
            if self.rollout_log_weight_bounds is None:
                raise ValueError(
                    "exact rollout early stopping requires log-weight bounds"
                )
            if self.rollout_design != "iid":
                raise ValueError(
                    "exact rollout early stopping currently requires iid rollouts"
                )
        elif self.rollout_log_weight_bounds is not None:
            raise ValueError(
                "rollout_log_weight_bounds require exact_rollout_early_stop=True"
            )


@dataclass(frozen=True, slots=True)
class IteratedConditionalISConfig:
    """Finite-pool i-SIR updates for each autoregressive candidate block."""

    pool_size: int = 3
    updates: int = 4
    rollout_count: int = 4
    block_size: int = 16
    total_length: int = 128
    reward_temperature: float = 1.0
    importance_log_ratio_clip: float | None = None
    apply_importance_correction: bool = True

    def __post_init__(self) -> None:
        for name in (
            "pool_size",
            "updates",
            "rollout_count",
            "block_size",
            "total_length",
        ):
            require_positive(name, getattr(self, name))
        if self.pool_size < 2:
            raise ValueError("pool_size must be at least two")
        require_positive("reward_temperature", self.reward_temperature)
        if self.importance_log_ratio_clip is not None:
            require_positive(
                "importance_log_ratio_clip",
                self.importance_log_ratio_clip,
            )
        if (
            not self.apply_importance_correction
            and self.importance_log_ratio_clip is not None
        ):
            raise ValueError(
                "importance_log_ratio_clip requires apply_importance_correction=True"
            )
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")

    @property
    def fresh_candidate_evaluations(self) -> int:
        """Distinct extended states evaluated at one generation step."""

        return 1 + self.updates * (self.pool_size - 1)

    @property
    def pool_candidate_uses(self) -> int:
        return self.updates * self.pool_size


@dataclass(frozen=True, slots=True)
class ProgressiveISConfig:
    """Pilot/evaluation split for cost-aware conditional-weight estimation."""

    candidate_count: int = 4
    pilot_rollouts_per_candidate: int = 2
    evaluation_cost_budget: float = 16.0
    minimum_evaluation_per_candidate: int = 1
    block_size: int = 16
    total_length: int = 128
    reward_temperature: float = 1.0
    importance_log_ratio_clip: float | None = None
    reward_workers: int = 4
    run_ahead_rollouts_per_candidate: int = 0
    evaluation_reference_rollouts_per_candidate: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "candidate_count",
            "pilot_rollouts_per_candidate",
            "minimum_evaluation_per_candidate",
            "block_size",
            "total_length",
            "reward_workers",
        ):
            require_positive(name, getattr(self, name))
        require_positive("evaluation_cost_budget", self.evaluation_cost_budget)
        require_positive("reward_temperature", self.reward_temperature)
        if self.importance_log_ratio_clip is not None:
            require_positive(
                "importance_log_ratio_clip", self.importance_log_ratio_clip
            )
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.run_ahead_rollouts_per_candidate < 0:
            raise ValueError("run_ahead_rollouts_per_candidate must be non-negative")
        if (
            self.evaluation_reference_rollouts_per_candidate is not None
            and self.evaluation_reference_rollouts_per_candidate <= 0
        ):
            raise ValueError(
                "evaluation_reference_rollouts_per_candidate must be positive"
            )


@dataclass(frozen=True, slots=True)
class BaseReplayConfig:
    candidate_count: int = 4
    block_size: int = 16
    total_length: int = 128
    reward_temperature: float = 1.0
    max_history_per_candidate: int = 8
    fresh_rollouts: int = 2
    truncation: float = 8.0
    reserve_rollouts: int = 0

    def __post_init__(self) -> None:
        for name in ("candidate_count", "block_size", "total_length"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.max_history_per_candidate < 0:
            raise ValueError("max_history_per_candidate must be non-negative")
        require_positive("fresh_rollouts", self.fresh_rollouts)
        require_positive("truncation", self.truncation)
        if self.reserve_rollouts < 0:
            raise ValueError("reserve_rollouts must be non-negative")


@dataclass(frozen=True, slots=True)
class DynamicISConfig:
    candidate_count: int = 4
    block_size: int = 16
    total_length: int = 128
    reward_temperature: float = 1.0
    max_history_per_candidate: int = 8
    truncation: float = 8.0
    reserve_rollouts: int = 0
    rollout_budget: float = 64.0
    auxiliary_mixture: float = 0.25
    minimum_fresh_per_candidate: int = 1

    def __post_init__(self) -> None:
        for name in ("candidate_count", "block_size", "total_length"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")
        if self.max_history_per_candidate < 0:
            raise ValueError("max_history_per_candidate must be non-negative")
        require_positive("truncation", self.truncation)
        if self.reserve_rollouts < 0:
            raise ValueError("reserve_rollouts must be non-negative")
        require_positive("rollout_budget", self.rollout_budget)
        require_probability("auxiliary_mixture", self.auxiliary_mixture)
        if self.auxiliary_mixture >= 1:
            raise ValueError("auxiliary_mixture must lie in [0, 1)")
        require_positive(
            "minimum_fresh_per_candidate", self.minimum_fresh_per_candidate
        )
