"""Conditional importance sampling.

Candidate blocks are always sampled from the base model in this module.  A
completion may be sampled on-policy or from a full-support off-policy proposal.
Only the completion suffix receives the ``p_base / q`` correction.  This is the
finite-candidate, finite-rollout sampling-importance-resampling algorithm used as
the foundation for the replay extensions.  Optional symmetric clipping of the
sequence log-ratio is recorded explicitly; it is a biased variance-control
setting, while the default ``None`` retains the exact importance ratio.  An
explicit uncorrected ablation skips target-model rescoring and instead estimates
each candidate's future reward weighting under the rollout proposal itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from math import exp, isfinite, log
from threading import Condition
from time import perf_counter
from typing import Any

from inference_scaling.arllm.acceleration import sample_batch_with_callback
from inference_scaling.arllm.config import ConditionalISConfig, SamplingConfig
from inference_scaling.shared.importance import (
    MonteCarloRolloutWeightProvider,
    RolloutObservation,
    logmeanexp,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.stepwise import (
    StepwiseCandidate,
    categorical_index_from_uniform,
    normalize_log_weights,
    run_stepwise_generation,
    stepwise_generation_step,
)
from inference_scaling.shared.verifier import TokenBatchReward, TokenReward
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GeneratedSequenceStatistics,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)

RewardFunction = TokenReward
RewardBatchFunction = TokenBatchReward
StageObserver = Callable[[str, int, float, Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class RolloutFrontierSnapshot:
    capacity: int
    admitted: int
    peak_admitted: int
    acquisitions: int
    wait_seconds: float
    waiters: int


class RolloutAdmissionController:
    """Bound rollout admission across all CIS jobs sharing one engine."""

    def __init__(self, *, capacity: int, batch_size: int) -> None:
        if capacity <= 0:
            raise ValueError("rollout frontier capacity must be positive")
        if batch_size <= 0:
            raise ValueError("rollout frontier batch size must be positive")
        if batch_size > capacity:
            raise ValueError("rollout frontier batch size cannot exceed capacity")
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self._condition = Condition()
        self._admitted = 0
        self._peak_admitted = 0
        self._acquisitions = 0
        self._wait_seconds = 0.0
        self._waiters = 0

    def acquire(self, count: int) -> float:
        if count <= 0 or count > self.capacity:
            raise ValueError("rollout frontier acquisition exceeds capacity")
        started = perf_counter()
        with self._condition:
            self._waiters += 1
            try:
                self._condition.wait_for(
                    lambda: self._admitted + count <= self.capacity
                )
            finally:
                self._waiters -= 1
            waited = perf_counter() - started
            self._admitted += count
            self._peak_admitted = max(self._peak_admitted, self._admitted)
            self._acquisitions += 1
            self._wait_seconds += waited
            return waited

    def release(self, count: int) -> None:
        if count <= 0:
            raise ValueError("rollout frontier release must be positive")
        with self._condition:
            if count > self._admitted:
                raise RuntimeError("rollout frontier released unadmitted work")
            self._admitted -= count
            self._condition.notify_all()

    def snapshot(self) -> RolloutFrontierSnapshot:
        with self._condition:
            return RolloutFrontierSnapshot(
                capacity=self.capacity,
                admitted=self._admitted,
                peak_admitted=self._peak_admitted,
                acquisitions=self._acquisitions,
                wait_seconds=self._wait_seconds,
                waiters=self._waiters,
            )


class StepAdmissionController:
    """Bound complete candidate-to-reduce steps across concurrent jobs."""

    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("active step limit must be positive")
        self.limit = int(limit)
        self._condition = Condition()
        self._active = 0

    def acquire(self) -> float:
        started = perf_counter()
        with self._condition:
            self._condition.wait_for(lambda: self._active < self.limit)
            self._active += 1
        return perf_counter() - started

    def release(self) -> None:
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("step admission released without acquisition")
            self._active -= 1
            self._condition.notify()


@lru_cache(maxsize=1)
def _record_function_factory() -> Any | None:
    try:
        from torch.autograd.profiler import record_function
    except (ImportError, ModuleNotFoundError):
        return None
    return record_function


@contextmanager
def _profile_range(name: str):
    factory = _record_function_factory()
    if factory is None:
        yield
        return
    with factory(f"conditional_is.{name}"):
        yield


def _observe_stage(
    observer: StageObserver | None,
    name: str,
    step_index: int,
    started: float,
    **metadata: Any,
) -> None:
    if observer is not None:
        observer(name, step_index, perf_counter() - started, metadata)


@dataclass(frozen=True, slots=True)
class RolloutEvaluation:
    token_ids: TokenSequence
    reward: float
    base_logprob: float | None
    proposal_logprob: float
    raw_log_importance_ratio: float | None
    applied_log_importance_ratio: float | None
    log_weight: float
    proposal_model_id: str
    proposal_policy_id: str
    generation_statistics: GeneratedSequenceStatistics | None = None


@dataclass(frozen=True, slots=True)
class ConditionalCandidate:
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    rollouts: tuple[RolloutEvaluation, ...]
    log_weight: float
    planned_rollout_count: int = 0
    log_weight_lower_bound: float | None = None
    log_weight_upper_bound: float | None = None
    base_token_topk_confidences: tuple[float, ...] | None = None
    base_confidence_top_k: int | None = None


@dataclass(frozen=True, slots=True)
class _RolloutRequestPlan:
    requests: tuple[GenerationRequest, ...]
    candidate_indices: tuple[int, ...]
    prefixes: tuple[TokenSequence, ...]
    terminal_candidates: frozenset[int]


@dataclass(frozen=True, slots=True)
class _StreamedRollouts:
    samples_by_request_id: Mapping[str, SequenceSample]
    submission_batches: int
    candidate_batch_size: int
    max_active_batches: int


@dataclass(frozen=True, slots=True)
class ConditionalISStep:
    generated_length_before: int
    candidates: tuple[ConditionalCandidate, ...]
    selected_index: int
    rollout_evaluations_planned: int = 0
    rollout_evaluations_performed: int = 0
    rollout_evaluations_skipped: int = 0
    rollout_evaluation_batches: int = 0
    exact_early_stop: bool = False
    selection_invariant_verified: bool = False

    @property
    def selected(self) -> ConditionalCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class ConditionalISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[ConditionalISStep, ...]


def _validate_base_sampling(sampling: SamplingConfig) -> None:
    if sampling.top_p < 1 or sampling.top_k is not None:
        raise ValueError(
            "base candidates must use a full-support autoregressive policy: "
            "top_p=1 and top_k=None"
        )


def _validate_rollout_sampling(sampling: SamplingConfig) -> None:
    if sampling.top_p < 1 or sampling.top_k is not None:
        raise ValueError(
            "off-policy IS requires proposal support wherever the base weighted target is positive; "
            "hard top-k/top-p truncation is not accepted"
        )


def _score_samples(
    base_backend: AutoregressiveBackend,
    prefixes: Sequence[TokenSequence],
    samples: Sequence[SequenceSample],
    base_sampling: SamplingConfig,
) -> list[float]:
    requests = [
        ScoreRequest(prefix, (sample.token_ids,), base_sampling)
        for prefix, sample in zip(prefixes, samples, strict=True)
    ]
    token_scores = base_backend.score_batch(requests)
    if len(token_scores) != len(samples):
        raise RuntimeError("backend returned an invalid number of base scores")
    totals: list[float] = []
    for sample, scores in zip(samples, token_scores, strict=True):
        if len(scores) != len(sample.token_ids):
            raise RuntimeError("backend returned an invalid base token score shape")
        total = float(sum(scores))
        if not isfinite(total):
            raise ValueError(
                "rollout proposal generated a completion outside base-model support"
            )
        totals.append(total)
    return totals


def _sample_candidates(
    base_backend: AutoregressiveBackend,
    prefix: TokenSequence,
    count: int,
    block_length: int,
    sampling: SamplingConfig,
    seeds: SeedStream,
    step_index: int,
    confidence_top_k: int | None = None,
    request_namespace: str = "conditional-is",
    stage_observer: StageObserver | None = None,
) -> list[SequenceSample]:
    requests = _candidate_requests(
        prefix=prefix,
        count=count,
        block_length=block_length,
        sampling=sampling,
        seeds=seeds,
        step_index=step_index,
        confidence_top_k=confidence_top_k,
        request_namespace=request_namespace,
    )
    started = perf_counter()
    with _profile_range("candidate"):
        candidates = base_backend.sample_batch(requests)
    _observe_stage(
        stage_observer,
        "candidate",
        step_index,
        started,
        sequence_count=count,
        block_length=block_length,
        prefix_tokens=len(prefix),
    )
    _validate_candidates(candidates, count, base_backend, sampling)
    return candidates


def _candidate_requests(
    *,
    prefix: TokenSequence,
    count: int,
    block_length: int,
    sampling: SamplingConfig,
    seeds: SeedStream,
    step_index: int,
    confidence_top_k: int | None,
    request_namespace: str,
) -> list[GenerationRequest]:
    return [
        GenerationRequest(
            prefix=prefix,
            max_new_tokens=block_length,
            sampling=sampling,
            seed=seeds.derive(
                "conditional_is", step_index, "candidate", candidate_index
            ),
            request_id=(
                f"{request_namespace}:step:{step_index}:candidate:{candidate_index}"
            ),
            confidence_top_k=confidence_top_k,
        )
        for candidate_index in range(count)
    ]


def _validate_candidates(
    candidates: Sequence[SequenceSample],
    count: int,
    base_backend: AutoregressiveBackend,
    sampling: SamplingConfig,
) -> None:
    if len(candidates) != count:
        raise RuntimeError("backend returned an invalid number of candidates")
    for candidate in candidates:
        if not candidate.token_ids:
            raise RuntimeError("a candidate block must contain at least one token")
        if (
            candidate.model_id != base_backend.model_id
            or candidate.policy_id != sampling.policy_id
        ):
            raise RuntimeError(
                "candidate was not sampled and scored by the requested base policy"
            )


def _rollout_requests_for_candidate(
    *,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    candidate_index: int,
    candidate: SequenceSample,
    rollout_length: int,
    rollout_count: int,
    rollout_sampling: SamplingConfig,
    seeds: SeedStream,
    step_index: int,
    rollout_design: str,
    rollout_index_offset: int,
    confidence_top_k: int | None,
    request_namespace: str,
) -> tuple[list[GenerationRequest], list[TokenSequence], bool]:
    full_generated_candidate = generated_prefix + candidate.token_ids
    eos = rollout_sampling.eos_token_id
    terminal = rollout_length == 0 or (
        eos is not None and candidate.token_ids[-1] == eos
    )
    if terminal:
        return [], [], True
    rollout_prefix = prompt + full_generated_candidate
    if rollout_design == "scrambled_sobol":
        from inference_scaling.experimental.shared.rqmc import (
            scrambled_sobol_uniforms,
        )

        token_uniforms = scrambled_sobol_uniforms(
            rollout_count,
            rollout_length,
            seed=seeds.derive(
                "conditional_is",
                step_index,
                "candidate",
                candidate_index,
                "scrambled_sobol",
            ),
        )
    else:
        token_uniforms = (None,) * rollout_count
    if rollout_design == "arithmetic_lattice":
        from inference_scaling.experimental.shared.rqmc import (
            randomized_lattice_uniforms,
        )

        arithmetic_uniforms = randomized_lattice_uniforms(
            rollout_count,
            seed=seeds.derive(
                "conditional_is",
                step_index,
                "candidate",
                candidate_index,
                "arithmetic_lattice",
            ),
        )
    else:
        arithmetic_uniforms = (None,) * rollout_count
    requests = []
    for rollout_index in range(rollout_count):
        global_rollout_index = rollout_index_offset + rollout_index
        requests.append(
            GenerationRequest(
                prefix=rollout_prefix,
                max_new_tokens=rollout_length,
                sampling=rollout_sampling,
                seed=seeds.derive(
                    "conditional_is",
                    step_index,
                    "candidate",
                    candidate_index,
                    "rollout",
                    global_rollout_index,
                ),
                request_id=(
                    f"{request_namespace}:"
                    f"step:{step_index}:candidate:{candidate_index}:"
                    f"rollout:{global_rollout_index}"
                ),
                uniforms=token_uniforms[rollout_index],
                arithmetic_uniform=arithmetic_uniforms[rollout_index],
                confidence_top_k=confidence_top_k,
            )
        )
    return requests, [rollout_prefix] * rollout_count, False


def _prepare_rollout_requests(
    *,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    candidates: Sequence[SequenceSample],
    rollout_length: int,
    rollout_count: int,
    rollout_sampling: SamplingConfig,
    seeds: SeedStream,
    step_index: int,
    rollout_design: str,
    rollout_index_offset: int,
    confidence_top_k: int | None,
    request_namespace: str,
) -> _RolloutRequestPlan:
    requests: list[GenerationRequest] = []
    candidate_indices: list[int] = []
    prefixes: list[TokenSequence] = []
    terminal_candidates: set[int] = set()
    for candidate_index, candidate in enumerate(candidates):
        candidate_requests, candidate_prefixes, terminal = (
            _rollout_requests_for_candidate(
                prompt=prompt,
                generated_prefix=generated_prefix,
                candidate_index=candidate_index,
                candidate=candidate,
                rollout_length=rollout_length,
                rollout_count=rollout_count,
                rollout_sampling=rollout_sampling,
                seeds=seeds,
                step_index=step_index,
                rollout_design=rollout_design,
                rollout_index_offset=rollout_index_offset,
                confidence_top_k=confidence_top_k,
                request_namespace=request_namespace,
            )
        )
        if terminal:
            terminal_candidates.add(candidate_index)
        requests.extend(candidate_requests)
        candidate_indices.extend([candidate_index] * len(candidate_requests))
        prefixes.extend(candidate_prefixes)
    return _RolloutRequestPlan(
        tuple(requests),
        tuple(candidate_indices),
        tuple(prefixes),
        frozenset(terminal_candidates),
    )


def _sample_candidates_with_streamed_rollouts(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    candidate_count: int,
    candidate_length: int,
    remaining_length: int,
    rollout_count: int,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    seeds: SeedStream,
    step_index: int,
    candidate_batch_size: int,
    max_active_batches: int,
    confidence_top_k: int | None,
    request_namespace: str,
    stage_observer: StageObserver | None,
) -> tuple[list[SequenceSample], _StreamedRollouts]:
    """Start bounded rollout groups as candidate requests complete."""

    requests = _candidate_requests(
        prefix=prompt + generated_prefix,
        count=candidate_count,
        block_length=candidate_length,
        sampling=base_sampling,
        seeds=seeds,
        step_index=step_index,
        confidence_top_k=confidence_top_k,
        request_namespace=request_namespace,
    )
    completed_candidates: list[SequenceSample | None] = [None] * candidate_count
    pending: list[tuple[int, SequenceSample]] = []
    futures: list[
        Future[tuple[tuple[GenerationRequest, ...], list[SequenceSample]]]
    ] = []
    rollout_started: float | None = None
    rollout_request_count = 0
    executor = ThreadPoolExecutor(
        max_workers=max_active_batches,
        thread_name_prefix="conditional-is-rollout",
    )

    def sample_group(
        group_requests: tuple[GenerationRequest, ...],
    ) -> tuple[tuple[GenerationRequest, ...], list[SequenceSample]]:
        with _profile_range("rollout"):
            return group_requests, rollout_backend.sample_batch(group_requests)

    def flush_ready(*, force: bool) -> None:
        nonlocal rollout_started, rollout_request_count
        first = completed_candidates[0]
        if first is None:
            return
        fixed_rollout_length = max(0, remaining_length - len(first.token_ids))
        while pending and (force or len(pending) >= candidate_batch_size):
            count = min(candidate_batch_size, len(pending))
            group = pending[:count]
            del pending[:count]
            group_requests: list[GenerationRequest] = []
            for candidate_index, candidate in group:
                candidate_requests, _, _ = _rollout_requests_for_candidate(
                    prompt=prompt,
                    generated_prefix=generated_prefix,
                    candidate_index=candidate_index,
                    candidate=candidate,
                    rollout_length=fixed_rollout_length,
                    rollout_count=rollout_count,
                    rollout_sampling=rollout_sampling,
                    seeds=seeds,
                    step_index=step_index,
                    rollout_design="iid",
                    rollout_index_offset=0,
                    confidence_top_k=confidence_top_k,
                    request_namespace=request_namespace,
                )
                group_requests.extend(candidate_requests)
            if not group_requests:
                continue
            if rollout_started is None:
                rollout_started = perf_counter()
            materialized = tuple(group_requests)
            rollout_request_count += len(materialized)
            futures.append(executor.submit(sample_group, materialized))

    def candidate_completed(index: int, sample: SequenceSample) -> None:
        completed_candidates[index] = sample
        pending.append((index, sample))
        flush_ready(force=False)

    candidate_started = perf_counter()
    try:
        with _profile_range("candidate"):
            candidates = sample_batch_with_callback(
                base_backend, requests, candidate_completed
            )
        candidate_finished = perf_counter()
        _observe_stage(
            stage_observer,
            "candidate",
            step_index,
            candidate_started,
            sequence_count=candidate_count,
            block_length=candidate_length,
            prefix_tokens=len(prompt) + len(generated_prefix),
            rollout_streaming=True,
        )
        _validate_candidates(candidates, candidate_count, base_backend, base_sampling)
        if any(candidate is None for candidate in completed_candidates):
            raise RuntimeError("candidate completion callback omitted a request")
        flush_ready(force=True)
        samples_by_request_id: dict[str, SequenceSample] = {}
        for future in futures:
            group_requests, samples = future.result()
            if len(samples) != len(group_requests):
                raise RuntimeError("streamed rollout batch returned an invalid size")
            for request, sample in zip(group_requests, samples, strict=True):
                if sample.request_id != request.request_id:
                    raise RuntimeError("streamed rollout returned the wrong request")
                samples_by_request_id[request.request_id] = sample
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    rollout_finished = perf_counter()
    effective_rollout_started = rollout_started or rollout_finished
    first_candidate = completed_candidates[0]
    if first_candidate is None:
        raise RuntimeError("candidate completion callback omitted the first request")
    _observe_stage(
        stage_observer,
        "rollout",
        step_index,
        effective_rollout_started,
        sequence_count=rollout_request_count,
        rollout_length=max(0, remaining_length - len(first_candidate.token_ids)),
        prefix_tokens=(
            len(prompt) + len(generated_prefix) + candidate_length
            if rollout_request_count
            else 0
        ),
        submission_batches=len(futures),
        submission_batch_size=None,
        candidate_batch_size=candidate_batch_size,
        max_active_batches=max_active_batches,
        candidate_overlap=True,
        overlap_seconds=max(
            0.0, candidate_finished - effective_rollout_started
        ),
    )
    return candidates, _StreamedRollouts(
        samples_by_request_id,
        len(futures),
        candidate_batch_size,
        max_active_batches,
    )


def estimate_conditional_weights(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    candidates: Sequence[SequenceSample],
    rollout_length: int,
    rollout_count: int,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    reward_temperature: float,
    importance_log_ratio_clip: float | None,
    apply_importance_correction: bool,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None = None,
    rollout_design: str = "iid",
    rollout_index_offset: int = 0,
    generated_prefix_statistics: GeneratedSequenceStatistics | None = None,
    rollout_submission_batch_size: int | None = None,
    rollout_admission_controller: RolloutAdmissionController | None = None,
    precomputed_rollouts: _StreamedRollouts | None = None,
    request_namespace: str = "conditional-is",
    stage_observer: StageObserver | None = None,
) -> tuple[ConditionalCandidate, ...]:
    """Estimate each candidate's conditional weight with on/off-policy rollouts."""

    _validate_rollout_sampling(rollout_sampling)
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")
    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")
    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one of reward or reward_batch")
    if rollout_design not in {
        "iid",
        "scrambled_sobol",
        "arithmetic_lattice",
    }:
        raise ValueError("unknown rollout_design")
    if rollout_index_offset < 0:
        raise ValueError("rollout_index_offset must be non-negative")
    if (
        rollout_submission_batch_size is not None
        and rollout_submission_batch_size <= 0
    ):
        raise ValueError("rollout_submission_batch_size must be positive")
    if rollout_index_offset and rollout_design != "iid":
        raise ValueError("staged rollout offsets currently require iid rollouts")
    if rollout_design != "iid" and reward_batch is not None:
        raise ValueError(
            "randomized QMC rollouts require a fixed pointwise reward; "
            "batch-coupled rewards change when rollout dependence changes"
        )
    confidence_top_k = getattr(reward, "generation_confidence_top_k", None)
    plan = _prepare_rollout_requests(
        prompt=prompt,
        generated_prefix=generated_prefix,
        candidates=candidates,
        rollout_length=rollout_length,
        rollout_count=rollout_count,
        rollout_sampling=rollout_sampling,
        seeds=seeds,
        step_index=step_index,
        rollout_design=rollout_design,
        rollout_index_offset=rollout_index_offset,
        confidence_top_k=confidence_top_k,
        request_namespace=request_namespace,
    )
    requests = plan.requests
    request_candidates = plan.candidate_indices
    rollout_prefixes = plan.prefixes
    terminal_candidates = plan.terminal_candidates

    if precomputed_rollouts is None:
        rollout_started = perf_counter()
        submission_batch_size = (
            rollout_admission_controller.batch_size
            if rollout_admission_controller is not None
            else rollout_submission_batch_size or len(requests) or 1
        )
        samples: list[SequenceSample] = []
        submission_batches = 0
        admission_wait_seconds = 0.0
        with _profile_range("rollout"):
            for start in range(0, len(requests), submission_batch_size):
                request_batch = requests[start : start + submission_batch_size]
                if rollout_admission_controller is None:
                    samples.extend(rollout_backend.sample_batch(request_batch))
                else:
                    admission_wait_seconds += rollout_admission_controller.acquire(
                        len(request_batch)
                    )
                    completed = 0

                    def release_completed(
                        _index: int, _sample: SequenceSample
                    ) -> None:
                        nonlocal completed
                        completed += 1
                        rollout_admission_controller.release(1)

                    try:
                        samples.extend(
                            sample_batch_with_callback(
                                rollout_backend,
                                request_batch,
                                release_completed,
                            )
                        )
                    finally:
                        remaining = len(request_batch) - completed
                        if remaining:
                            rollout_admission_controller.release(remaining)
                submission_batches += 1
        frontier = (
            rollout_admission_controller.snapshot()
            if rollout_admission_controller is not None
            else None
        )
        _observe_stage(
            stage_observer,
            "rollout",
            step_index,
            rollout_started,
            sequence_count=len(requests),
            rollout_length=rollout_length,
            prefix_tokens=(len(rollout_prefixes[0]) if rollout_prefixes else 0),
            submission_batches=submission_batches,
            submission_batch_size=(
                submission_batch_size
                if rollout_admission_controller is not None
                else rollout_submission_batch_size
            ),
            candidate_overlap=False,
            admission_wait_seconds=admission_wait_seconds,
            frontier_capacity=(None if frontier is None else frontier.capacity),
            frontier_peak_admitted=(
                None if frontier is None else frontier.peak_admitted
            ),
            frontier_waiters=(None if frontier is None else frontier.waiters),
        )
    else:
        expected = {request.request_id for request in requests}
        observed = set(precomputed_rollouts.samples_by_request_id)
        if observed != expected:
            raise RuntimeError("streamed rollout request set does not match the plan")
        samples = [
            precomputed_rollouts.samples_by_request_id[request.request_id]
            for request in requests
        ]
    if len(samples) != len(requests):
        raise RuntimeError("backend returned an invalid number of rollouts")
    if rollout_backend is not base_backend:
        observe = getattr(base_backend, "observe_draft_samples", None)
        if callable(observe):
            observe(samples)
    rollout_is_base_policy = (
        rollout_backend.model_id == base_backend.model_id
        and rollout_sampling == base_sampling
    )
    if rollout_is_base_policy:
        base_totals: list[float | None] = [sample.logprob for sample in samples]
    elif apply_importance_correction:
        scoring_started = perf_counter()
        with _profile_range("scoring"):
            base_totals = (
                _score_samples(
                    base_backend,
                    rollout_prefixes,
                    samples,
                    base_sampling,
                )
                if samples
                else []
            )
        _observe_stage(
            stage_observer,
            "scoring",
            step_index,
            scoring_started,
            sequence_count=len(samples),
            token_count=sum(len(sample.token_ids) for sample in samples),
        )
    else:
        # This is a deliberate biased ablation, not an IS estimate of the base
        # continuation distribution.  Keep the score absent so diagnostics and
        # backend accounting cannot mistake it for an evaluated zero log-ratio.
        base_totals = [None for _ in samples]

    prefix_statistics = generated_prefix_statistics or GeneratedSequenceStatistics()
    candidate_statistics = [
        prefix_statistics.extend(
            token_ids=candidate.token_ids,
            token_logprobs=candidate.token_logprobs,
            model_id=candidate.model_id,
            policy_id=candidate.policy_id,
            token_topk_confidences=candidate.token_topk_confidences,
            confidence_top_k=candidate.confidence_top_k,
        )
        for candidate in candidates
    ]
    pending_by_candidate: list[
        list[
            tuple[
                TokenSequence,
                float | None,
                float,
                str,
                str,
                TokenSequence,
                GeneratedSequenceStatistics,
            ]
        ]
    ] = [[] for _ in candidates]
    for candidate_index in terminal_candidates:
        generated = generated_prefix + candidates[candidate_index].token_ids
        pending_by_candidate[candidate_index].append(
            (
                (),
                0.0,
                0.0,
                rollout_backend.model_id,
                rollout_sampling.policy_id,
                generated,
                candidate_statistics[candidate_index],
            )
        )
    for candidate_index, sample, base_logprob in zip(
        request_candidates, samples, base_totals, strict=True
    ):
        generated = (
            generated_prefix + candidates[candidate_index].token_ids + sample.token_ids
        )
        proposal_logprob = sample.logprob
        pending_by_candidate[candidate_index].append(
            (
                sample.token_ids,
                base_logprob,
                proposal_logprob,
                sample.model_id,
                sample.policy_id,
                generated,
                candidate_statistics[candidate_index].extend(
                    token_ids=sample.token_ids,
                    token_logprobs=sample.token_logprobs,
                    model_id=sample.model_id,
                    policy_id=sample.policy_id,
                    token_topk_confidences=sample.token_topk_confidences,
                    confidence_top_k=sample.confidence_top_k,
                ),
            )
        )

    pending = [item for group in pending_by_candidate for item in group]
    generated_statistics = [item[-1] for item in pending]
    generated_sequences = [item.token_ids for item in generated_statistics]
    reward_started = perf_counter()
    with _profile_range("reward"):
        if reward_batch is not None:
            rewards = tuple(
                float(value) for value in reward_batch(prompt, generated_sequences)
            )
            if len(rewards) != len(pending):
                raise ValueError("reward_batch returned an invalid number of rewards")
        else:
            assert reward is not None
            statistics_batch = getattr(reward, "batch_statistics", None)
            if callable(statistics_batch):
                rewards = tuple(
                    float(value)
                    for value in statistics_batch(prompt, generated_statistics)
                )
                if len(rewards) != len(pending):
                    raise ValueError(
                        "sample-aware reward returned an invalid number of rewards"
                    )
            else:
                rewards = tuple(
                    float(reward(prompt, generated))
                    for generated in generated_sequences
                )
    _observe_stage(
        stage_observer,
        "reward",
        step_index,
        reward_started,
        sequence_count=len(pending),
        fused_statistics=callable(getattr(reward, "batch_statistics", None)),
    )
    if any(not isfinite(value) for value in rewards):
        raise ValueError("reward must be finite")

    weighting_started = perf_counter()
    importance_weights = MonteCarloRolloutWeightProvider[
        tuple[TokenSequence, str, str]
    ](
        reward_temperature=reward_temperature,
        correction="importance",
        log_ratio_clip=importance_log_ratio_clip,
    )
    reward_only_weights = MonteCarloRolloutWeightProvider[
        tuple[TokenSequence, str, str]
    ](
        reward_temperature=reward_temperature,
        correction="none",
    )
    by_candidate: list[list[RolloutEvaluation]] = [[] for _ in candidates]
    reward_index = 0
    for candidate_index, group in enumerate(pending_by_candidate):
        for (
            token_ids,
            base_logprob,
            proposal_logprob,
            model_id,
            policy_id,
            _,
            statistics,
        ) in group:
            reward_value = rewards[reward_index]
            reward_index += 1
            observation = RolloutObservation(
                reward=reward_value,
                target_logprob=base_logprob,
                proposal_logprob=proposal_logprob,
                payload=(token_ids, model_id, policy_id),
            )
            weighted = (
                importance_weights.weight(observation)
                if base_logprob is not None
                else reward_only_weights.weight(observation)
            )
            by_candidate[candidate_index].append(
                RolloutEvaluation(
                    token_ids=token_ids,
                    reward=reward_value,
                    base_logprob=base_logprob,
                    proposal_logprob=proposal_logprob,
                    raw_log_importance_ratio=weighted.raw_log_importance_ratio,
                    applied_log_importance_ratio=weighted.applied_log_importance_ratio,
                    log_weight=weighted.log_weight,
                    proposal_model_id=model_id,
                    proposal_policy_id=policy_id,
                    generation_statistics=statistics,
                )
            )

    evaluated: list[ConditionalCandidate] = []
    for candidate_index, candidate in enumerate(candidates):
        evaluations = by_candidate[candidate_index]
        if not evaluations:
            raise RuntimeError(
                "each candidate must have at least one weight contribution"
            )
        candidate_log_weight = logmeanexp([item.log_weight for item in evaluations])
        evaluated.append(
            ConditionalCandidate(
                token_ids=candidate.token_ids,
                base_token_logprobs=candidate.token_logprobs,
                rollouts=tuple(evaluations),
                log_weight=candidate_log_weight,
                planned_rollout_count=len(evaluations),
                log_weight_lower_bound=candidate_log_weight,
                log_weight_upper_bound=candidate_log_weight,
                base_token_topk_confidences=candidate.token_topk_confidences,
                base_confidence_top_k=candidate.confidence_top_k,
            )
        )
    result = tuple(evaluated)
    _observe_stage(
        stage_observer,
        "weight",
        step_index,
        weighting_started,
        candidate_count=len(candidates),
        rollout_evaluations=len(pending),
    )
    return result


class AutoregressiveStepwiseAdapter:
    """Expose conditional AR generation through the common stepwise protocol."""

    def __init__(
        self,
        *,
        base_backend: AutoregressiveBackend,
        rollout_backend: AutoregressiveBackend,
        prompt: TokenSequence,
        config: ConditionalISConfig,
        base_sampling: SamplingConfig,
        rollout_sampling: SamplingConfig,
        reward: RewardFunction | None,
        reward_batch: RewardBatchFunction | None = None,
        rollout_admission_controller: RolloutAdmissionController | None = None,
        step_admission_controller: StepAdmissionController | None = None,
        request_namespace: str = "conditional-is",
        stage_observer: StageObserver | None = None,
    ) -> None:
        self.base_backend = base_backend
        self.rollout_backend = rollout_backend
        self.prompt = prompt
        self.config = config
        self.base_sampling = base_sampling
        self.rollout_sampling = rollout_sampling
        self.reward = reward
        self.reward_batch = reward_batch
        self.rollout_admission_controller = rollout_admission_controller
        self.step_admission_controller = step_admission_controller
        self.request_namespace = request_namespace
        self.stage_observer = stage_observer
        self._step_started: dict[int, float] = {}
        self._admitted_steps: set[int] = set()
        self._streamed_rollouts: dict[int, _StreamedRollouts] = {}
        self._statistics_by_state: dict[TokenSequence, GeneratedSequenceStatistics] = {
            (): GeneratedSequenceStatistics()
        }

    @property
    def initial_state(self) -> TokenSequence:
        return ()

    def is_terminal(self, state: TokenSequence) -> bool:
        eos = self.base_sampling.eos_token_id
        return len(state) >= self.config.total_length or (
            eos is not None and eos in state
        )

    def propose(
        self,
        state: TokenSequence,
        step_index: int,
        seeds: SeedStream,
    ) -> Sequence[SequenceSample]:
        admission_started = perf_counter()
        if self.step_admission_controller is not None:
            wait_seconds = self.step_admission_controller.acquire()
            self._admitted_steps.add(step_index)
            _observe_stage(
                self.stage_observer,
                "step_admission_wait",
                step_index,
                admission_started,
                wait_seconds=wait_seconds,
                active_step_limit=self.step_admission_controller.limit,
            )
        self._step_started[step_index] = perf_counter()
        try:
            _validate_base_sampling(self.base_sampling)
            remaining = self.config.total_length - len(state)
            if remaining <= 0:
                raise ValueError("generated prefix has already reached total_length")
            if self.config.stream_candidate_rollouts:
                proposals, streamed = _sample_candidates_with_streamed_rollouts(
                    base_backend=self.base_backend,
                    rollout_backend=self.rollout_backend,
                    prompt=self.prompt,
                    generated_prefix=state,
                    candidate_count=self.config.candidate_count,
                    candidate_length=min(self.config.block_size, remaining),
                    remaining_length=remaining,
                    rollout_count=self.config.rollout_count,
                    base_sampling=self.base_sampling,
                    rollout_sampling=self.rollout_sampling,
                    seeds=seeds,
                    step_index=step_index,
                    candidate_batch_size=(
                        self.config.rollout_stream_candidate_batch_size
                    ),
                    max_active_batches=self.config.rollout_stream_max_batches,
                    confidence_top_k=getattr(
                        self.reward, "generation_confidence_top_k", None
                    ),
                    request_namespace=self.request_namespace,
                    stage_observer=self.stage_observer,
                )
                self._streamed_rollouts[step_index] = streamed
                return proposals
            return _sample_candidates(
                self.base_backend,
                self.prompt + state,
                self.config.candidate_count,
                min(self.config.block_size, remaining),
                self.base_sampling,
                seeds,
                step_index,
                confidence_top_k=getattr(
                    self.reward, "generation_confidence_top_k", None
                ),
                request_namespace=self.request_namespace,
                stage_observer=self.stage_observer,
            )
        except BaseException:
            self._release_step(step_index)
            raise

    def _release_step(self, step_index: int) -> None:
        if step_index in self._admitted_steps:
            self._admitted_steps.remove(step_index)
            assert self.step_admission_controller is not None
            self.step_admission_controller.release()

    def release_pending_steps(self) -> None:
        """Release admissions left behind when a step aborts outside the adapter."""

        for step_index in tuple(self._admitted_steps):
            self._release_step(step_index)

    def evaluate(
        self,
        state: TokenSequence,
        proposals: Sequence[SequenceSample],
        step_index: int,
        seeds: SeedStream,
    ) -> Sequence[StepwiseCandidate[ConditionalCandidate]]:
        remaining = self.config.total_length - len(state)
        candidate_length = len(proposals[0].token_ids)
        try:
            evaluated = estimate_conditional_weights(
                base_backend=self.base_backend,
                rollout_backend=self.rollout_backend,
                prompt=self.prompt,
                generated_prefix=state,
                candidates=proposals,
                rollout_length=max(0, remaining - candidate_length),
                rollout_count=self.config.rollout_count,
                base_sampling=self.base_sampling,
                rollout_sampling=self.rollout_sampling,
                reward_temperature=self.config.reward_temperature,
                importance_log_ratio_clip=self.config.importance_log_ratio_clip,
                apply_importance_correction=self.config.apply_importance_correction,
                reward=self.reward,
                seeds=seeds,
                step_index=step_index,
                reward_batch=self.reward_batch,
                rollout_design=self.config.rollout_design,
                generated_prefix_statistics=self._statistics_by_state.get(state),
                rollout_submission_batch_size=(
                    self.config.rollout_submission_batch_size
                ),
                rollout_admission_controller=self.rollout_admission_controller,
                precomputed_rollouts=self._streamed_rollouts.pop(step_index, None),
                request_namespace=self.request_namespace,
                stage_observer=self.stage_observer,
            )
        except BaseException:
            self._release_step(step_index)
            raise
        return tuple(
            StepwiseCandidate(candidate, candidate.log_weight)
            for candidate in evaluated
        )

    def advance(
        self,
        state: TokenSequence,
        selected: ConditionalCandidate,
        step_index: int,
    ) -> TokenSequence:
        started = perf_counter()
        generated = state + selected.token_ids
        previous = self._statistics_by_state.get(state)
        if previous is not None:
            self._statistics_by_state[generated] = previous.extend(
                token_ids=selected.token_ids,
                token_logprobs=selected.base_token_logprobs,
                model_id=self.base_backend.model_id,
                policy_id=self.base_sampling.policy_id,
                token_topk_confidences=selected.base_token_topk_confidences,
                confidence_top_k=selected.base_confidence_top_k,
            )
        eos = self.base_sampling.eos_token_id
        if eos is not None and eos in generated:
            generated = generated[: generated.index(eos) + 1]
        _observe_stage(
            self.stage_observer,
            "resample",
            step_index,
            started,
            selected_tokens=len(selected.token_ids),
        )
        block_started = self._step_started.pop(step_index, None)
        if block_started is not None:
            _observe_stage(
                self.stage_observer,
                "block",
                step_index,
                block_started,
                generated_tokens_before=len(state),
                generated_tokens_after=len(generated),
            )
        self._release_step(step_index)
        return generated


def _bounded_conditional_is_step(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    config: ConditionalISConfig,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None,
    request_namespace: str = "conditional-is",
    stage_observer: StageObserver | None = None,
) -> ConditionalISStep:
    """Evaluate rollout batches until the fixed categorical choice is known."""

    if reward is None or reward_batch is not None:
        raise ValueError(
            "exact rollout early stopping requires a fixed pointwise reward"
        )
    if config.rollout_log_weight_bounds is None:
        raise ValueError("exact rollout early stopping requires log-weight bounds")
    if config.rollout_design != "iid":
        raise ValueError("exact rollout early stopping currently requires iid rollouts")
    _validate_base_sampling(base_sampling)
    remaining_length = config.total_length - len(generated_prefix)
    if remaining_length <= 0:
        raise ValueError("generated prefix has already reached total_length")
    candidate_length = min(config.block_size, remaining_length)
    proposals = _sample_candidates(
        base_backend,
        prompt + generated_prefix,
        config.candidate_count,
        candidate_length,
        base_sampling,
        seeds,
        step_index,
        confidence_top_k=getattr(reward, "generation_confidence_top_k", None),
        request_namespace=request_namespace,
        stage_observer=stage_observer,
    )
    rollout_length = max(0, remaining_length - len(proposals[0].token_ids))
    eos = rollout_sampling.eos_token_id
    terminal = tuple(
        rollout_length == 0 or (eos is not None and proposal.token_ids[-1] == eos)
        for proposal in proposals
    )
    planned = tuple(
        1 if is_terminal else config.rollout_count for is_terminal in terminal
    )
    planned_total = sum(planned)
    selection_uniform = float(
        seeds.generator("conditional_is", step_index, "select").random()
    )
    lower_log_weight, upper_log_weight = config.rollout_log_weight_bounds
    try:
        minimum_contribution = exp(lower_log_weight)
        maximum_contribution = exp(upper_log_weight)
    except OverflowError as error:
        raise ValueError("rollout log-weight bounds cannot be exponentiated") from error
    if (
        not isfinite(minimum_contribution)
        or not isfinite(maximum_contribution)
        or minimum_contribution <= 0.0
    ):
        raise ValueError(
            "rollout log-weight bounds must map to finite positive weights"
        )

    collected: list[list[RolloutEvaluation]] = [[] for _ in proposals]
    lower_candidate_weights: list[float] = []
    upper_candidate_weights: list[float] = []
    invariant_index: int | None = None
    rollout_offset = 0
    evaluation_batches = 0
    while rollout_offset < config.rollout_count:
        batch_size = min(
            config.rollout_evaluation_batch_size,
            config.rollout_count - rollout_offset,
        )
        batch = estimate_conditional_weights(
            base_backend=base_backend,
            rollout_backend=rollout_backend,
            prompt=prompt,
            generated_prefix=generated_prefix,
            candidates=proposals,
            rollout_length=rollout_length,
            rollout_count=batch_size,
            base_sampling=base_sampling,
            rollout_sampling=rollout_sampling,
            reward_temperature=config.reward_temperature,
            importance_log_ratio_clip=config.importance_log_ratio_clip,
            apply_importance_correction=config.apply_importance_correction,
            reward=reward,
            seeds=seeds,
            step_index=step_index,
            rollout_design="iid",
            rollout_index_offset=rollout_offset,
            request_namespace=request_namespace,
            stage_observer=stage_observer,
        )
        evaluation_batches += 1
        for candidate_index, evaluated in enumerate(batch):
            if terminal[candidate_index]:
                if not collected[candidate_index]:
                    collected[candidate_index].append(evaluated.rollouts[0])
                continue
            for rollout in evaluated.rollouts:
                if not lower_log_weight <= rollout.log_weight <= upper_log_weight:
                    raise ValueError(
                        "observed rollout log-weight lies outside the declared bounds"
                    )
                collected[candidate_index].append(rollout)
        rollout_offset += batch_size

        lower_candidate_weights = []
        upper_candidate_weights = []
        for candidate_index, evaluations in enumerate(collected):
            contributions = [exp(item.log_weight) for item in evaluations]
            if terminal[candidate_index]:
                exact_weight = contributions[0]
                lower_candidate_weights.append(exact_weight)
                upper_candidate_weights.append(exact_weight)
                continue
            unseen = config.rollout_count - len(evaluations)
            lower_candidate_weights.append(
                (sum(contributions) + unseen * minimum_contribution)
                / config.rollout_count
            )
            upper_candidate_weights.append(
                (sum(contributions) + unseen * maximum_contribution)
                / config.rollout_count
            )
        from inference_scaling.experimental.shared.bounded_selection import (
            invariant_categorical_index,
        )

        invariant_index = invariant_categorical_index(
            lower_candidate_weights,
            upper_candidate_weights,
            uniform=selection_uniform,
        )
        if invariant_index is not None:
            break

    evaluated_candidates: list[ConditionalCandidate] = []
    for candidate_index, proposal in enumerate(proposals):
        evaluations = collected[candidate_index]
        if not evaluations:
            raise RuntimeError("bounded evaluation omitted a candidate")
        lower_weight = lower_candidate_weights[candidate_index]
        upper_weight = upper_candidate_weights[candidate_index]
        representative_weight = (lower_weight + upper_weight) / 2.0
        evaluated_candidates.append(
            ConditionalCandidate(
                token_ids=proposal.token_ids,
                base_token_logprobs=proposal.token_logprobs,
                rollouts=tuple(evaluations),
                log_weight=log(representative_weight),
                planned_rollout_count=planned[candidate_index],
                log_weight_lower_bound=log(lower_weight),
                log_weight_upper_bound=log(upper_weight),
            )
        )
    probabilities = normalize_log_weights(
        [candidate.log_weight for candidate in evaluated_candidates]
    )
    selected_index = categorical_index_from_uniform(
        probabilities,
        selection_uniform,
    )
    if invariant_index is not None and selected_index != invariant_index:
        raise RuntimeError(
            "bounded categorical proof disagrees with representative weights"
        )
    performed_total = sum(len(candidate.rollouts) for candidate in evaluated_candidates)
    skipped_total = planned_total - performed_total
    return ConditionalISStep(
        generated_length_before=len(generated_prefix),
        candidates=tuple(evaluated_candidates),
        selected_index=selected_index,
        rollout_evaluations_planned=planned_total,
        rollout_evaluations_performed=performed_total,
        rollout_evaluations_skipped=skipped_total,
        rollout_evaluation_batches=evaluation_batches,
        exact_early_stop=skipped_total > 0,
        selection_invariant_verified=skipped_total > 0 and invariant_index is not None,
    )


def conditional_is_step(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    config: ConditionalISConfig,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None = None,
    rollout_admission_controller: RolloutAdmissionController | None = None,
    step_admission_controller: StepAdmissionController | None = None,
    request_namespace: str = "conditional-is",
    stage_observer: StageObserver | None = None,
) -> ConditionalISStep:
    if (
        rollout_admission_controller is None
        and config.rollout_frontier_capacity is not None
    ):
        rollout_admission_controller = RolloutAdmissionController(
            capacity=config.rollout_frontier_capacity,
            batch_size=config.rollout_frontier_batch_size,
        )
    if config.exact_rollout_early_stop:
        return _bounded_conditional_is_step(
            base_backend=base_backend,
            rollout_backend=rollout_backend,
            prompt=prompt,
            generated_prefix=generated_prefix,
            config=config,
            base_sampling=base_sampling,
            rollout_sampling=rollout_sampling,
            reward=reward,
            seeds=seeds,
            step_index=step_index,
            reward_batch=reward_batch,
            request_namespace=request_namespace,
            stage_observer=stage_observer,
        )
    adapter = AutoregressiveStepwiseAdapter(
        base_backend=base_backend,
        rollout_backend=rollout_backend,
        prompt=prompt,
        config=config,
        base_sampling=base_sampling,
        rollout_sampling=rollout_sampling,
        reward=reward,
        reward_batch=reward_batch,
        rollout_admission_controller=rollout_admission_controller,
        step_admission_controller=step_admission_controller,
        request_namespace=request_namespace,
        stage_observer=stage_observer,
    )
    try:
        selection = stepwise_generation_step(
            adapter,
            generated_prefix,
            step_index,
            seeds,
            selection_namespace=("conditional_is",),
        )
    finally:
        adapter.release_pending_steps()
    evaluated_candidates = tuple(candidate.value for candidate in selection.candidates)
    performed = sum(len(candidate.rollouts) for candidate in evaluated_candidates)
    return ConditionalISStep(
        generated_length_before=len(generated_prefix),
        candidates=evaluated_candidates,
        selected_index=selection.selected_index,
        rollout_evaluations_planned=performed,
        rollout_evaluations_performed=performed,
        rollout_evaluation_batches=1,
    )


def run_conditional_is(
    base_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: ConditionalISConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    *,
    base_sampling: SamplingConfig | None = None,
    rollout_backend: AutoregressiveBackend | None = None,
    rollout_sampling: SamplingConfig | None = None,
    reward_batch: RewardBatchFunction | None = None,
    rollout_admission_controller: RolloutAdmissionController | None = None,
    step_admission_controller: StepAdmissionController | None = None,
    request_namespace: str = "conditional-is",
    stage_observer: StageObserver | None = None,
) -> ConditionalISResult:
    """Generate a sequence by repeatedly applying finite conditional-IS steps."""

    base_sampling = base_sampling or SamplingConfig()
    rollout_backend = rollout_backend or base_backend
    rollout_sampling = rollout_sampling or base_sampling
    _validate_base_sampling(base_sampling)
    _validate_rollout_sampling(rollout_sampling)
    if base_sampling.eos_token_id != rollout_sampling.eos_token_id:
        raise ValueError("candidate and rollout policies must agree on eos_token_id")
    if (
        rollout_admission_controller is None
        and config.rollout_frontier_capacity is not None
    ):
        rollout_admission_controller = RolloutAdmissionController(
            capacity=config.rollout_frontier_capacity,
            batch_size=config.rollout_frontier_batch_size,
        )

    if config.exact_rollout_early_stop:
        generated: TokenSequence = ()
        steps: list[ConditionalISStep] = []
        step_index = 0
        eos = base_sampling.eos_token_id
        while len(generated) < config.total_length and (
            eos is None or eos not in generated
        ):
            step = conditional_is_step(
                base_backend=base_backend,
                rollout_backend=rollout_backend,
                prompt=prompt,
                generated_prefix=generated,
                config=config,
                base_sampling=base_sampling,
                rollout_sampling=rollout_sampling,
                reward=reward,
                seeds=seeds,
                step_index=step_index,
                reward_batch=reward_batch,
                rollout_admission_controller=rollout_admission_controller,
                step_admission_controller=step_admission_controller,
                request_namespace=request_namespace,
                stage_observer=stage_observer,
            )
            generated += step.selected.token_ids
            if eos is not None and eos in generated:
                generated = generated[: generated.index(eos) + 1]
            steps.append(step)
            step_index += 1
        return ConditionalISResult(
            prompt=prompt,
            token_ids=generated,
            steps=tuple(steps),
        )

    adapter = AutoregressiveStepwiseAdapter(
        base_backend=base_backend,
        rollout_backend=rollout_backend,
        prompt=prompt,
        config=config,
        base_sampling=base_sampling,
        rollout_sampling=rollout_sampling,
        reward=reward,
        reward_batch=reward_batch,
        rollout_admission_controller=rollout_admission_controller,
        step_admission_controller=step_admission_controller,
        request_namespace=request_namespace,
        stage_observer=stage_observer,
    )
    try:
        generic = run_stepwise_generation(
            adapter,
            seeds,
            selection_namespace=("conditional_is",),
        )
    finally:
        adapter.release_pending_steps()
    steps: list[ConditionalISStep] = []
    for step in generic.steps:
        candidates = tuple(candidate.value for candidate in step.candidates)
        performed = sum(len(candidate.rollouts) for candidate in candidates)
        steps.append(
            ConditionalISStep(
                generated_length_before=len(step.state_before),
                candidates=candidates,
                selected_index=step.selected_index,
                rollout_evaluations_planned=performed,
                rollout_evaluations_performed=performed,
                rollout_evaluation_batches=1,
            )
        )
    return ConditionalISResult(
        prompt=prompt,
        token_ids=generic.final_state,
        steps=tuple(steps),
    )
