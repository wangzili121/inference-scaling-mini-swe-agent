"""Core request and backend contracts.

Algorithms depend on these contracts rather than on Transformers or a particular
inference server.  A backend must return probabilities under the *actual* sampling
policy, not merely unprocessed model logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, Sequence, runtime_checkable

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prefix: TokenSequence
    max_new_tokens: int
    sampling: SamplingConfig
    seed: int
    request_id: str
    uniforms: tuple[float, ...] | None = None
    arithmetic_uniform: float | None = None
    confidence_top_k: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.uniforms is not None:
            if len(self.uniforms) != self.max_new_tokens:
                raise ValueError("explicit sampling uniforms must match max_new_tokens")
            if any(
                not isfinite(value) or not 0.0 <= value < 1.0 for value in self.uniforms
            ):
                raise ValueError("sampling uniforms must be finite values in [0, 1)")
        if self.arithmetic_uniform is not None:
            if self.uniforms is not None:
                raise ValueError(
                    "token uniforms and an arithmetic uniform are mutually exclusive"
                )
            if (
                not isfinite(self.arithmetic_uniform)
                or not 0.0 <= self.arithmetic_uniform < 1.0
            ):
                raise ValueError(
                    "the arithmetic sampling uniform must be finite and in [0, 1)"
                )
        if self.confidence_top_k is not None and self.confidence_top_k <= 0:
            raise ValueError("confidence_top_k must be positive")


@dataclass(frozen=True, slots=True)
class GeneratedSequenceStatistics:
    """Small generation-time statistics sufficient for model-derived rewards."""

    token_ids: TokenSequence = ()
    token_logprobs: tuple[float, ...] = ()
    model_id: str | None = None
    policy_id: str | None = None
    token_topk_confidences: tuple[float, ...] | None = None
    confidence_top_k: int | None = None

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_logprobs):
            raise ValueError("generation statistics require one log-probability per token")
        if any(not isfinite(value) for value in self.token_logprobs):
            raise ValueError("generation log-probabilities must be finite")
        if (self.token_topk_confidences is None) != (self.confidence_top_k is None):
            raise ValueError("top-K confidences and confidence_top_k must be provided together")
        if self.token_topk_confidences is not None:
            if len(self.token_topk_confidences) != len(self.token_ids):
                raise ValueError("generation statistics require one top-K confidence per token")
            if any(not isfinite(value) for value in self.token_topk_confidences):
                raise ValueError("top-K confidences must be finite")
            if self.confidence_top_k is None or self.confidence_top_k <= 0:
                raise ValueError("confidence_top_k must be positive")

    @property
    def logprob(self) -> float:
        return float(sum(self.token_logprobs))

    def extend(
        self,
        *,
        token_ids: TokenSequence,
        token_logprobs: tuple[float, ...],
        model_id: str,
        policy_id: str,
        token_topk_confidences: tuple[float, ...] | None = None,
        confidence_top_k: int | None = None,
    ) -> "GeneratedSequenceStatistics":
        if len(token_ids) != len(token_logprobs):
            raise ValueError("extension requires one log-probability per token")
        homogeneous_model = model_id if not self.token_ids else (
            model_id if self.model_id == model_id else None
        )
        homogeneous_policy = policy_id if not self.token_ids else (
            policy_id if self.policy_id == policy_id else None
        )
        if not self.token_ids:
            combined_confidences = token_topk_confidences
            combined_top_k = confidence_top_k
        elif (
            self.token_topk_confidences is not None
            and token_topk_confidences is not None
            and self.confidence_top_k == confidence_top_k
        ):
            combined_confidences = self.token_topk_confidences + token_topk_confidences
            combined_top_k = self.confidence_top_k
        else:
            combined_confidences = None
            combined_top_k = None
        return GeneratedSequenceStatistics(
            token_ids=self.token_ids + tuple(token_ids),
            token_logprobs=self.token_logprobs + tuple(token_logprobs),
            model_id=homogeneous_model,
            policy_id=homogeneous_policy,
            token_topk_confidences=combined_confidences,
            confidence_top_k=combined_top_k,
        )


@dataclass(frozen=True, slots=True)
class SequenceSample:
    prefix: TokenSequence
    token_ids: TokenSequence
    token_logprobs: tuple[float, ...]
    policy_id: str
    model_id: str
    request_id: str
    finish_reason: str = "length"
    reference_token_logprobs: tuple[float, ...] | None = None
    reference_policy_id: str | None = None
    token_topk_confidences: tuple[float, ...] | None = None
    confidence_top_k: int | None = None

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_logprobs):
            raise ValueError(
                "each sampled token must have one actual-policy log-probability"
            )
        if any(not isfinite(value) for value in self.token_logprobs):
            raise ValueError("actual-policy token log-probabilities must be finite")
        if (self.reference_token_logprobs is None) != (
            self.reference_policy_id is None
        ):
            raise ValueError(
                "reference token probabilities and their policy id must be provided together"
            )
        if self.reference_token_logprobs is not None and len(self.token_ids) != len(
            self.reference_token_logprobs
        ):
            raise ValueError(
                "each sampled token must have one reference-policy log-probability"
            )
        if (self.token_topk_confidences is None) != (self.confidence_top_k is None):
            raise ValueError(
                "top-K confidences and confidence_top_k must be provided together"
            )
        if self.token_topk_confidences is not None:
            if len(self.token_ids) != len(self.token_topk_confidences):
                raise ValueError("each sampled token must have one top-K confidence")
            if any(not isfinite(value) for value in self.token_topk_confidences):
                raise ValueError("sampled top-K confidences must be finite")
            if self.confidence_top_k is None or self.confidence_top_k <= 0:
                raise ValueError("confidence_top_k must be positive")

    @property
    def logprob(self) -> float:
        return float(sum(self.token_logprobs))

    @property
    def full_sequence(self) -> TokenSequence:
        return self.prefix + self.token_ids

    @property
    def statistics(self) -> GeneratedSequenceStatistics:
        return GeneratedSequenceStatistics(
            token_ids=self.token_ids,
            token_logprobs=self.token_logprobs,
            model_id=self.model_id,
            policy_id=self.policy_id,
            token_topk_confidences=self.token_topk_confidences,
            confidence_top_k=self.confidence_top_k,
        )


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    prefix: TokenSequence
    continuations: tuple[TokenSequence, ...]
    sampling: SamplingConfig | None = None


@runtime_checkable
class AutoregressiveBackend(Protocol):
    """Minimal interface required by MH, conditional IS, and replay correction."""

    @property
    def model_id(self) -> str: ...

    def sample_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[SequenceSample]: ...

    def score_batch(
        self, requests: Sequence[ScoreRequest]
    ) -> list[tuple[float, ...]]: ...
