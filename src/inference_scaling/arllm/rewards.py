"""Autoregressive rewards derived from model probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, ScoreRequest
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class SequenceLogProbabilityReward:
    """Return a scaled full-sequence ``log p(completion | prompt)``.

    This is a model-derived reward, not an external verifier.  The backend must
    support exact scoring under ``sampling``.  With scale ``c`` and reward
    temperature ``tau``, reward reweighting targets
    ``p(completion | prompt) ** (1 + c / tau)``.
    """

    backend: AutoregressiveBackend
    sampling: SamplingConfig | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.scale):
            raise ValueError("log-probability reward scale must be finite")

    def __call__(self, prompt: TokenSequence, completion: TokenSequence) -> float:
        return self.batch(prompt, (completion,))[0]

    def batch(
        self,
        prompt: TokenSequence,
        completions: Sequence[TokenSequence],
    ) -> tuple[float, ...]:
        if not completions:
            return ()
        scored = self.backend.score_batch(
            [ScoreRequest(prompt, tuple(map(tuple, completions)), self.sampling)]
        )
        if len(scored) != len(completions):
            raise RuntimeError("backend returned an invalid log-probability score batch")
        if any(
            len(token_scores) != len(completion)
            for token_scores, completion in zip(scored, completions, strict=True)
        ):
            raise RuntimeError("backend returned an invalid token score shape")
        return tuple(self.scale * float(sum(token_scores)) for token_scores in scored)

    def describe(self) -> dict[str, object]:
        return {
            "source": "model_sequence_log_probability",
            "model_id": self.backend.model_id,
            "policy_id": self.sampling.policy_id if self.sampling is not None else None,
            "scale": self.scale,
        }


@dataclass(frozen=True, slots=True)
class ConsilienceReward:
    """Verifier-free confidence-trajectory reward for one generated sequence.

    At token ``t``, confidence is the negative mean log-probability of the
    model's top-``K`` next-token candidates.  The sequence reward is the final
    window mean minus ``initial_penalty`` times the initial window mean.  The
    first ``skip_fraction`` of tokens are omitted from the initial window.

    The result is pointwise: it never depends on the other candidates in a
    batch.  It can therefore be used unchanged by Best-of-N, conditional IS,
    and replay-based methods.
    """

    backend: AutoregressiveBackend
    sampling: SamplingConfig | None = None
    top_k: int = 5
    window_fraction: float = 0.2
    window_tokens: int | None = None
    skip_fraction: float = 0.05
    initial_penalty: float = 3.0
    scale: float = 1.0
    reasoning_end_token_ids: TokenSequence | None = None

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("Consilience top_k must be positive")
        if not isfinite(self.window_fraction) or not 0 < self.window_fraction <= 1:
            raise ValueError("Consilience window_fraction must be in (0, 1]")
        if self.window_tokens is not None and self.window_tokens <= 0:
            raise ValueError("Consilience window_tokens must be positive")
        if not isfinite(self.skip_fraction) or not 0 <= self.skip_fraction < 1:
            raise ValueError("Consilience skip_fraction must be in [0, 1)")
        if not isfinite(self.initial_penalty) or self.initial_penalty < 0:
            raise ValueError("Consilience initial_penalty must be finite and non-negative")
        if not isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Consilience scale must be finite and positive")
        if self.reasoning_end_token_ids is not None and not self.reasoning_end_token_ids:
            raise ValueError("reasoning_end_token_ids must be nonempty when provided")

    def __call__(self, prompt: TokenSequence, completion: TokenSequence) -> float:
        return self.batch(prompt, (completion,))[0]

    def _reasoning_tokens(self, completion: TokenSequence) -> TokenSequence:
        token_ids = tuple(completion)
        marker = self.reasoning_end_token_ids
        if marker is None:
            return token_ids
        marker_length = len(marker)
        for start in range(len(token_ids) - marker_length + 1):
            if token_ids[start : start + marker_length] == marker:
                return token_ids[:start]
        return token_ids

    def _trajectory_score(self, values: Sequence[float]) -> float:
        length = len(values)
        if length == 0:
            raise ValueError("Consilience reward requires a nonempty reasoning sequence")
        skipped = min(length - 1, int(length * self.skip_fraction))
        requested_window = (
            self.window_tokens
            if self.window_tokens is not None
            else max(1, int(length * self.window_fraction))
        )
        window = min(requested_window, length - skipped)
        initial = sum(values[skipped : skipped + window]) / window
        final = sum(values[length - window :]) / window
        return self.scale * (final - self.initial_penalty * initial)

    def batch(
        self,
        prompt: TokenSequence,
        completions: Sequence[TokenSequence],
    ) -> tuple[float, ...]:
        if not completions:
            return ()
        reasoning_sequences = tuple(
            self._reasoning_tokens(completion) for completion in completions
        )
        if any(not sequence for sequence in reasoning_sequences):
            raise ValueError("Consilience reward requires a nonempty reasoning sequence")
        callback: Any = getattr(self.backend, "score_statistics_batch", None)
        if callback is None:
            raise ValueError(
                "Consilience reward requires a backend with top-K score statistics"
            )
        statistics = callback(
            [ScoreRequest(prompt, reasoning_sequences, self.sampling)],
            confidence_top_k=self.top_k,
        )
        if len(statistics) != len(reasoning_sequences):
            raise RuntimeError("backend returned an invalid Consilience score batch")
        rewards: list[float] = []
        for sequence, item in zip(reasoning_sequences, statistics, strict=True):
            values = tuple(float(value) for value in item.token_topk_confidences)
            if len(values) != len(sequence):
                raise RuntimeError("backend returned an invalid Consilience trajectory")
            rewards.append(self._trajectory_score(values))
        return tuple(rewards)

    def describe(self) -> dict[str, object]:
        return {
            "source": "model_consilience",
            "model_id": self.backend.model_id,
            "policy_id": self.sampling.policy_id if self.sampling is not None else None,
            "top_k": self.top_k,
            "window_fraction": self.window_fraction,
            "window_tokens": self.window_tokens,
            "skip_fraction": self.skip_fraction,
            "initial_penalty": self.initial_penalty,
            "scale": self.scale,
            "reasoning_end_token_ids": self.reasoning_end_token_ids,
        }


__all__ = ["ConsilienceReward", "SequenceLogProbabilityReward"]
