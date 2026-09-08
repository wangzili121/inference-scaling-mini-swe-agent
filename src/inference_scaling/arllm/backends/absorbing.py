"""Fixed-length view of a variable-length language model.

Metropolis--Hastings is simplest on a fixed product space.  This adapter makes
EOS absorbing: once EOS is drawn, every remaining position is deterministically
EOS with log-probability zero.  The decoded response is unchanged, while every
MH proposal has exactly the requested length and a well-defined probability.
"""

from __future__ import annotations

from collections.abc import Sequence

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)


class AbsorbingEOSBackend:
    def __init__(
        self,
        backend: AutoregressiveBackend,
        eos_token_id: int,
        *,
        absorbing_after: int = 0,
    ) -> None:
        if eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative")
        if absorbing_after < 0:
            raise ValueError("absorbing_after must be non-negative")
        self.backend = backend
        self.eos_token_id = int(eos_token_id)
        self.absorbing_after = int(absorbing_after)
        self._model_id = (
            f"{backend.model_id}|absorbing-eos={self.eos_token_id};"
            f"after={self.absorbing_after}"
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    def _inner_sampling(self, sampling: SamplingConfig | None) -> SamplingConfig:
        source = sampling or SamplingConfig()
        if source.eos_token_id not in (None, self.eos_token_id):
            raise ValueError("sampling uses a different EOS token")
        return SamplingConfig(
            temperature=source.temperature,
            top_p=source.top_p,
            top_k=source.top_k,
            eos_token_id=self.eos_token_id,
        )

    def _prefix_is_terminal(self, prefix: TokenSequence) -> bool:
        if len(prefix) < self.absorbing_after:
            raise ValueError("prefix is shorter than the protected prompt")
        generated_prefix = prefix[self.absorbing_after :]
        if self.eos_token_id not in generated_prefix:
            return False
        first = generated_prefix.index(self.eos_token_id)
        if any(token != self.eos_token_id for token in generated_prefix[first:]):
            raise ValueError(
                "an absorbing-EOS prefix contains a non-EOS token after EOS"
            )
        return True

    def sample_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[SequenceSample]:
        outputs: list[SequenceSample | None] = [None] * len(requests)
        pending: list[GenerationRequest] = []
        pending_indices: list[int] = []
        for index, request in enumerate(requests):
            if request.sampling.eos_token_id is not None:
                raise ValueError(
                    "the outer fixed-length policy must leave eos_token_id unset; "
                    "the adapter supplies absorbing EOS semantics"
                )
            if self._prefix_is_terminal(request.prefix):
                outputs[index] = SequenceSample(
                    prefix=request.prefix,
                    token_ids=(self.eos_token_id,) * request.max_new_tokens,
                    token_logprobs=(0.0,) * request.max_new_tokens,
                    policy_id=request.sampling.policy_id,
                    model_id=self.model_id,
                    request_id=request.request_id,
                    finish_reason="eos",
                    reference_token_logprobs=(0.0,) * request.max_new_tokens,
                    reference_policy_id=SamplingConfig().policy_id,
                )
                continue
            pending.append(
                GenerationRequest(
                    prefix=request.prefix,
                    max_new_tokens=request.max_new_tokens,
                    sampling=self._inner_sampling(request.sampling),
                    seed=request.seed,
                    request_id=request.request_id,
                    uniforms=request.uniforms,
                    arithmetic_uniform=request.arithmetic_uniform,
                )
            )
            pending_indices.append(index)

        sampled = self.backend.sample_batch(pending) if pending else []
        if len(sampled) != len(pending):
            raise RuntimeError("wrapped backend returned an invalid number of samples")
        for index, outer, inner in zip(
            pending_indices,
            (requests[i] for i in pending_indices),
            sampled,
            strict=True,
        ):
            missing = outer.max_new_tokens - len(inner.token_ids)
            if missing < 0:
                raise RuntimeError("wrapped backend exceeded max_new_tokens")
            tokens = inner.token_ids + (self.eos_token_id,) * missing
            logprobs = inner.token_logprobs + (0.0,) * missing
            expected_inner_reference = SamplingConfig(
                eos_token_id=self.eos_token_id
            ).policy_id
            if inner.reference_policy_id != expected_inner_reference:
                reference_logprobs = None
                reference_policy_id = None
            else:
                assert inner.reference_token_logprobs is not None
                reference_logprobs = inner.reference_token_logprobs + (0.0,) * missing
                reference_policy_id = SamplingConfig().policy_id
            outputs[index] = SequenceSample(
                prefix=outer.prefix,
                token_ids=tokens,
                token_logprobs=logprobs,
                policy_id=outer.sampling.policy_id,
                model_id=self.model_id,
                request_id=outer.request_id,
                finish_reason=inner.finish_reason,
                reference_token_logprobs=reference_logprobs,
                reference_policy_id=reference_policy_id,
            )
        return [output for output in outputs if output is not None]

    def _score_one(
        self,
        request: ScoreRequest,
        continuation: TokenSequence,
    ) -> tuple[float, ...]:
        if not continuation:
            return ()
        if self._prefix_is_terminal(request.prefix):
            return tuple(
                0.0 if token == self.eos_token_id else float("-inf")
                for token in continuation
            )

        try:
            eos_position = continuation.index(self.eos_token_id)
        except ValueError:
            stochastic = continuation
            tail: TokenSequence = ()
        else:
            stochastic = continuation[: eos_position + 1]
            tail = continuation[eos_position + 1 :]

        inner = self.backend.score_batch(
            [
                ScoreRequest(
                    request.prefix,
                    (stochastic,),
                    self._inner_sampling(request.sampling),
                )
            ]
        )
        if len(inner) != 1 or len(inner[0]) != len(stochastic):
            raise RuntimeError("wrapped backend returned an invalid score shape")
        forced = tuple(
            0.0 if token == self.eos_token_id else float("-inf") for token in tail
        )
        return inner[0] + forced

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        return [
            self._score_one(request, continuation)
            for request in requests
            for continuation in request.continuations
        ]
