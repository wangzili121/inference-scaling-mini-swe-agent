from __future__ import annotations

import asyncio
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest

from inference_scaling.arllm.acceleration import ActiveBatchSpeculationConfig
from inference_scaling.arllm.algorithms.mh import run_mh_chain
from inference_scaling.arllm.backends import AsyncVLLMBackend, VLLMBackend
from inference_scaling.arllm.backends.vllm_backend import _load_vllm_sampling_api
from inference_scaling.arllm.config import MHConfig, SamplingConfig
from inference_scaling.arllm.types import (
    CISRequestMetadata,
    GenerationRequest,
    ScoreRequest,
)
from inference_scaling.shared.rng import SeedStream


@dataclass
class _Logprob:
    logprob: float


@dataclass
class _Completion:
    token_ids: list[int]
    logprobs: list[dict[int, _Logprob]]
    finish_reason: str = "length"
    stop_reason: str | None = None
    power_logprobs: list[dict[int, _Logprob]] | None = None
    index: int = 0


@dataclass
class _Output:
    outputs: list[_Completion]
    prompt_logprobs: list[dict[int, _Logprob] | None] | None = None
    num_cached_tokens: int = 0
    request_id: str | None = None


class _SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _BeamParams(_SamplingParams):
    pass


def test_vllm_sampling_api_uses_025_and_026_public_import_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm = types.ModuleType("vllm")
    sampling_params = types.ModuleType("vllm.sampling_params")
    setattr(vllm, "SamplingParams", _SamplingParams)
    setattr(vllm, "TokensPrompt", dict)
    setattr(sampling_params, "BeamSearchParams", _BeamParams)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

    assert _load_vllm_sampling_api() == (_SamplingParams, dict, _BeamParams)


def test_vllm_sampling_api_tolerates_v018_without_public_prompt_or_beam_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm = types.ModuleType("vllm")
    sampling_params = types.ModuleType("vllm.sampling_params")
    setattr(vllm, "SamplingParams", _SamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

    loaded_sampling, loaded_prompt, loaded_beam = _load_vllm_sampling_api()

    assert loaded_sampling is _SamplingParams
    assert loaded_prompt is None
    assert loaded_beam(beam_width=2).beam_width == 2


@dataclass
class _Metric:
    name: str
    value: int
    labels: dict[str, str]


class _Tokenizer:
    bos_token_id = 9
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, text, add_special_tokens=True):
        values = [ord(value) % 10 for value in text]
        return ([self.bos_token_id] if add_special_tokens else []) + values

    def decode(self, tokens, skip_special_tokens=True):
        return ",".join(str(token) for token in tokens)


class _Engine:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.profile_events = []

    @staticmethod
    def _ids(prompt):
        return list(prompt["prompt_token_ids"])

    def generate(self, prompts, *, sampling_params, use_tqdm, **kwargs):
        self.calls.append((prompts, sampling_params, use_tqdm, kwargs))
        params = (
            sampling_params
            if isinstance(sampling_params, list)
            else [sampling_params] * len(prompts)
        )
        outputs = []
        for prompt, policy in zip(prompts, params, strict=True):
            prompt_ids = self._ids(prompt)
            if hasattr(policy, "prompt_logprobs"):
                prompt_scores = [None] + [
                    {token: _Logprob(-float(index) / 10)}
                    for index, token in enumerate(prompt_ids[1:], 1)
                ]
                outputs.append(
                    _Output(
                        [_Completion([7], [{7: _Logprob(-0.7)}])],
                        prompt_logprobs=prompt_scores,
                        num_cached_tokens=min(1, len(prompt_ids)),
                    )
                )
                continue
            count = min(2, policy.max_tokens)
            tokens = [int(policy.seed % 5) + 3] * count
            if policy.stop_token_ids and policy.seed == 12:
                tokens[-1] = policy.stop_token_ids[0]
            outputs.append(
                _Output(
                    [_Completion(tokens, [{token: _Logprob(-0.25)} for token in tokens])],
                    num_cached_tokens=min(2, len(prompt_ids)),
                )
            )
        return outputs

    def shutdown(self):
        self.closed = True

    def start_profile(self, prefix=None):
        self.profile_events.append(("start", prefix))

    def stop_profile(self):
        self.profile_events.append(("stop", None))


class _TopKEngine(_Engine):
    def generate(self, prompts, *, sampling_params, use_tqdm, **kwargs):
        self.calls.append((prompts, sampling_params, use_tqdm, kwargs))
        params = sampling_params if isinstance(sampling_params, list) else [sampling_params]
        return [
            _Output(
                [
                    _Completion(
                        [3],
                        [
                            {
                                3: _Logprob(-0.1),
                                4: _Logprob(-0.2),
                                5: _Logprob(-0.6),
                            }
                        ],
                    )
                ]
            )
            for _prompt, _policy in zip(prompts, params, strict=True)
        ]


class _BeamEngine(_Engine):
    def beam_search(self, **kwargs):
        self.calls.append(kwargs)
        prompt = self._ids(kwargs["prompts"][0])
        sequence = type("Beam", (), {"tokens": prompt + [4, 2]})()
        return [type("BeamOutput", (), {"sequences": [sequence]})()]


class _MetricEngine(_Engine):
    def get_metrics(self):
        return [
            _Metric("vllm:spec_decode_num_drafts", 3, {"model_name": "fake"}),
            _Metric("vllm:spec_decode_num_draft_tokens", 7, {"model_name": "fake"}),
            _Metric(
                "vllm:spec_decode_num_accepted_tokens",
                2,
                {"model_name": "fake"},
            ),
            _Metric(
                "vllm:spec_decode_num_draft_tokens",
                100,
                {"model_name": "another-model"},
            ),
            _Metric("vllm:num_preemptions", 4, {"model_name": "fake"}),
        ]


class _FusedEngine(_Engine):
    def __init__(self):
        super().__init__()
        self.references = {}
        self.rpc_calls = []

    def generate(self, prompts, *, sampling_params, use_tqdm, **kwargs):
        outputs = super().generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            **kwargs,
        )
        call = len(self.calls)
        for index, output in enumerate(outputs):
            request_id = f"engine:{call}:{index}"
            output.request_id = request_id
            tokens = output.outputs[0].token_ids
            self.references[request_id] = tuple(-0.4 for _ in tokens)
        return outputs

    def collective_rpc(self, method, *, args):
        self.rpc_calls.append((method, args))
        request_ids = args[0]
        return [
            {
                request_id: self.references.pop(request_id)
                for request_id in request_ids
                if request_id in self.references
            }
        ]


class _Fallback:
    model_id = "fake"

    def sample_batch(self, requests):
        raise AssertionError("fallback generation must not be used")

    def score_batch(self, requests):
        return [tuple(-0.5 for _ in continuation) for request in requests for continuation in request.continuations]

    def score_statistics_batch(self, requests, **_kwargs):
        return [
            {"tokens": continuation}
            for request in requests
            for continuation in request.continuations
        ]


def _backend(*, fallback=None):
    engine = _Engine()
    backend = VLLMBackend(
        engine,
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        scoring_backend=fallback,
    )
    return backend, engine


def test_vllm_generation_captures_requested_topk_confidence() -> None:
    engine = _TopKEngine()
    backend = VLLMBackend(
        engine,
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
    )
    request = GenerationRequest(
        (1,),
        1,
        SamplingConfig(),
        7,
        "top-k",
        confidence_top_k=3,
    )

    sample = backend.sample_batch([request])[0]

    assert engine.calls[0][1][0].logprobs == 3
    assert sample.token_topk_confidences == pytest.approx((0.3,))
    assert sample.confidence_top_k == 3


def test_terminal_parent_fork_waiter_does_not_charge_prefill() -> None:
    backend, _ = _backend()
    request = GenerationRequest((1, 2, 3), 4, SamplingConfig(), 7, "rollout")
    output = _Output(
        outputs=[
            _Completion(
                token_ids=[],
                logprobs=[],
                finish_reason="abort",
                stop_reason="cis_parent_terminal",
            )
        ]
    )

    _, prefill_tokens, cached_tokens, forward_slots = backend._sample_from_output(
        request, output
    )

    assert (prefill_tokens, cached_tokens, forward_slots) == (0, 0, 0)


def test_vllm_sampling_preserves_per_request_seed_policy_and_order() -> None:
    backend, engine = _backend()
    sampling = SamplingConfig(temperature=0.7, top_p=0.9, top_k=4, eos_token_id=2)
    requests = [
        GenerationRequest((1, 2), 2, sampling, seed, f"r{seed}")
        for seed in (11, 12)
    ]

    samples = backend.sample_batch(requests)

    assert [sample.request_id for sample in samples] == ["r11", "r12"]
    assert samples[0].token_logprobs == (-0.25, -0.25)
    assert samples[1].token_ids[-1] == 2
    assert samples[1].finish_reason == "eos"
    params = engine.calls[0][1]
    assert [item.seed for item in params] == [11, 12]
    assert all(item.temperature == 0.7 for item in params)
    assert all(item.top_p == 0.9 and item.top_k == 4 for item in params)
    assert all(item.stop_token_ids == [2] and item.ignore_eos for item in params)
    snapshot = backend.snapshot()
    assert snapshot.sampled_sequences == 2
    assert snapshot.generated_tokens == 4
    assert snapshot.shared_prefill_tokens_saved == 4
    assert snapshot.prefill_tokens == 0


def test_vllm_accepts_upstream_power_logprobs_without_rescoring() -> None:
    engine = _Engine()
    backend = VLLMBackend(
        engine,
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
    )
    original_generate = engine.generate

    def generate(*args, **kwargs):
        outputs = original_generate(*args, **kwargs)
        for output in outputs:
            completion = output.outputs[0]
            completion.power_logprobs = [
                {token: _Logprob(-0.4)} for token in completion.token_ids
            ]
        return outputs

    engine.generate = generate
    request = GenerationRequest(
        (1,), 2, SamplingConfig(temperature=0.5), 4, "power"
    )

    sample = backend.sample_batch([request])[0]

    assert sample.reference_token_logprobs == (-0.4, -0.4)
    assert sample.reference_policy_id == SamplingConfig().policy_id
    assert backend.snapshot().fused_reference_tokens == 2


def test_vllm_fused_reference_eliminates_mh_score_forward() -> None:
    engine = _FusedEngine()
    backend = VLLMBackend(
        engine,
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        mh_fused_logprobs=True,
    )

    result = run_mh_chain(
        backend,
        (1,),
        MHConfig(alpha=2.0, total_length=2, block_size=2, steps_per_block=1),
        SamplingConfig(temperature=0.5),
        SeedStream(7),
    )

    assert len(result.token_ids) == 2
    assert result.base_token_logprobs == (-0.4, -0.4)
    snapshot = backend.snapshot()
    assert snapshot.score_calls == 0
    assert snapshot.mh_fused_logprobs
    assert snapshot.fused_reference_sequences == 2
    assert snapshot.fused_reference_tokens >= 3
    assert len(engine.rpc_calls) == 2


def test_vllm_rejects_explicit_token_uniforms() -> None:
    backend, _ = _backend()
    request = GenerationRequest(
        (1,),
        2,
        SamplingConfig(),
        11,
        "explicit-uniforms",
        uniforms=(0.1, 0.2),
    )

    with pytest.raises(NotImplementedError, match="request-local token uniforms"):
        backend.sample_batch([request])


def test_vllm_native_score_extracts_continuation_prompt_logprobs() -> None:
    backend, _ = _backend()

    scores = backend.score_batch(
        [ScoreRequest((8, 6), ((4, 5), (), (3,)), SamplingConfig())]
    )

    assert scores == [(-0.2, -0.3), (), (-0.2,)]
    snapshot = backend.snapshot()
    assert snapshot.native_score_sequences == 2
    assert snapshot.scored_tokens == 3
    assert snapshot.score_forward_token_slots == 5
    assert snapshot.shared_prefill_tokens_saved == 2


def test_vllm_nonunit_score_requires_or_uses_exact_fallback() -> None:
    backend, _ = _backend()
    request = ScoreRequest((1,), ((2, 3),), SamplingConfig(temperature=0.7))
    with pytest.raises(ValueError, match="exact scoring_backend"):
        backend.score_batch([request])

    backend, _ = _backend(fallback=_Fallback())
    assert backend.score_batch([request]) == [(-0.5, -0.5)]
    snapshot = backend.snapshot()
    assert snapshot.delegated_score_sequences == 1
    assert snapshot.delegated_score_forward_token_slots == 3
    assert snapshot.delegated_estimated_dense_forward_flops == 600


def test_vllm_delegates_full_vocabulary_confidence_statistics() -> None:
    backend, _ = _backend(fallback=_Fallback())
    request = ScoreRequest((1,), ((2, 3),), SamplingConfig())

    assert backend.score_statistics_batch([request]) == [{"tokens": (2, 3)}]
    snapshot = backend.snapshot()
    assert snapshot.delegated_score_sequences == 1
    assert snapshot.score_forward_token_slots == 3
    assert snapshot.estimated_dense_forward_flops == 600


def test_vllm_encode_decode_and_close() -> None:
    backend, engine = _backend()
    assert backend.encode("ab", add_special_tokens=False) == (7, 8)
    assert backend.decode((1, 2, 3)) == "1,2,3"
    backend.close()
    backend.close()
    assert engine.closed


def test_vllm_direct_greedy_and_sync_beam_generation() -> None:
    engine = _BeamEngine()
    backend = VLLMBackend(
        engine,
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        beam_search_params_factory=_BeamParams,
    )

    assert backend.direct_generate((1,), max_new_tokens=2) == (3, 3)
    assert backend.direct_generate((1,), max_new_tokens=5, num_beams=4) == (4, 2)
    beam_call = engine.calls[-1]
    assert beam_call["params"].beam_width == 4
    assert beam_call["use_tqdm"] is False
    backend.start_profile("sync")
    backend.stop_profile()
    assert engine.profile_events == [("start", "sync"), ("stop", None)]


def test_vllm_snapshot_accounts_rejected_native_draft_slots() -> None:
    backend = VLLMBackend(
        _MetricEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        native_suffix_speculation=True,
    )

    snapshot = backend.snapshot()

    assert snapshot.native_speculative_drafts == 3
    assert snapshot.native_draft_tokens == 7
    assert snapshot.native_accepted_draft_tokens == 2
    assert snapshot.rejected_verification_token_slots == 5
    assert snapshot.num_preemptions == 4
    assert snapshot.generation_forward_token_slots == 5
    assert snapshot.estimated_dense_forward_flops == 1000


class _AsyncEngine(_Engine):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.maximum_active = 0

    async def _stream(self, *, prompt, sampling_params, request_id, **kwargs):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0.02)
        token = int(sampling_params.seed % 5) + 3
        self.active -= 1
        yield _Output(
            [_Completion([token], [{token: _Logprob(-0.25)}])],
            request_id=request_id,
        )

    def generate(self, **kwargs):
        return self._stream(**kwargs)

    async def shutdown(self):
        self.closed = True

    async def start_profile(self, prefix=None):
        self.profile_events.append(("start", prefix))

    async def stop_profile(self):
        self.profile_events.append(("stop", None))


class _ParallelAsyncEngine(_AsyncEngine):
    def __init__(self):
        super().__init__()
        self.parent_calls = []

    async def _stream(self, *, prompt, sampling_params, request_id, **kwargs):
        self.parent_calls.append((request_id, sampling_params))
        seeds = sampling_params.extra_args["cis_child_seeds"]
        self.active += len(seeds)
        self.maximum_active = max(self.maximum_active, self.active)
        for index, seed in enumerate(seeds):
            await asyncio.sleep(0.001)
            token = int(seed % 5) + 3
            self.active -= 1
            yield _Output(
                [
                    _Completion(
                        [token],
                        [{token: _Logprob(-0.25)}],
                        index=index,
                    )
                ],
                num_cached_tokens=1,
                request_id=request_id,
            )


class _UtilityCore:
    def __init__(self):
        self.calls = []

    async def call_utility_async(self, method, *args):
        self.calls.append((method, args))
        if method == "cis_kv_cache_geometry":
            return {
                "num_gpu_blocks": 123,
                "block_size": 128,
                "token_capacity": 15744,
            }
        return {"evicted_blocks": 7}


def test_async_vllm_overlaps_requests_from_independent_callers() -> None:
    engine = _AsyncEngine()
    backend = AsyncVLLMBackend(
        engine,
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
    )
    sampling = SamplingConfig()
    start = threading.Barrier(2)

    def generate(seed):
        start.wait()
        return backend.sample_batch(
            [GenerationRequest((1,), 1, sampling, seed, f"r{seed}")]
        )[0]

    with ThreadPoolExecutor(max_workers=2) as executor:
        samples = list(executor.map(generate, (1, 2)))

    assert [sample.request_id for sample in samples] == ["r1", "r2"]
    assert engine.maximum_active == 2
    assert backend.snapshot().maximum_in_flight_requests == 2
    assert backend.direct_generate((1,), max_new_tokens=2, num_beams=2) == (3, 3)
    backend.start_profile("async")
    backend.stop_profile()
    assert engine.profile_events == [("start", "async"), ("stop", None)]
    backend.close()
    assert engine.closed


def test_async_vllm_exposes_initialized_kv_token_capacity() -> None:
    engine = _AsyncEngine()
    engine.vllm_config = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(
            num_gpu_blocks=123,
            block_size=128,
        )
    )
    backend = AsyncVLLMBackend(
        engine,
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
    )

    assert backend.kv_token_capacity == 123 * 128
    assert backend.kv_cache_geometry == {
        "num_gpu_blocks": 123,
        "block_size": 128,
        "token_capacity": 123 * 128,
    }
    backend.close()


def test_async_vllm_reads_kv_capacity_from_engine_core() -> None:
    engine = _AsyncEngine()
    engine.engine_core = _UtilityCore()
    backend = AsyncVLLMBackend(
        engine,
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
    )

    assert backend.kv_token_capacity == 15744
    assert backend.kv_token_capacity == 15744
    assert engine.engine_core.calls == [("cis_kv_cache_geometry", ())]
    backend.close()


def test_async_vllm_streams_completion_callbacks_and_draft_observations() -> None:
    engine = _AsyncEngine()
    config = ActiveBatchSpeculationConfig(min_context_tokens=1)
    backend = AsyncVLLMBackend(
        engine,
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        speculation=config,
        native_suffix_speculation=True,
    )
    completed = []
    requests = [
        GenerationRequest((1,), 1, SamplingConfig(), seed, f"r{seed}")
        for seed in (1, 2, 3)
    ]
    try:
        outputs = backend.sample_batch_with_callback(
            requests,
            lambda index, sample: completed.append((index, sample.request_id)),
        )
        assert sorted(completed) == [(0, "r1"), (1, "r2"), (2, "r3")]
        assert [sample.request_id for sample in outputs] == ["r1", "r2", "r3"]
        snapshot = backend.snapshot()
        assert snapshot.native_suffix_speculation
        assert snapshot.observed_draft_sequences == 3
        assert backend.draft_cache_snapshot() is None
    finally:
        backend.close()


def test_async_vllm_native_parallel_sampling_preserves_child_seeds() -> None:
    engine = _ParallelAsyncEngine()
    backend = AsyncVLLMBackend(
        engine,
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        native_parallel_sampling=True,
    )
    completed = []
    requests = [
        GenerationRequest(
            (1, 2),
            1,
            SamplingConfig(),
            seed,
            f"job-a:step:0:candidate:{index}",
        )
        for index, seed in enumerate((103, 211, 307))
    ]
    try:
        outputs = backend.sample_batch_with_callback(
            requests,
            lambda index, sample: completed.append((index, sample.request_id)),
        )
        snapshot = backend.snapshot()
    finally:
        backend.close()

    assert [sample.token_ids for sample in outputs] == [(6,), (4,), (5,)]
    assert sorted(completed) == [
        (0, "job-a:step:0:candidate:0"),
        (1, "job-a:step:0:candidate:1"),
        (2, "job-a:step:0:candidate:2"),
    ]
    assert len(engine.parent_calls) == 1
    parent_id, params = engine.parent_calls[0]
    assert parent_id.endswith(":parent:3")
    assert params.n == 3
    assert params.extra_args["cis_child_seeds"] == [103, 211, 307]
    assert snapshot.native_parallel_groups == 1
    assert snapshot.native_parallel_children == 3
    assert snapshot.engine_requests == 3
    assert snapshot.maximum_in_flight_requests == 3


def test_async_vllm_segmented_rng_adds_boundary_metadata() -> None:
    request = GenerationRequest(
        (1, 2),
        4,
        SamplingConfig(),
        7,
        "job-a:step:0:candidate:0:rollout:0",
        rng_switch_after_tokens=2,
        rng_switch_seed=11,
        rng_prefix_group="job-a:step:0:candidate:0",
        rng_prefix_group_size=3,
    )
    disabled = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
    )
    try:
        with pytest.raises(RuntimeError, match="native_segmented_rng"):
            disabled._sampling_params(request)
    finally:
        disabled.close()

    enabled = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        native_segmented_rng=True,
    )
    try:
        params = enabled._sampling_params(request)
    finally:
        enabled.close()

    assert params.extra_args == {
        "cis_rng_switch_after_tokens": 2,
        "cis_rng_switch_seed": 11,
        "cis_rng_prefix_group": "job-a:step:0:candidate:0",
        "cis_rng_prefix_group_size": 3,
    }


def test_async_vllm_packed_forest_attention_adds_sibling_metadata() -> None:
    request = GenerationRequest(
        (1, 2),
        4,
        SamplingConfig(),
        7,
        "job-a:step:0:candidate:0:rollout:1",
        forest_group_id="job-a:step:0:candidate:0",
        forest_branch_index=1,
        forest_group_size=3,
    )
    disabled = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
    )
    try:
        assert getattr(disabled._sampling_params(request), "extra_args", None) is None
    finally:
        disabled.close()

    enabled = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        native_packed_forest_attention=True,
    )
    try:
        params = enabled._sampling_params(request)
    finally:
        enabled.close()

    assert params.extra_args == {
        "cis_forest_group_id": "job-a:step:0:candidate:0",
        "cis_forest_branch_index": 1,
        "cis_forest_group_size": 3,
    }


def test_async_vllm_native_kv_fork_adds_parent_child_metadata() -> None:
    backend = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        native_kv_fork=True,
    )
    try:
        parent = GenerationRequest(
            (1,),
            1,
            SamplingConfig(),
            7,
            "job-a:step:0:candidate:0",
            fork_expected_children=3,
            fork_group_id="job-a:step:0",
            fork_group_size=8,
            fork_release_remaining=2,
            fork_adaptive_release=True,
            fork_adaptive_runnable_fraction=0.5,
        )
        child = GenerationRequest(
            (1, 4),
            1,
            SamplingConfig(),
            8,
            "job-a:step:0:candidate:0:rollout:0",
            fork_parent_request_id=parent.request_id,
        )
        parent_params = backend._sampling_params(parent)
        child_params = backend._sampling_params(child)
    finally:
        backend.close()

    assert parent_params.extra_args == {
        "cis_fork_handle": "job-a:step:0:candidate:0",
        "cis_fork_expected_children": 3,
        "cis_fork_group_id": "job-a:step:0",
        "cis_fork_group_size": 8,
        "cis_fork_release_remaining": 2,
        "cis_fork_adaptive_release": True,
        "cis_fork_adaptive_runnable_fraction": 0.5,
    }
    assert child_params.extra_args == {
        "cis_fork_parent_request_id": "job-a:step:0:candidate:0"
    }


def test_async_vllm_native_kv_fork_waiter_adds_park_metadata() -> None:
    backend = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        native_kv_fork=True,
        native_kv_fork_waiters=True,
        native_kv_fork_lease=True,
        native_kv_fork_lease_scope="full_parent",
    )
    try:
        request = GenerationRequest(
            (1,),
            4,
            SamplingConfig(),
            8,
            "job-a:step:0:candidate:0:rollout:0",
            fork_parent_request_id="job-a:step:0:candidate:0",
            fork_wait_for_parent=True,
        )
        params = backend._sampling_params(request)
    finally:
        backend.close()

    assert params.extra_args == {
        "cis_fork_parent_request_id": "job-a:step:0:candidate:0",
        "cis_fork_wait_for_parent": True,
        "cis_fork_frontend_prompt_tokens": 1,
    }


def test_async_vllm_compact_fork_waiter_adds_engine_hint() -> None:
    backend = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        native_kv_fork=True,
        native_kv_fork_waiters=True,
        native_kv_fork_compact_waiters=True,
        native_kv_fork_lease=True,
        native_kv_fork_lease_scope="full_parent",
    )
    try:
        request = GenerationRequest(
            (1, 2, 3),
            4,
            SamplingConfig(),
            8,
            "job-a:step:0:candidate:0:rollout:0",
            fork_parent_request_id="job-a:step:0:candidate:0",
            fork_wait_for_parent=True,
        )
        params = backend._sampling_params(request)
    finally:
        backend.close()

    assert params.extra_args == {
        "cis_fork_parent_request_id": "job-a:step:0:candidate:0",
        "cis_fork_wait_for_parent": True,
        "cis_fork_compact_waiter": True,
        "cis_fork_frontend_prompt_tokens": 3,
    }


def test_async_vllm_compact_fork_waiter_requires_waiters() -> None:
    with pytest.raises(ValueError, match="requires native_kv_fork_waiters"):
        AsyncVLLMBackend(
            _AsyncEngine(),
            _Tokenizer(),
            model_id="fake",
            parameter_count=100,
            sampling_params_factory=_SamplingParams,
            native_kv_fork=True,
            native_kv_fork_compact_waiters=True,
            native_kv_fork_lease=True,
        )


def test_async_vllm_native_kv_fork_lease_marks_candidate_suffix() -> None:
    backend = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        native_kv_fork=True,
        native_kv_fork_lease=True,
    )
    try:
        request = GenerationRequest(
            (1, 2, 3),
            1,
            SamplingConfig(),
            7,
            "job-a:step:0:candidate:0",
            fork_expected_children=3,
        )
        params = backend._sampling_params(request)
    finally:
        backend.close()

    assert params.extra_args == {
        "cis_fork_handle": "job-a:step:0:candidate:0",
        "cis_fork_expected_children": 3,
        "cis_fork_shared_prefix_tokens": 3,
        "cis_fork_lease_scope": "candidate_suffix",
        "cis_fork_candidate_prefix_tokens": 3,
        "cis_fork_lease_ms": 300_000,
    }


def test_async_vllm_native_kv_fork_full_parent_handoff() -> None:
    backend = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        native_kv_fork=True,
        native_kv_fork_lease=True,
        native_kv_fork_lease_scope="full_parent",
        native_kv_fork_lease_max_fraction=0.2,
    )
    try:
        request = GenerationRequest(
            (1, 2, 3),
            1,
            SamplingConfig(),
            7,
            "job-a:step:0:candidate:0",
            fork_expected_children=3,
        )
        params = backend._sampling_params(request)
    finally:
        backend.close()

    assert params.extra_args == {
        "cis_fork_handle": "job-a:step:0:candidate:0",
        "cis_fork_expected_children": 3,
        "cis_fork_shared_prefix_tokens": 0,
        "cis_fork_lease_scope": "full_parent",
        "cis_fork_candidate_prefix_tokens": 3,
        "cis_fork_lease_max_fraction": 0.2,
        "cis_fork_lease_ms": 300_000,
    }


def test_async_vllm_branch_eviction_marks_rollout_tail() -> None:
    backend = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        native_kv_branch_eviction=True,
    )
    try:
        candidate = GenerationRequest(
            (1, 2),
            1,
            SamplingConfig(),
            7,
            "job-a:step:0:candidate:0",
        )
        rollout = GenerationRequest(
            (1, 2, 3, 4),
            1,
            SamplingConfig(),
            8,
            "job-a:step:0:candidate:0:rollout:0",
        )
        candidate_params = backend._sampling_params(candidate)
        rollout_params = backend._sampling_params(rollout)
    finally:
        backend.close()

    assert candidate_params.extra_args is None
    assert rollout_params.extra_args == {"cis_disposable_from_token": 4}


def test_async_vllm_resample_gc_records_candidate_and_rollout_suffixes() -> None:
    backend = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        native_kv_resample_gc=True,
    )
    try:
        candidate = GenerationRequest(
            (1, 2),
            1,
            SamplingConfig(),
            7,
            "job-a:step:0:candidate:0",
        )
        rollout = GenerationRequest(
            (1, 2, 3, 4),
            1,
            SamplingConfig(),
            8,
            "job-a:step:0:candidate:0:rollout:0",
        )
        candidate_params = backend._sampling_params(candidate)
        rollout_params = backend._sampling_params(rollout)
    finally:
        backend.close()

    assert candidate_params.extra_args == {
        "cis_branch_handle": "job-a:step:0:candidate:0",
        "cis_branch_record_from_token": 2,
        "cis_branch_kind": "candidate",
    }
    assert rollout_params.extra_args == {
        "cis_branch_handle": "job-a:step:0:candidate:0:rollout:0",
        "cis_branch_record_from_token": 4,
        "cis_branch_kind": "rollout",
    }


def test_async_vllm_resample_gc_calls_engine_core_utility() -> None:
    engine = _AsyncEngine()
    engine.engine_core = _UtilityCore()
    backend = AsyncVLLMBackend(
        engine,
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        native_kv_resample_gc=True,
    )
    try:
        result = backend.resolve_cis_branches(
            selected_candidate_id="candidate:1",
            candidate_ids=("candidate:0", "candidate:1"),
            rollout_ids=("rollout:0", "rollout:1"),
        )
    finally:
        backend.close()

    assert result == {"evicted_blocks": 7}
    assert engine.engine_core.calls == [
        (
            "cis_resample_gc",
            (
                "candidate:1",
                ["candidate:0", "candidate:1"],
                ["rollout:0", "rollout:1"],
            ),
        )
    ]


def test_async_vllm_emits_algorithm_request_lifecycle() -> None:
    engine = _AsyncEngine()
    backend = AsyncVLLMBackend(
        engine,
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
    )
    events = []
    backend.set_request_trace_observer(events.append)
    request = GenerationRequest((1, 2), 1, SamplingConfig(), 7, "job:step:0:candidate:3")
    try:
        output = backend.sample_batch([request])
    finally:
        backend.close()

    assert output[0].request_id == request.request_id
    assert [event["event"] for event in events] == [
        "submitted",
        "first_output",
        "finished",
    ]
    assert {event["engine_request_id"] for event in events} == {request.request_id}
    assert events[-1]["output_tokens"] == 1


def test_async_vllm_assigns_step_and_rollout_priorities() -> None:
    backend = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        request_priority_policy="rollout_first",
    )
    try:
        first = GenerationRequest(
            (1,), 1, SamplingConfig(), 1, "job-a:step:0:candidate:0"
        )
        child = GenerationRequest(
            (1,), 1, SamplingConfig(), 2, "job-a:step:0:candidate:0:rollout:0"
        )
        later = GenerationRequest(
            (1,), 1, SamplingConfig(), 3, "job-b:step:0:candidate:0"
        )
        assert backend._request_priority(first) == 0
        assert backend._request_priority(child) == -1_000_000
        assert backend._request_priority(later) == 1
    finally:
        backend.close()


def test_async_vllm_groups_steps_into_work_conserving_priority_cohorts() -> None:
    backend = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        request_priority_policy="step_cohort",
        request_priority_cohort_size=2,
    )
    try:
        requests = [
            GenerationRequest(
                (1,),
                1,
                SamplingConfig(),
                index,
                f"job-{index}:step:0:candidate:0",
            )
            for index in range(5)
        ]
        assert [backend._request_priority(request) for request in requests] == [
            0,
            0,
            1,
            1,
            2,
        ]
    finally:
        backend.close()


def test_async_vllm_priority_prefers_structured_cis_metadata() -> None:
    backend = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        request_priority_policy="step_fifo",
    )
    try:
        request = GenerationRequest(
            (1,),
            1,
            SamplingConfig(),
            1,
            "opaque-request-id",
            cis=CISRequestMetadata(
                job_id="job-a",
                step_index=3,
                node_type="candidate",
                candidate_index=0,
                candidate_count=1,
                expected_rollouts=3,
            ),
        )
        assert backend._request_priority(request) == 0
        assert backend._step_priorities == {"job-a:step:3": 0}
        assert backend._sampling_params(request).extra_args == {
            "cis_request": {
                "schema_version": 1,
                "job_id": "job-a",
                "step_index": 3,
                "node_type": "candidate",
                "candidate_index": 0,
                "candidate_count": 1,
                "rollout_index": None,
                "expected_rollouts": 3,
                "step_rollout_count": None,
                "candidate_max_tokens": None,
                "rollout_max_tokens": None,
            }
        }
    finally:
        backend.close()


def test_async_vllm_job_fifo_keeps_later_steps_with_their_job() -> None:
    backend = AsyncVLLMBackend(
        _AsyncEngine(),
        _Tokenizer(),
        model_id="fake",
        parameter_count=100,
        sampling_params_factory=_SamplingParams,
        request_priority_policy="job_fifo",
    )
    try:
        def request(job: str, step: int) -> GenerationRequest:
            return GenerationRequest(
                (1,),
                1,
                SamplingConfig(),
                step,
                f"{job}:step:{step}:candidate:0",
                cis=CISRequestMetadata(
                    job_id=job,
                    step_index=step,
                    node_type="candidate",
                    candidate_index=0,
                    candidate_count=1,
                ),
            )

        assert backend._request_priority(request("job-a", 0)) == 0
        assert backend._request_priority(request("job-b", 0)) == 1
        assert backend._request_priority(request("job-a", 1)) == 0
        assert backend._step_priorities == {"job-a": 0, "job-b": 1}
    finally:
        backend.close()
