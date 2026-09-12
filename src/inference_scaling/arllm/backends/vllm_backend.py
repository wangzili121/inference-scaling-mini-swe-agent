"""vLLM backends with exact sampling-policy accounting.

The synchronous backend deliberately submits one vLLM request per framework
``GenerationRequest``.  This retains request-local seeds while still allowing
vLLM to batch requests and reuse their common prefixes.  Collapsing repeated
prompts into ``SamplingParams(n=...)`` would replace those independent seeds by
one group seed and make results depend on how callers happened to be batched.

vLLM can return processed log-probabilities for generated tokens, but prompt
log-probabilities are always raw model probabilities.  Native continuation
scoring is therefore exact for the full-support temperature-one policy.  Other
policies are delegated to an optional exact scoring backend instead of silently
using incorrect importance ratios.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import itertools
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Any

from packaging.version import Version

from inference_scaling.arllm.acceleration import (
    ActiveBatchSpeculationConfig,
    RolloutTokenTree,
    RolloutTokenTreeSnapshot,
    SampleCompletionCallback,
)
from inference_scaling.shared.compute import dense_forward_flops
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)

_PROTECTED_ENGINE_KWARGS = frozenset(
    {
        "model",
        "pipeline_parallel_size",
        "dtype",
        "tensor_parallel_size",
        "data_parallel_size",
        "gpu_memory_utilization",
        "quantization",
        "enforce_eager",
        "trust_remote_code",
        "revision",
        "download_dir",
        "seed",
        "enable_prefix_caching",
        "generation_config",
        "logprobs_mode",
        "enable_lora",
        "max_lora_rank",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "worker_cls",
        "async_scheduling",
    }
)

_MH_FUSED_WORKER = (
    "inference_scaling.arllm.backends.vllm_mh_worker.MHFusedLogprobWorker"
)

_CIS_STEP_REQUEST = re.compile(
    r"^(?P<job>.*):step:(?P<step>\d+):candidate:(?P<candidate>\d+)"
    r"(?::rollout:(?P<rollout>\d+))?$"
)


def _cis_parallel_group_key(request: GenerationRequest) -> tuple[Any, ...] | None:
    """Return the native parent group for one Conditional IS child request."""

    if request.uniforms is not None or request.arithmetic_uniform is not None:
        return None
    match = _CIS_STEP_REQUEST.match(request.request_id)
    if match is None:
        return None
    if match.group("rollout") is None:
        node = (match.group("job"), match.group("step"), "candidate")
    else:
        node = (
            match.group("job"),
            match.group("step"),
            "rollout",
            match.group("candidate"),
        )
    return (
        *node,
        request.prefix,
        request.max_new_tokens,
        request.sampling,
        request.confidence_top_k,
    )


def _validate_mh_fused_vllm_version() -> None:
    """Fail before allocating a model when the worker adapter is incompatible."""

    installed = Version(importlib.metadata.version("vllm"))
    if not Version("0.26") <= installed < Version("0.27"):
        raise RuntimeError(
            f"MH fused log-probabilities require vLLM >=0.26,<0.27; found {installed}"
        )


def _load_vllm_sampling_api() -> tuple[Any, Any, Any]:
    """Load public sampling types across the supported vLLM 0.18--0.26 range."""

    from vllm import SamplingParams

    try:
        from vllm import TokensPrompt
    except ImportError:
        TokensPrompt = None
    try:
        from vllm.sampling_params import BeamSearchParams
    except ImportError:
        try:
            from vllm import BeamSearchParams
        except ImportError:
            BeamSearchParams = _BeamSearchParamsShim

    return SamplingParams, TokensPrompt, BeamSearchParams


@dataclass
class _BeamSearchParamsShim:
    """Minimal v0.18-compatible beam configuration for non-beam CIS runs."""

    beam_width: int = 1
    max_tokens: int = 16
    ignore_eos: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.beam_width = int(kwargs.get("beam_width", 1))
        self.max_tokens = int(kwargs.get("max_tokens", 16))
        self.ignore_eos = bool(kwargs.get("ignore_eos", False))


@dataclass(frozen=True, slots=True)
class _SingleCompletionOutput:
    """Minimal RequestOutput view used to parse one parent child completion."""

    outputs: tuple[Any, ...]
    num_cached_tokens: int
    request_id: str


@dataclass(frozen=True, slots=True)
class VLLMBackendSnapshot:
    sample_calls: int
    score_calls: int
    sampled_sequences: int
    generated_tokens: int
    prefill_tokens: int
    shared_prefill_tokens_saved: int
    scored_tokens: int
    generation_forward_token_slots: int
    score_forward_token_slots: int
    estimated_dense_forward_flops: int
    engine_requests: int
    native_score_sequences: int
    delegated_score_sequences: int
    delegated_score_forward_token_slots: int
    delegated_estimated_dense_forward_flops: int
    maximum_in_flight_requests: int
    active_engine_requests: int = 0
    mh_fused_logprobs: bool = False
    fused_reference_sequences: int = 0
    fused_reference_tokens: int = 0
    native_suffix_speculation: bool = False
    observed_draft_sequences: int = 0
    native_speculative_drafts: int = 0
    native_draft_tokens: int = 0
    native_accepted_draft_tokens: int = 0
    rejected_verification_token_slots: int = 0
    num_preemptions: int = 0
    native_parallel_groups: int = 0
    native_parallel_children: int = 0


class _AsyncLoopRunner:
    """Own one event loop and construct the vLLM engine on that loop's thread."""

    def __init__(self, engine_factory: Callable[[], Any]) -> None:
        self._engine_factory = engine_factory
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._engine: Any | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run_loop,
            name="inference-scaling-vllm",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            raise RuntimeError(
                "failed to initialize the asynchronous vLLM engine"
            ) from self._error

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self._engine = self._engine_factory()
        except BaseException as error:
            self._error = error
            self._ready.set()
            asyncio.set_event_loop(None)
            loop.close()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            asyncio.set_event_loop(None)
            loop.close()

    @property
    def engine(self) -> Any:
        if self._engine is None:
            raise RuntimeError("asynchronous vLLM engine is unavailable")
        return self._engine

    def run(self, coroutine):
        if self._loop is None or not self._thread.is_alive():
            raise RuntimeError("asynchronous vLLM event loop is not running")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def close(self) -> None:
        if self._loop is None or not self._thread.is_alive():
            return

        async def shutdown() -> None:
            callback = getattr(self.engine, "shutdown", None)
            if callback is None:
                return
            result = callback()
            if inspect.isawaitable(result):
                await result

        try:
            self.run(shutdown())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()


def _checkpoint_parameter_count(model_name_or_path: str) -> int | None:
    """Read local safetensor shapes without materializing model weights."""

    root = Path(model_name_or_path)
    if not root.exists():
        return None
    files = [root] if root.is_file() and root.suffix == ".safetensors" else []
    if root.is_dir():
        files = sorted(root.glob("*.safetensors"))
    if not files:
        return None
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    total = 0
    names: set[str] = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in names:
                    raise ValueError(
                        f"duplicate tensor {name!r} across checkpoint shards"
                    )
                names.add(name)
                size = 1
                for dimension in handle.get_slice(name).get_shape():
                    size *= int(dimension)
                total += size
    return total


def _logprob_value(position: Any, token_id: int) -> float:
    if position is None:
        raise RuntimeError("vLLM omitted a required token log-probability")
    try:
        value = position[int(token_id)]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"vLLM did not return the chosen token {token_id} in its log-probabilities"
        ) from error
    return float(getattr(value, "logprob", value))


class VLLMBackend:
    """Synchronous offline vLLM implementation of ``AutoregressiveBackend``.

    The supplied engine must use ``logprobs_mode='processed_logprobs'`` and
    ``generation_config='vllm'``.  Prefer :meth:`from_pretrained`, which enforces
    both settings and explicitly enables automatic prefix caching.
    """

    supports_native_continuous_batching = False

    def __init__(
        self,
        engine: Any,
        tokenizer: Any,
        *,
        model_id: str,
        parameter_count: int,
        sampling_params_factory: Callable[..., Any],
        tokens_prompt_factory: Callable[..., Any] | None = None,
        beam_search_params_factory: Callable[..., Any] | None = None,
        scoring_backend: AutoregressiveBackend | None = None,
        lora_request: Any | None = None,
        draft_tree: RolloutTokenTree | None = None,
        speculation: ActiveBatchSpeculationConfig | None = None,
        native_suffix_speculation: bool = False,
        mh_fused_logprobs: bool = False,
    ) -> None:
        if parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        if scoring_backend is not None and scoring_backend.model_id != model_id:
            raise ValueError("the exact scoring backend must use the same model_id")
        self._engine = engine
        self.tokenizer = tokenizer
        self._model_id = str(model_id)
        self._metric_model_name = self._model_id.split("+adapter:", 1)[0]
        self._parameter_count = int(parameter_count)
        self._sampling_params_factory = sampling_params_factory
        self._tokens_prompt_factory = tokens_prompt_factory
        self._beam_search_params_factory = beam_search_params_factory
        self._scoring_backend = scoring_backend
        self._lora_request = lora_request
        self._speculation = speculation
        # Native suffix decoding owns its global tree inside vLLM.  A separate
        # Python tree would consume CPU and memory without influencing drafts;
        # retain one only when a caller explicitly requests diagnostics.
        self._draft_tree = draft_tree
        self._native_suffix_speculation = bool(native_suffix_speculation)
        self._mh_fused_logprobs = bool(mh_fused_logprobs)
        if self._mh_fused_logprobs and speculation is not None:
            raise ValueError(
                "MH fused log-probabilities currently require speculative decoding "
                "to be disabled"
            )
        pad = getattr(tokenizer, "pad_token_id", None)
        eos = getattr(tokenizer, "eos_token_id", None)
        if pad is None and eos is not None:
            tokenizer.pad_token_id = eos
        self.bos_token_id = (
            None
            if getattr(tokenizer, "bos_token_id", None) is None
            else int(tokenizer.bos_token_id)
        )
        self._engine_lock = threading.RLock()
        self._delegated_score_lock = threading.RLock()
        self._statistics_lock = threading.Lock()
        self._closed = False
        self._sample_calls = 0
        self._score_calls = 0
        self._sampled_sequences = 0
        self._generated_tokens = 0
        self._prefill_tokens = 0
        self._shared_prefill_tokens_saved = 0
        self._scored_tokens = 0
        self._generation_forward_token_slots = 0
        self._score_forward_token_slots = 0
        self._estimated_dense_forward_flops = 0
        self._engine_requests = 0
        self._native_score_sequences = 0
        self._delegated_score_sequences = 0
        self._delegated_score_forward_token_slots = 0
        self._delegated_estimated_dense_forward_flops = 0
        self._active_engine_requests = 0
        self._maximum_in_flight_requests = 0
        self._observed_draft_sequences = 0
        self._fused_reference_sequences = 0
        self._fused_reference_tokens = 0
        self._native_parallel_groups = 0
        self._native_parallel_children = 0

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        adapter_name_or_path: str | None = None,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        quantization: str | None = None,
        enforce_eager: bool = False,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        seed: int = 0,
        parameter_count: int | None = None,
        scoring_backend: AutoregressiveBackend | None = None,
        enable_prefix_caching: bool = True,
        async_scheduling: bool | None = None,
        max_lora_rank: int = 16,
        draft_tree: RolloutTokenTree | None = None,
        speculation: ActiveBatchSpeculationConfig | None = None,
        dynamic_speculation: bool = False,
        enable_mh_fused_logprobs: bool = False,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> "VLLMBackend":
        try:
            from transformers import AutoTokenizer
            from vllm import LLM

            SamplingParams, TokensPrompt, BeamSearchParams = _load_vllm_sampling_api()
        except ImportError as error:  # pragma: no cover - optional GPU installation
            raise ModuleNotFoundError(
                "VLLMBackend.from_pretrained requires the project's vllm extra"
            ) from error

        base_model = model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            local_files_only=Path(base_model).exists(),
            trust_remote_code=trust_remote_code,
            revision=revision,
            cache_dir=download_dir,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if enable_mh_fused_logprobs:
            _validate_mh_fused_vllm_version()
            if speculation is not None:
                raise ValueError(
                    "MH fused log-probabilities cannot be combined with speculative "
                    "decoding"
                )

        kwargs: dict[str, Any] = {
            "model": base_model,
            "dtype": dtype,
            "tensor_parallel_size": int(tensor_parallel_size),
            "pipeline_parallel_size": int(pipeline_parallel_size),
            "data_parallel_size": int(data_parallel_size),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "quantization": quantization,
            "enforce_eager": bool(enforce_eager),
            "trust_remote_code": bool(trust_remote_code),
            "revision": revision,
            "download_dir": download_dir,
            "seed": int(seed),
            "enable_prefix_caching": bool(enable_prefix_caching),
            "generation_config": "vllm",
            "logprobs_mode": "processed_logprobs",
            "enable_lora": adapter_name_or_path is not None,
            "max_lora_rank": int(max_lora_rank),
        }
        if enable_mh_fused_logprobs:
            # The adapter targets vLLM's stable V1 runner in 0.26.  Sampling
            # remains synchronous inside the engine so the selected raw
            # probability is associated with the same request step.
            kwargs["worker_cls"] = _MH_FUSED_WORKER
            kwargs["async_scheduling"] = False
        optional = {
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "async_scheduling": async_scheduling,
        }
        kwargs.update(
            {name: value for name, value in optional.items() if value is not None}
        )
        if speculation is not None:
            kwargs["speculative_config"] = speculation.vllm_suffix_config(
                dynamic=dynamic_speculation
            )
        if engine_kwargs:
            if speculation is not None and "speculative_config" in engine_kwargs:
                raise ValueError(
                    "engine_kwargs.speculative_config conflicts with the explicit "
                    "active-batch speculation config"
                )
            overlap = _PROTECTED_ENGINE_KWARGS.intersection(engine_kwargs)
            if overlap:
                raise ValueError(
                    "engine_kwargs cannot override correctness-critical settings: "
                    + ", ".join(sorted(overlap))
                )
            kwargs.update(engine_kwargs)
        previous_v2_runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
        if enable_mh_fused_logprobs:
            if (
                previous_v2_runner is not None
                and previous_v2_runner.strip().lower()
                in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            ):
                raise ValueError(
                    "MH fused log-probabilities conflict with "
                    "VLLM_USE_V2_MODEL_RUNNER=1"
                )
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        try:
            engine = LLM(**kwargs)
        finally:
            if enable_mh_fused_logprobs:
                if previous_v2_runner is None:
                    os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
                else:
                    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = previous_v2_runner

        lora_request = None
        if adapter_name_or_path is not None:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("inference-scaling", 1, adapter_name_or_path)
        model_id = (
            base_model
            if adapter_name_or_path is None
            else f"{base_model}+adapter:{adapter_name_or_path}"
        )
        counted = parameter_count or _checkpoint_parameter_count(base_model)
        if counted is None:
            raise ValueError(
                "parameter_count could not be read from a local safetensors checkpoint; "
                "pass parameter_count explicitly"
            )
        return cls(
            engine,
            tokenizer,
            model_id=model_id,
            parameter_count=counted,
            sampling_params_factory=SamplingParams,
            tokens_prompt_factory=TokensPrompt,
            beam_search_params_factory=BeamSearchParams,
            scoring_backend=scoring_backend,
            lora_request=lora_request,
            draft_tree=draft_tree,
            speculation=speculation,
            native_suffix_speculation=speculation is not None,
            mh_fused_logprobs=enable_mh_fused_logprobs,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    @property
    def scoring_backend(self) -> AutoregressiveBackend | None:
        """Optional exact backend used for policies vLLM cannot rescore itself."""

        return self._scoring_backend

    def _model_prefix(self, prefix: TokenSequence) -> TokenSequence:
        if prefix:
            return prefix
        if self.bos_token_id is None:
            raise ValueError("an empty prefix requires a tokenizer bos_token_id")
        return (self.bos_token_id,)

    def _prompt(self, prefix: TokenSequence) -> Any:
        token_ids = list(self._model_prefix(prefix))
        if self._tokens_prompt_factory is None:
            return {"prompt_token_ids": token_ids}
        return self._tokens_prompt_factory(prompt_token_ids=token_ids)

    def _sampling_params(self, request: GenerationRequest) -> Any:
        if request.uniforms is not None or request.arithmetic_uniform is not None:
            raise NotImplementedError(
                "vLLM does not expose request-local token uniforms or arithmetic uniforms; "
                "use the Transformers backend for randomized QMC rollouts"
            )
        policy = request.sampling
        return self._sampling_params_factory(
            max_tokens=int(request.max_new_tokens),
            temperature=float(policy.temperature),
            top_p=float(policy.top_p),
            top_k=0 if policy.top_k is None else int(policy.top_k),
            seed=int(request.seed),
            logprobs=int(request.confidence_top_k or 0),
            flat_logprobs=False,
            ignore_eos=True,
            stop_token_ids=(
                [] if policy.eos_token_id is None else [int(policy.eos_token_id)]
            ),
            detokenize=False,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

    def _score_params(self) -> Any:
        return self._sampling_params_factory(
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            seed=0,
            prompt_logprobs=0,
            ignore_eos=True,
            detokenize=False,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

    def _generate(self, prompts: Sequence[Any], params: Any) -> list[Any]:
        if self._closed:
            raise RuntimeError("vLLM backend is closed")
        kwargs = {
            "sampling_params": params,
            "use_tqdm": False,
        }
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        self._engine_requests_started(len(prompts))
        try:
            with self._engine_lock:
                outputs = self._engine.generate(list(prompts), **kwargs)
        finally:
            self._engine_requests_finished(len(prompts))
        return list(outputs)

    def _collect_mh_reference_logprobs(
        self, outputs: Sequence[Any]
    ) -> dict[str, tuple[float, ...]]:
        callback = getattr(self._engine, "collective_rpc", None)
        if callback is None:
            raise RuntimeError(
                "the configured vLLM engine does not expose collective_rpc; "
                "disable mh_fused_logprobs or use the supported offline LLM frontend"
            )
        request_ids: list[str] = []
        for output in outputs:
            request_id = getattr(output, "request_id", None)
            if request_id is None:
                raise RuntimeError(
                    "vLLM omitted the request id required for fused MH accounting"
                )
            request_ids.append(str(request_id))
        if len(set(request_ids)) != len(request_ids):
            raise RuntimeError("vLLM returned duplicate request ids")

        responses = callback(
            "pop_mh_reference_logprobs",
            args=(tuple(request_ids),),
        )
        if inspect.isawaitable(responses):
            raise RuntimeError(
                "MH fused log-probabilities require the synchronous vLLM frontend"
            )
        if isinstance(responses, Mapping):
            worker_responses: Sequence[Any] = (responses,)
        else:
            worker_responses = tuple(responses)

        merged: dict[str, tuple[float, ...]] = {}
        for response in worker_responses:
            if not isinstance(response, Mapping):
                raise RuntimeError(
                    "the fused MH worker returned an invalid probability payload"
                )
            for raw_request_id, raw_values in response.items():
                request_id = str(raw_request_id)
                values = tuple(float(value) for value in raw_values)
                if any(not isfinite(value) for value in values):
                    raise RuntimeError(
                        "the fused MH worker returned a non-finite probability"
                    )
                previous = merged.get(request_id)
                if previous is not None and (
                    len(previous) != len(values)
                    or any(
                        not isclose(left, right, rel_tol=1e-5, abs_tol=1e-6)
                        for left, right in zip(previous, values, strict=True)
                    )
                ):
                    raise RuntimeError(
                        "tensor-parallel vLLM workers disagreed on base log-probabilities"
                    )
                merged[request_id] = values

        missing = [request_id for request_id in request_ids if request_id not in merged]
        if missing:
            raise RuntimeError(
                "the fused MH worker omitted completed requests: " + ", ".join(missing)
            )
        return merged

    def _generate_with_mh_reference(
        self, prompts: Sequence[Any], params: Any
    ) -> tuple[list[Any], dict[str, tuple[float, ...]]]:
        """Generate and drain the worker side channel under one engine lock."""

        if not self._mh_fused_logprobs:
            return self._generate(prompts, params), {}
        if self._closed:
            raise RuntimeError("vLLM backend is closed")
        kwargs = {
            "sampling_params": params,
            "use_tqdm": False,
        }
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        self._engine_requests_started(len(prompts))
        try:
            with self._engine_lock:
                outputs = list(self._engine.generate(list(prompts), **kwargs))
                references = self._collect_mh_reference_logprobs(outputs)
        finally:
            self._engine_requests_finished(len(prompts))
        return outputs, references

    def _engine_requests_started(self, count: int) -> None:
        with self._statistics_lock:
            self._active_engine_requests += int(count)
            self._maximum_in_flight_requests = max(
                self._maximum_in_flight_requests,
                self._active_engine_requests,
            )

    def _engine_requests_finished(self, count: int) -> None:
        with self._statistics_lock:
            self._active_engine_requests -= int(count)
            if self._active_engine_requests < 0:
                raise RuntimeError("vLLM in-flight request accounting became negative")

    def active_engine_request_count(self) -> int:
        """Return local admission occupancy without querying engine metrics."""

        with self._statistics_lock:
            return self._active_engine_requests

    @staticmethod
    def _sum_metric_values(
        metrics: Any, *, model_name: str | None = None
    ) -> tuple[int, int, int, int]:
        totals = {
            "vllm:spec_decode_num_drafts": 0.0,
            "vllm:spec_decode_num_draft_tokens": 0.0,
            "vllm:spec_decode_num_accepted_tokens": 0.0,
            "vllm:num_preemptions": 0.0,
        }
        for metric in metrics or ():
            labels = getattr(metric, "labels", {}) or {}
            if (
                model_name is not None
                and labels.get("model_name") is not None
                and str(labels["model_name"]) != model_name
            ):
                continue
            name = str(getattr(metric, "name", ""))
            if name == "vllm:num_preemptions_total":
                name = "vllm:num_preemptions"
            if name not in totals:
                continue
            value = getattr(metric, "value", 0.0)
            if isinstance(value, (list, tuple)):
                totals[name] += sum(float(item) for item in value)
            else:
                totals[name] += float(value)
        return (
            int(totals["vllm:spec_decode_num_drafts"]),
            int(totals["vllm:spec_decode_num_draft_tokens"]),
            int(totals["vllm:spec_decode_num_accepted_tokens"]),
            int(totals["vllm:num_preemptions"]),
        )

    def _engine_metric_totals(self) -> tuple[int, int, int, int]:
        callback = getattr(self._engine, "get_metrics", None)
        if callback is None or self._closed:
            return 0, 0, 0, 0
        try:
            metrics = callback()
        except (AssertionError, RuntimeError):
            return 0, 0, 0, 0
        if inspect.isawaitable(metrics):
            raise RuntimeError("an asynchronous metrics API requires AsyncVLLMBackend")
        return self._sum_metric_values(metrics, model_name=self._metric_model_name)

    @staticmethod
    def _completion(output: Any) -> Any:
        completions = getattr(output, "outputs", None)
        if not completions or len(completions) != 1:
            raise RuntimeError("vLLM must return exactly one completion per request")
        return completions[0]

    def _sample_from_output(
        self,
        request: GenerationRequest,
        output: Any,
        reference_token_logprobs: Sequence[float] | None = None,
    ) -> tuple[SequenceSample, int, int, int]:
        completion = self._completion(output)
        tokens = tuple(int(token) for token in completion.token_ids)
        positions = completion.logprobs
        if positions is None or len(positions) != len(tokens):
            raise RuntimeError(
                "vLLM returned an invalid generated log-probability shape"
            )
        token_logprobs = tuple(
            _logprob_value(position, token)
            for position, token in zip(positions, tokens, strict=True)
        )
        token_topk_confidences: tuple[float, ...] | None = None
        effective_top_k: int | None = None
        if request.confidence_top_k is not None:
            confidence_values: list[float] = []
            for position in positions:
                if not isinstance(position, Mapping):
                    raise RuntimeError("vLLM top-K log-probabilities must be a mapping")
                logprobs = sorted(
                    (
                        float(getattr(value, "logprob", value))
                        for value in position.values()
                    ),
                    reverse=True,
                )
                effective = request.confidence_top_k
                if len(logprobs) < effective:
                    raise RuntimeError("vLLM omitted requested top-K log-probabilities")
                confidence_values.append(-sum(logprobs[:effective]) / effective)
                effective_top_k = effective
            token_topk_confidences = tuple(confidence_values)
        reference_values: tuple[float, ...] | None = None
        if reference_token_logprobs is not None:
            reference_values = tuple(float(value) for value in reference_token_logprobs)
        else:
            raw_positions = getattr(completion, "reference_logprobs", None)
            if raw_positions is None:
                # Accept the earlier fused-worker payload name as well.
                raw_positions = getattr(completion, "power_logprobs", None)
            if raw_positions is not None:
                if len(raw_positions) != len(tokens):
                    raise RuntimeError(
                        "vLLM returned an invalid reference log-probability shape"
                    )
                parsed_reference: list[float] = []
                for position, token in zip(raw_positions, tokens, strict=True):
                    if isinstance(position, Mapping):
                        parsed_reference.append(_logprob_value(position, token))
                    else:
                        parsed_reference.append(
                            float(getattr(position, "logprob", position))
                        )
                reference_values = tuple(parsed_reference)

        reference_sampling = SamplingConfig(eos_token_id=request.sampling.eos_token_id)
        if reference_values is None and request.sampling == reference_sampling:
            reference_values = token_logprobs
        if reference_values is not None:
            if len(reference_values) != len(tokens):
                raise RuntimeError(
                    "vLLM returned an invalid reference log-probability shape"
                )
            if any(not isfinite(value) for value in reference_values):
                raise RuntimeError(
                    "vLLM returned a non-finite reference log-probability"
                )
        finish_reason = str(getattr(completion, "finish_reason", "length") or "length")
        eos = request.sampling.eos_token_id
        if eos is not None and tokens and tokens[-1] == eos:
            finish_reason = "eos"
        sample = SequenceSample(
            prefix=request.prefix,
            token_ids=tokens,
            token_logprobs=token_logprobs,
            policy_id=request.sampling.policy_id,
            model_id=self.model_id,
            request_id=request.request_id,
            finish_reason=finish_reason,
            reference_token_logprobs=reference_values,
            reference_policy_id=(
                None if reference_values is None else reference_sampling.policy_id
            ),
            token_topk_confidences=token_topk_confidences,
            confidence_top_k=effective_top_k,
        )
        if (
            getattr(completion, "stop_reason", None) == "cis_parent_terminal"
            and not tokens
        ):
            # Parked rollout placeholders cancelled by a terminal parent never
            # enter the model. Do not charge their logical prompt as prefill.
            return sample, 0, 0, 0
        prompt_length = len(self._model_prefix(request.prefix))
        cached = min(prompt_length, int(getattr(output, "num_cached_tokens", 0) or 0))
        return (
            sample,
            prompt_length - cached,
            cached,
            prompt_length - cached + max(0, len(tokens) - 1),
        )

    def _record_sample_batch(
        self,
        samples: Sequence[SequenceSample],
        *,
        prefill_tokens: int,
        cached_tokens: int,
        forward_slots: int,
    ) -> None:
        with self._statistics_lock:
            self._sample_calls += 1
            self._sampled_sequences += len(samples)
            self._generated_tokens += sum(len(sample.token_ids) for sample in samples)
            self._prefill_tokens += prefill_tokens
            self._shared_prefill_tokens_saved += cached_tokens
            self._generation_forward_token_slots += forward_slots
            self._estimated_dense_forward_flops += dense_forward_flops(
                self.parameter_count, forward_slots
            )
            self._engine_requests += len(samples)
            fused = [
                sample
                for sample in samples
                if sample.reference_token_logprobs is not None
                and sample.reference_policy_id != sample.policy_id
            ]
            self._fused_reference_sequences += len(fused)
            self._fused_reference_tokens += sum(
                len(sample.token_ids) for sample in fused
            )

    def sample_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[SequenceSample]:
        if not requests:
            return []
        outputs, references = self._generate_with_mh_reference(
            [self._prompt(request.prefix) for request in requests],
            [self._sampling_params(request) for request in requests],
        )
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM returned an invalid number of request outputs")
        parsed = []
        for request, output in zip(requests, outputs, strict=True):
            output_request_id = getattr(output, "request_id", None)
            reference = (
                None
                if output_request_id is None
                else references.get(str(output_request_id))
            )
            parsed.append(self._sample_from_output(request, output, reference))
        samples = [item[0] for item in parsed]
        self._record_sample_batch(
            samples,
            prefill_tokens=sum(item[1] for item in parsed),
            cached_tokens=sum(item[2] for item in parsed),
            forward_slots=sum(item[3] for item in parsed),
        )
        self.observe_draft_samples(samples)
        return samples

    def sample_batch_with_callback(
        self,
        requests: Sequence[GenerationRequest],
        on_complete: SampleCompletionCallback,
    ) -> list[SequenceSample]:
        """Synchronous LLM fallback; AsyncVLLMBackend overrides this hook."""

        samples = self.sample_batch(requests)
        for index, sample in enumerate(samples):
            on_complete(index, sample)
        return samples

    def observe_draft_samples(self, samples: Iterable[SequenceSample]) -> None:
        materialized = tuple(samples)
        if self._draft_tree is not None:
            self._draft_tree.observe_samples(materialized)
        with self._statistics_lock:
            self._observed_draft_sequences += len(materialized)

    def observe_draft_sequences(self, sequences: Iterable[TokenSequence]) -> None:
        materialized = tuple(
            tuple(int(token) for token in sequence) for sequence in sequences
        )
        if self._draft_tree is not None:
            for sequence in materialized:
                self._draft_tree.observe(sequence)
        with self._statistics_lock:
            self._observed_draft_sequences += len(materialized)

    def draft_cache_snapshot(self) -> RolloutTokenTreeSnapshot | None:
        return None if self._draft_tree is None else self._draft_tree.snapshot()

    @staticmethod
    def _supports_native_score(sampling: SamplingConfig | None) -> bool:
        policy = sampling or SamplingConfig()
        return policy.temperature == 1 and policy.top_p == 1 and policy.top_k is None

    def _score_native(
        self,
        items: Sequence[tuple[int, ScoreRequest, TokenSequence]],
        results: list[tuple[float, ...]],
    ) -> tuple[int, int, int]:
        if not items:
            return 0, 0, 0
        prompts = [
            self._prompt(self._model_prefix(request.prefix) + continuation)
            for _, request, continuation in items
        ]
        outputs = self._generate(prompts, self._score_params())
        if len(outputs) != len(items):
            raise RuntimeError("vLLM returned an invalid number of scoring outputs")
        forward_slots = 0
        cached_tokens = 0
        for (index, request, continuation), output in zip(items, outputs, strict=True):
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            prefix = self._model_prefix(request.prefix)
            if prompt_logprobs is None or len(prompt_logprobs) != len(prefix) + len(
                continuation
            ):
                raise RuntimeError(
                    "vLLM returned an invalid prompt log-probability shape"
                )
            positions = prompt_logprobs[len(prefix) :]
            results[index] = tuple(
                _logprob_value(position, token)
                for position, token in zip(positions, continuation, strict=True)
            )
            prompt_length = len(prefix) + len(continuation)
            cached = min(
                prompt_length,
                int(getattr(output, "num_cached_tokens", 0) or 0),
            )
            forward_slots += prompt_length - cached
            cached_tokens += cached
        return len(items), forward_slots, cached_tokens

    def _delegated_compute_delta(
        self,
        before: Any | None,
        after: Any | None,
        requests: Sequence[ScoreRequest],
    ) -> tuple[int, int]:
        if before is not None and after is not None:
            slots = int(after.score_forward_token_slots) - int(
                before.score_forward_token_slots
            )
            flops = int(after.estimated_dense_forward_flops) - int(
                before.estimated_dense_forward_flops
            )
            return slots, flops
        slots = sum(
            len(self._model_prefix(request.prefix)) + len(continuation)
            for request in requests
            for continuation in request.continuations
        )
        return slots, dense_forward_flops(self.parameter_count, slots)

    def _run_delegated_score(
        self,
        requests: Sequence[ScoreRequest],
        method: str,
        **kwargs: Any,
    ):
        if self._scoring_backend is None:
            raise RuntimeError("an exact scoring backend is not configured")
        callback = getattr(self._scoring_backend, method, None)
        if callback is None:
            raise ValueError(
                f"the configured exact scoring backend does not implement {method}"
            )
        snapshot = getattr(self._scoring_backend, "snapshot", None)
        with self._delegated_score_lock:
            before = snapshot() if snapshot is not None else None
            outputs = callback(requests, **kwargs)
            after = snapshot() if snapshot is not None else None
        slots, flops = self._delegated_compute_delta(before, after, requests)
        return outputs, slots, flops

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        flattened = [
            (request, continuation)
            for request in requests
            for continuation in request.continuations
        ]
        results: list[tuple[float, ...]] = [()] * len(flattened)
        native: list[tuple[int, ScoreRequest, TokenSequence]] = []
        delegated: list[tuple[int, ScoreRequest, TokenSequence]] = []
        for index, (request, continuation) in enumerate(flattened):
            if not continuation:
                continue
            item = (index, request, continuation)
            if self._supports_native_score(request.sampling):
                native.append(item)
            else:
                delegated.append(item)

        native_count, score_slots, cached_tokens = self._score_native(native, results)
        delegated_slots = 0
        delegated_flops = 0
        if delegated:
            if self._scoring_backend is None:
                policies = sorted(
                    {
                        item[1].sampling.policy_id
                        for item in delegated
                        if item[1].sampling
                    }
                )
                raise ValueError(
                    "vLLM prompt log-probabilities cannot exactly score temperature/top-k/top-p "
                    "policies; configure an exact scoring_backend for: "
                    + ", ".join(policies)
                )
            delegated_requests = [
                ScoreRequest(request.prefix, (continuation,), request.sampling)
                for _, request, continuation in delegated
            ]
            delegated_outputs, delegated_slots, delegated_flops = (
                self._run_delegated_score(
                    delegated_requests,
                    "score_batch",
                )
            )
            if len(delegated_outputs) != len(delegated):
                raise RuntimeError(
                    "exact scoring backend returned an invalid result count"
                )
            for (index, _, continuation), scores in zip(
                delegated, delegated_outputs, strict=True
            ):
                if len(scores) != len(continuation):
                    raise RuntimeError(
                        "exact scoring backend returned an invalid score shape"
                    )
                results[index] = scores

        with self._statistics_lock:
            self._score_calls += 1
            self._scored_tokens += sum(
                len(continuation) for _, continuation in flattened
            )
            self._shared_prefill_tokens_saved += cached_tokens
            self._score_forward_token_slots += score_slots + delegated_slots
            self._estimated_dense_forward_flops += (
                dense_forward_flops(self.parameter_count, score_slots) + delegated_flops
            )
            self._engine_requests += native_count
            self._native_score_sequences += native_count
            self._delegated_score_sequences += len(delegated)
            self._delegated_score_forward_token_slots += delegated_slots
            self._delegated_estimated_dense_forward_flops += delegated_flops
        return results

    def score_statistics_batch(
        self,
        requests: Sequence[ScoreRequest],
        *,
        confidence_top_k: int | None = None,
    ) -> list[Any]:
        """Delegate full-vocabulary confidence statistics to the exact backend.

        Selected-token prompt log-probabilities are enough for IS and MH at the
        base policy, but entropy and self-certainty require the whole vocabulary.
        Keeping that operation explicit prevents a vLLM speed setting from
        silently changing a confidence reward.
        """

        flattened = [
            continuation
            for request in requests
            for continuation in request.continuations
        ]
        outputs, slots, flops = self._run_delegated_score(
            requests,
            "score_statistics_batch",
            confidence_top_k=confidence_top_k,
        )
        if len(outputs) != len(flattened):
            raise RuntimeError("exact scoring backend returned an invalid result count")
        with self._statistics_lock:
            self._score_calls += 1
            self._scored_tokens += sum(len(continuation) for continuation in flattened)
            self._score_forward_token_slots += slots
            self._estimated_dense_forward_flops += flops
            self._delegated_score_sequences += len(flattened)
            self._delegated_score_forward_token_slots += slots
            self._delegated_estimated_dense_forward_flops += flops
        return list(outputs)

    def snapshot(self) -> VLLMBackendSnapshot:
        drafts, draft_tokens, accepted, preemptions = self._engine_metric_totals()
        rejected = max(0, draft_tokens - accepted)
        with self._statistics_lock:
            return VLLMBackendSnapshot(
                sample_calls=self._sample_calls,
                score_calls=self._score_calls,
                sampled_sequences=self._sampled_sequences,
                generated_tokens=self._generated_tokens,
                prefill_tokens=self._prefill_tokens,
                shared_prefill_tokens_saved=self._shared_prefill_tokens_saved,
                scored_tokens=self._scored_tokens,
                generation_forward_token_slots=(
                    self._generation_forward_token_slots + rejected
                ),
                score_forward_token_slots=self._score_forward_token_slots,
                estimated_dense_forward_flops=(
                    self._estimated_dense_forward_flops
                    + dense_forward_flops(self.parameter_count, rejected)
                ),
                engine_requests=self._engine_requests,
                native_score_sequences=self._native_score_sequences,
                delegated_score_sequences=self._delegated_score_sequences,
                delegated_score_forward_token_slots=(
                    self._delegated_score_forward_token_slots
                ),
                delegated_estimated_dense_forward_flops=(
                    self._delegated_estimated_dense_forward_flops
                ),
                maximum_in_flight_requests=self._maximum_in_flight_requests,
                active_engine_requests=self._active_engine_requests,
                mh_fused_logprobs=self._mh_fused_logprobs,
                fused_reference_sequences=self._fused_reference_sequences,
                fused_reference_tokens=self._fused_reference_tokens,
                native_suffix_speculation=self._native_suffix_speculation,
                observed_draft_sequences=self._observed_draft_sequences,
                native_speculative_drafts=drafts,
                native_draft_tokens=draft_tokens,
                native_accepted_draft_tokens=accepted,
                rejected_verification_token_slots=rejected,
                num_preemptions=preemptions,
                native_parallel_groups=self._native_parallel_groups,
                native_parallel_children=self._native_parallel_children,
            )

    def encode(self, text: str, *, add_special_tokens: bool = True) -> TokenSequence:
        return tuple(
            int(token)
            for token in self.tokenizer.encode(
                text, add_special_tokens=add_special_tokens
            )
        )

    def decode(self, tokens: TokenSequence, *, skip_special_tokens: bool = True) -> str:
        return str(
            self.tokenizer.decode(list(tokens), skip_special_tokens=skip_special_tokens)
        )

    def direct_generate(
        self,
        prefix: TokenSequence,
        *,
        max_new_tokens: int,
        num_beams: int = 1,
    ) -> TokenSequence:
        """Run the greedy/beam baseline without changing algorithm sampling."""

        if max_new_tokens <= 0 or num_beams <= 0:
            raise ValueError("generation length and beam count must be positive")
        model_prefix = self._model_prefix(prefix)
        prompt = self._prompt(model_prefix)
        if num_beams == 1:
            params = self._sampling_params_factory(
                max_tokens=int(max_new_tokens),
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                seed=0,
                ignore_eos=False,
                stop_token_ids=[],
                detokenize=False,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )
            outputs = self._generate([prompt], params)
            if len(outputs) != 1:
                raise RuntimeError("vLLM returned an invalid greedy output count")
            return tuple(int(token) for token in self._completion(outputs[0]).token_ids)

        beam_search = getattr(self._engine, "beam_search", None)
        if beam_search is None or self._beam_search_params_factory is None:
            delegated = getattr(self._scoring_backend, "direct_generate", None)
            if delegated is not None:
                return tuple(
                    delegated(
                        prefix,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                    )
                )
            raise ValueError(
                "beam search requires runtime.backend='vllm-sync' or a "
                "Transformers exact_scoring_backend"
            )
        kwargs: dict[str, Any] = {
            "prompts": [prompt],
            "params": self._beam_search_params_factory(
                beam_width=int(num_beams),
                max_tokens=int(max_new_tokens),
                ignore_eos=False,
            ),
            "use_tqdm": False,
        }
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        outputs = list(beam_search(**kwargs))
        if len(outputs) != 1 or not getattr(outputs[0], "sequences", None):
            raise RuntimeError("vLLM returned an invalid beam-search output")
        full_tokens = tuple(int(token) for token in outputs[0].sequences[0].tokens)
        if full_tokens[: len(model_prefix)] != model_prefix:
            raise RuntimeError("vLLM beam-search output did not preserve the prompt")
        return full_tokens[len(model_prefix) :]

    def start_profile(self, profile_prefix: str | None = None) -> None:
        callback = getattr(self._engine, "start_profile", None)
        if callback is None:
            raise RuntimeError("this vLLM frontend does not expose start_profile")
        result = callback(profile_prefix)
        if inspect.isawaitable(result):
            raise RuntimeError(
                "an asynchronous vLLM engine requires AsyncVLLMBackend profiling"
            )

    def stop_profile(self) -> None:
        callback = getattr(self._engine, "stop_profile", None)
        if callback is None:
            raise RuntimeError("this vLLM frontend does not expose stop_profile")
        result = callback()
        if inspect.isawaitable(result):
            raise RuntimeError(
                "an asynchronous vLLM engine requires AsyncVLLMBackend profiling"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        shutdown = getattr(self._engine, "shutdown", None)
        if shutdown is None:
            shutdown = getattr(
                getattr(self._engine, "llm_engine", None), "shutdown", None
            )
        if shutdown is not None:
            result = shutdown()
            if inspect.isawaitable(result):
                raise RuntimeError(
                    "an asynchronous vLLM engine requires AsyncVLLMBackend"
                )

    def __enter__(self) -> "VLLMBackend":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


class AsyncVLLMBackend(VLLMBackend):
    """Persistent AsyncLLM backend with cross-caller continuous batching."""

    supports_native_continuous_batching = True

    def __init__(
        self,
        engine: Any | None,
        tokenizer: Any,
        *,
        model_id: str,
        parameter_count: int,
        sampling_params_factory: Callable[..., Any],
        tokens_prompt_factory: Callable[..., Any] | None = None,
        beam_search_params_factory: Callable[..., Any] | None = None,
        scoring_backend: AutoregressiveBackend | None = None,
        lora_request: Any | None = None,
        engine_factory: Callable[[], Any] | None = None,
        draft_tree: RolloutTokenTree | None = None,
        speculation: ActiveBatchSpeculationConfig | None = None,
        native_suffix_speculation: bool = False,
        request_priority_policy: str = "none",
        native_parallel_sampling: bool = False,
        native_segmented_rng: bool = False,
        native_kv_fork: bool = False,
        native_kv_fork_waiters: bool = False,
        native_kv_fork_compact_waiters: bool = False,
        native_kv_fork_lease: bool = False,
        native_kv_fork_lease_scope: str = "candidate_suffix",
        native_kv_fork_lease_max_fraction: float | None = None,
        native_kv_branch_eviction: bool = False,
        native_kv_resample_gc: bool = False,
    ) -> None:
        if (engine is None) == (engine_factory is None):
            raise ValueError("provide exactly one of engine or engine_factory")
        self._runner = _AsyncLoopRunner(
            engine_factory if engine_factory is not None else lambda: engine
        )
        self._request_counter = itertools.count()
        self._request_counter_lock = threading.Lock()
        super().__init__(
            self._runner.engine,
            tokenizer,
            model_id=model_id,
            parameter_count=parameter_count,
            sampling_params_factory=sampling_params_factory,
            tokens_prompt_factory=tokens_prompt_factory,
            beam_search_params_factory=beam_search_params_factory,
            scoring_backend=scoring_backend,
            lora_request=lora_request,
            draft_tree=draft_tree,
            speculation=speculation,
            native_suffix_speculation=native_suffix_speculation,
        )
        self._request_trace_observer: Callable[[Mapping[str, Any]], None] | None = None
        if request_priority_policy not in {"none", "step_fifo", "rollout_first"}:
            raise ValueError("unknown CIS request priority policy")
        self._request_priority_policy = request_priority_policy
        self._native_parallel_sampling = bool(native_parallel_sampling)
        self._native_segmented_rng = bool(native_segmented_rng)
        self._native_kv_fork = bool(native_kv_fork)
        self._native_kv_fork_waiters = bool(native_kv_fork_waiters)
        self._native_kv_fork_compact_waiters = bool(
            native_kv_fork_compact_waiters
        )
        self._native_kv_fork_lease = bool(native_kv_fork_lease)
        if native_kv_fork_lease_scope not in {"candidate_suffix", "full_parent"}:
            raise ValueError("unknown native KV fork lease scope")
        self._native_kv_fork_lease_scope = native_kv_fork_lease_scope
        if native_kv_fork_lease_max_fraction is not None and not (
            0.0 < native_kv_fork_lease_max_fraction <= 1.0
        ):
            raise ValueError("native KV fork lease fraction must be in (0, 1]")
        self._native_kv_fork_lease_max_fraction = (
            native_kv_fork_lease_max_fraction
        )
        self._native_kv_branch_eviction = bool(native_kv_branch_eviction)
        self._native_kv_resample_gc = bool(native_kv_resample_gc)
        if self._native_kv_fork_lease and not self._native_kv_fork:
            raise ValueError("native_kv_fork_lease requires native_kv_fork")
        if self._native_kv_fork_waiters and not self._native_kv_fork_lease:
            raise ValueError("native_kv_fork_waiters requires a native KV fork lease")
        if (
            self._native_kv_fork_compact_waiters
            and not self._native_kv_fork_waiters
        ):
            raise ValueError(
                "native_kv_fork_compact_waiters requires native_kv_fork_waiters"
            )
        if (
            self._native_kv_fork_lease_scope != "candidate_suffix"
            and not self._native_kv_fork_lease
        ):
            raise ValueError(
                "native_kv_fork_lease_scope requires native_kv_fork_lease"
            )
        if (
            self._native_kv_fork_lease_max_fraction is not None
            and not self._native_kv_fork_lease
        ):
            raise ValueError(
                "native_kv_fork_lease_max_fraction requires native_kv_fork_lease"
            )
        if self._native_parallel_sampling and self._native_kv_fork:
            raise ValueError(
                "native_parallel_sampling and native_kv_fork are separate experiments"
            )
        self._step_priority_lock = threading.Lock()
        self._step_priorities: dict[str, int] = {}
        self._next_step_priority = itertools.count()

    def set_request_trace_observer(
        self,
        observer: Callable[[Mapping[str, Any]], None] | None,
    ) -> None:
        """Observe request-level lifecycle events without changing scheduling."""

        self._request_trace_observer = observer

    def _observe_request(self, event: Mapping[str, Any]) -> None:
        observer = self._request_trace_observer
        if observer is not None:
            observer(event)

    def _request_priority(self, request: GenerationRequest | None) -> int:
        if request is None or self._request_priority_policy == "none":
            return 0
        match = _CIS_STEP_REQUEST.match(request.request_id)
        if match is None:
            return 0
        step_key = f"{match.group('job')}:step:{match.group('step')}"
        with self._step_priority_lock:
            priority = self._step_priorities.get(step_key)
            if priority is None:
                priority = next(self._next_step_priority)
                self._step_priorities[step_key] = priority
        if (
            self._request_priority_policy == "rollout_first"
            and match.group("rollout") is not None
        ):
            return priority - 1_000_000
        return priority

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        adapter_name_or_path: str | None = None,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        quantization: str | None = None,
        enforce_eager: bool = False,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        seed: int = 0,
        parameter_count: int | None = None,
        scoring_backend: AutoregressiveBackend | None = None,
        enable_prefix_caching: bool = True,
        async_scheduling: bool | None = None,
        max_lora_rank: int = 16,
        draft_tree: RolloutTokenTree | None = None,
        speculation: ActiveBatchSpeculationConfig | None = None,
        dynamic_speculation: bool = False,
        engine_kwargs: dict[str, Any] | None = None,
        request_priority_policy: str = "none",
        native_parallel_sampling: bool = False,
        native_segmented_rng: bool = False,
        native_kv_fork: bool = False,
        native_kv_fork_waiters: bool = False,
        native_kv_fork_compact_waiters: bool = False,
        native_kv_fork_lease: bool = False,
        native_kv_fork_lease_scope: str = "candidate_suffix",
        native_kv_fork_lease_max_fraction: float | None = None,
        native_kv_branch_eviction: bool = False,
        native_kv_resample_gc: bool = False,
    ) -> "AsyncVLLMBackend":
        try:
            from transformers import AutoTokenizer
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM
            from vllm.v1.engine.parallel_sampling import ParentRequest
            from vllm.v1.core.kv_cache_manager import KVCacheManager
            from vllm.v1.core.sched.scheduler import Scheduler
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner

            SamplingParams, TokensPrompt, BeamSearchParams = _load_vllm_sampling_api()
        except ImportError as error:  # pragma: no cover - optional GPU installation
            raise ModuleNotFoundError(
                "AsyncVLLMBackend.from_pretrained requires the project's vllm extra"
            ) from error
        if native_parallel_sampling and not bool(
            getattr(ParentRequest, "cis_child_final_streaming_supported", False)
        ):
            raise RuntimeError(
                "native_parallel_sampling requires the CIS ParentRequest runtime patch"
            )
        if native_segmented_rng and not bool(
            getattr(GPUModelRunner, "cis_segmented_rng_supported", False)
        ):
            raise RuntimeError(
                "native_segmented_rng requires the CIS segmented RNG runtime patch"
            )
        if native_kv_fork and not bool(
            getattr(KVCacheManager, "cis_fork_supported", False)
        ):
            raise RuntimeError("native_kv_fork requires the CIS KV fork runtime patch")
        if native_kv_fork_waiters and not bool(
            getattr(Scheduler, "cis_fork_waiter_supported", False)
        ):
            raise RuntimeError(
                "native_kv_fork_waiters requires the CIS fork waiter runtime patch"
            )
        if native_kv_fork_compact_waiters and not bool(
            getattr(Scheduler, "cis_fork_compact_waiter_supported", False)
        ):
            raise RuntimeError(
                "native_kv_fork_compact_waiters requires the compact waiter patch"
            )
        if native_kv_fork_lease and not bool(
            getattr(KVCacheManager, "cis_fork_lease_supported", False)
        ):
            raise RuntimeError(
                "native_kv_fork_lease requires the CIS KV fork lease runtime patch"
            )
        if native_kv_fork_lease_scope == "full_parent" and not bool(
            getattr(
                KVCacheManager,
                "cis_fork_full_parent_handoff_supported",
                False,
            )
        ):
            raise RuntimeError(
                "full-parent KV handoff requires the matching CIS runtime patch"
            )
        if native_kv_fork_lease_max_fraction is not None and not bool(
            getattr(KVCacheManager, "cis_fork_lease_budget_supported", False)
        ):
            raise RuntimeError(
                "bounded KV fork handoff requires the matching CIS runtime patch"
            )
        if native_kv_branch_eviction and not bool(
            getattr(KVCacheManager, "cis_branch_eviction_supported", False)
        ):
            raise RuntimeError(
                "native_kv_branch_eviction requires the CIS branch eviction runtime patch"
            )
        if native_kv_resample_gc and not bool(
            getattr(KVCacheManager, "cis_resample_gc_supported", False)
        ):
            raise RuntimeError(
                "native_kv_resample_gc requires the CIS resample GC runtime patch"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=Path(model_name_or_path).exists(),
            trust_remote_code=trust_remote_code,
            revision=revision,
            cache_dir=download_dir,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        kwargs: dict[str, Any] = {
            "model": model_name_or_path,
            "dtype": dtype,
            "tensor_parallel_size": int(tensor_parallel_size),
            "pipeline_parallel_size": int(pipeline_parallel_size),
            "data_parallel_size": int(data_parallel_size),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "quantization": quantization,
            "enforce_eager": bool(enforce_eager),
            "trust_remote_code": bool(trust_remote_code),
            "revision": revision,
            "download_dir": download_dir,
            "seed": int(seed),
            "enable_prefix_caching": bool(enable_prefix_caching),
            "generation_config": "vllm",
            "logprobs_mode": "processed_logprobs",
            "enable_lora": adapter_name_or_path is not None,
            "max_lora_rank": int(max_lora_rank),
        }
        optional = {
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "async_scheduling": async_scheduling,
        }
        kwargs.update(
            {name: value for name, value in optional.items() if value is not None}
        )
        if speculation is not None:
            kwargs["speculative_config"] = speculation.vllm_suffix_config(
                dynamic=dynamic_speculation
            )
        if request_priority_policy != "none":
            kwargs["scheduling_policy"] = "priority"
        if engine_kwargs:
            if speculation is not None and "speculative_config" in engine_kwargs:
                raise ValueError(
                    "engine_kwargs.speculative_config conflicts with the explicit "
                    "active-batch speculation config"
                )
            overlap = _PROTECTED_ENGINE_KWARGS.intersection(engine_kwargs)
            if overlap:
                raise ValueError(
                    "engine_kwargs cannot override correctness-critical settings: "
                    + ", ".join(sorted(overlap))
                )
            kwargs.update(engine_kwargs)

        lora_request = None
        if adapter_name_or_path is not None:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("inference-scaling", 1, adapter_name_or_path)
        model_id = (
            model_name_or_path
            if adapter_name_or_path is None
            else f"{model_name_or_path}+adapter:{adapter_name_or_path}"
        )
        counted = parameter_count or _checkpoint_parameter_count(model_name_or_path)
        if counted is None:
            raise ValueError(
                "parameter_count could not be read from a local safetensors checkpoint; "
                "pass parameter_count explicitly"
            )

        def create_engine():
            return AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))

        return cls(
            None,
            tokenizer,
            model_id=model_id,
            parameter_count=counted,
            sampling_params_factory=SamplingParams,
            tokens_prompt_factory=TokensPrompt,
            beam_search_params_factory=BeamSearchParams,
            scoring_backend=scoring_backend,
            lora_request=lora_request,
            engine_factory=create_engine,
            draft_tree=draft_tree,
            speculation=speculation,
            native_suffix_speculation=speculation is not None,
            request_priority_policy=request_priority_policy,
            native_parallel_sampling=native_parallel_sampling,
            native_segmented_rng=native_segmented_rng,
            native_kv_fork=native_kv_fork,
            native_kv_fork_waiters=native_kv_fork_waiters,
            native_kv_fork_compact_waiters=native_kv_fork_compact_waiters,
            native_kv_fork_lease=native_kv_fork_lease,
            native_kv_fork_lease_scope=native_kv_fork_lease_scope,
            native_kv_fork_lease_max_fraction=native_kv_fork_lease_max_fraction,
            native_kv_branch_eviction=native_kv_branch_eviction,
            native_kv_resample_gc=native_kv_resample_gc,
        )

    def _sampling_params(self, request: GenerationRequest) -> Any:
        params = super()._sampling_params(request)
        if request.rng_switch_after_tokens is not None:
            assert request.rng_switch_seed is not None
            if not self._native_segmented_rng:
                raise RuntimeError(
                    "segmented RNG requests require native_segmented_rng=true"
                )
            extra_args = dict(getattr(params, "extra_args", None) or {})
            extra_args["cis_rng_switch_after_tokens"] = int(
                request.rng_switch_after_tokens
            )
            extra_args["cis_rng_switch_seed"] = int(request.rng_switch_seed)
            if request.rng_prefix_group is not None:
                assert request.rng_prefix_group_size is not None
                extra_args["cis_rng_prefix_group"] = request.rng_prefix_group
                extra_args["cis_rng_prefix_group_size"] = int(
                    request.rng_prefix_group_size
                )
            params.extra_args = extra_args
        if (
            not self._native_kv_fork
            and not self._native_kv_branch_eviction
            and not self._native_kv_resample_gc
        ):
            return params
        extra_args = dict(getattr(params, "extra_args", None) or {})
        if self._native_kv_fork and request.fork_expected_children:
            extra_args["cis_fork_handle"] = request.request_id
            extra_args["cis_fork_expected_children"] = int(
                request.fork_expected_children
            )
            if request.fork_group_id is not None:
                extra_args["cis_fork_group_id"] = request.fork_group_id
                extra_args["cis_fork_group_size"] = int(request.fork_group_size or 0)
                extra_args["cis_fork_release_remaining"] = int(
                    request.fork_release_remaining or 0
                )
                if request.fork_adaptive_release:
                    extra_args["cis_fork_adaptive_release"] = True
                if request.fork_adaptive_runnable_fraction is not None:
                    extra_args["cis_fork_adaptive_runnable_fraction"] = float(
                        request.fork_adaptive_runnable_fraction
                    )
            if self._native_kv_fork_lease:
                extra_args["cis_fork_shared_prefix_tokens"] = (
                    0
                    if self._native_kv_fork_lease_scope == "full_parent"
                    else len(request.prefix)
                )
                extra_args["cis_fork_lease_scope"] = self._native_kv_fork_lease_scope
                extra_args["cis_fork_candidate_prefix_tokens"] = len(request.prefix)
                if self._native_kv_fork_lease_max_fraction is not None:
                    extra_args["cis_fork_lease_max_fraction"] = (
                        self._native_kv_fork_lease_max_fraction
                    )
                # A saturated CIS step can queue a child for well over a minute.
                # The hold is released as soon as every rollout child is admitted.
                extra_args["cis_fork_lease_ms"] = 300_000
        if self._native_kv_fork and request.fork_parent_request_id is not None:
            extra_args["cis_fork_parent_request_id"] = request.fork_parent_request_id
            if request.fork_wait_for_parent:
                if not self._native_kv_fork_waiters:
                    raise RuntimeError(
                        "fork waiter requests require native_kv_fork_waiters=true"
                    )
                extra_args["cis_fork_wait_for_parent"] = True
                if self._native_kv_fork_compact_waiters:
                    extra_args["cis_fork_compact_waiter"] = True
                # The EngineCore expands this prompt with the parent's candidate
                # tokens. The frontend never sees that mutation, so keep its
                # cache metrics bounded by the prompt length it registered.
                extra_args["cis_fork_frontend_prompt_tokens"] = len(request.prefix)
        match = _CIS_STEP_REQUEST.match(request.request_id)
        if self._native_kv_resample_gc and match is not None:
            extra_args["cis_branch_handle"] = request.request_id
            extra_args["cis_branch_record_from_token"] = len(request.prefix)
            extra_args["cis_branch_kind"] = (
                "rollout" if match.group("rollout") is not None else "candidate"
            )
        if (
            self._native_kv_branch_eviction
            and match is not None
            and match.group("rollout") is not None
        ):
            extra_args["cis_disposable_from_token"] = len(request.prefix)
        params.extra_args = extra_args or None
        return params

    def resolve_cis_branches(
        self,
        *,
        selected_candidate_id: str,
        candidate_ids: Sequence[str],
        rollout_ids: Sequence[str],
    ) -> Mapping[str, int] | None:
        """Apply the algorithm's exact winner/loser transition to cached KV."""

        if not self._native_kv_resample_gc:
            return None

        async def resolve() -> Mapping[str, int]:
            engine_core = getattr(self._engine, "engine_core", None)
            callback = getattr(engine_core, "call_utility_async", None)
            if callback is None:
                raise RuntimeError("AsyncLLM does not expose EngineCore utility calls")
            result = await callback(
                "cis_resample_gc",
                selected_candidate_id,
                list(candidate_ids),
                list(rollout_ids),
            )
            if not isinstance(result, Mapping):
                raise RuntimeError("CIS resample GC returned an invalid result")
            return result

        return self._runner.run(resolve())

    def _next_request_id(self) -> str:
        with self._request_counter_lock:
            value = next(self._request_counter)
        return f"inference-scaling:{self.model_id}:{value}"

    @staticmethod
    def _native_parallel_params(params: Any, requests: Sequence[GenerationRequest]) -> Any:
        try:
            from vllm.sampling_params import RequestOutputKind

            params.output_kind = RequestOutputKind.FINAL_ONLY
        except ImportError:  # pragma: no cover - exercised by lightweight fakes
            params.output_kind = "final_only"
        params.n = len(requests)
        extra_args = dict(getattr(params, "extra_args", None) or {})
        extra_args.update(
            cis_child_seeds=[int(request.seed) for request in requests],
            cis_emit_child_finals=True,
        )
        params.extra_args = extra_args
        return params

    @staticmethod
    def _partition_native_parallel_requests(
        requests: Sequence[GenerationRequest],
    ) -> list[list[tuple[int, GenerationRequest]]]:
        groups: list[list[tuple[int, GenerationRequest]]] = []
        positions: dict[tuple[Any, ...], int] = {}
        for index, request in enumerate(requests):
            key = _cis_parallel_group_key(request)
            if key is None:
                groups.append([(index, request)])
                continue
            position = positions.get(key)
            if position is None:
                positions[key] = len(groups)
                groups.append([(index, request)])
            else:
                groups[position].append((index, request))
        return groups

    async def _generate_parent_group(
        self,
        indexed_requests: Sequence[tuple[int, GenerationRequest]],
        on_complete: SampleCompletionCallback | None,
    ) -> list[tuple[int, tuple[SequenceSample, int, int, int]]]:
        requests = tuple(request for _, request in indexed_requests)
        first = requests[0]
        prompt = self._prompt(first.prefix)
        params = self._native_parallel_params(self._sampling_params(first), requests)
        parent_request_id = f"{first.request_id}:parent:{len(requests)}"
        priority = self._request_priority(first)
        kwargs = {
            "prompt": prompt,
            "sampling_params": params,
            "request_id": parent_request_id,
            "priority": priority,
        }
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request

        submitted_at = time.time()
        for child_index, request in enumerate(requests):
            self._observe_request(
                {
                    "schema_version": 1,
                    "event": "submitted",
                    "event_unix_us": int(submitted_at * 1e6),
                    "request_id": request.request_id,
                    "engine_request_id": f"{child_index}_{parent_request_id}",
                    "parent_request_id": parent_request_id,
                    "native_parallel_group_size": len(requests),
                    "native_parallel_child_index": child_index,
                    "prefix_tokens": len(request.prefix),
                    "max_new_tokens": request.max_new_tokens,
                    "seed": request.seed,
                    "priority": priority,
                }
            )

        self._engine_requests_started(len(requests))
        with self._statistics_lock:
            self._native_parallel_groups += 1
            self._native_parallel_children += len(requests)
        completed: dict[int, tuple[SequenceSample, int, int, int]] = {}
        try:
            async for output in self._engine.generate(**kwargs):
                cached_tokens = int(getattr(output, "num_cached_tokens", 0) or 0)
                for completion in getattr(output, "outputs", ()):
                    child_index = int(getattr(completion, "index", 0))
                    if child_index in completed:
                        continue
                    if getattr(completion, "finish_reason", None) is None:
                        continue
                    if not 0 <= child_index < len(requests):
                        raise RuntimeError("vLLM returned an invalid parent child index")
                    request = requests[child_index]
                    view = _SingleCompletionOutput(
                        (completion,), cached_tokens, parent_request_id
                    )
                    item = self._sample_from_output(request, view)
                    completed[child_index] = item
                    self._observe_request(
                        {
                            "schema_version": 1,
                            "event": "finished",
                            "event_unix_us": int(time.time() * 1e6),
                            "request_id": request.request_id,
                            "engine_request_id": f"{child_index}_{parent_request_id}",
                            "parent_request_id": parent_request_id,
                            "native_parallel_group_size": len(requests),
                            "native_parallel_child_index": child_index,
                            "prefix_tokens": len(request.prefix),
                            "max_new_tokens": request.max_new_tokens,
                            "seed": request.seed,
                            "priority": priority,
                            "output_tokens": len(item[0].token_ids),
                            "cached_tokens": cached_tokens,
                            "finish_reason": item[0].finish_reason,
                        }
                    )
                    if on_complete is not None:
                        on_complete(indexed_requests[child_index][0], item[0])
        finally:
            self._engine_requests_finished(len(requests))
        if len(completed) != len(requests):
            raise RuntimeError("vLLM parent request omitted one or more child outputs")
        return [
            (indexed_requests[index][0], completed[index])
            for index in range(len(requests))
        ]

    async def _generate_native_parallel(
        self,
        requests: Sequence[GenerationRequest],
        on_complete: SampleCompletionCallback | None,
    ) -> list[tuple[SequenceSample, int, int, int]]:
        async def run_group(indexed: list[tuple[int, GenerationRequest]]):
            if len(indexed) > 1:
                return await self._generate_parent_group(indexed, on_complete)
            index, request = indexed[0]
            output = await self._generate_one(
                self._prompt(request.prefix), self._sampling_params(request), request
            )
            item = self._sample_from_output(request, output)
            if on_complete is not None:
                on_complete(index, item[0])
            return [(index, item)]

        grouped = await asyncio.gather(
            *(
                run_group(group)
                for group in self._partition_native_parallel_requests(requests)
            )
        )
        ordered: list[tuple[SequenceSample, int, int, int] | None] = [None] * len(
            requests
        )
        for group in grouped:
            for index, item in group:
                ordered[index] = item
        if any(item is None for item in ordered):
            raise RuntimeError("native parallel sampling omitted a request")
        return [item for item in ordered if item is not None]

    def start_profile(self, profile_prefix: str | None = None) -> None:
        async def start() -> None:
            callback = getattr(self._engine, "start_profile", None)
            if callback is None:
                raise RuntimeError(
                    "this AsyncLLM frontend does not expose start_profile"
                )
            result = callback(profile_prefix)
            if inspect.isawaitable(result):
                await result

        self._runner.run(start())

    def stop_profile(self) -> None:
        async def stop() -> None:
            callback = getattr(self._engine, "stop_profile", None)
            if callback is None:
                raise RuntimeError(
                    "this AsyncLLM frontend does not expose stop_profile"
                )
            result = callback()
            if inspect.isawaitable(result):
                await result

        self._runner.run(stop())

    def _engine_metric_totals(self) -> tuple[int, int, int, int]:
        if self._closed:
            return 0, 0, 0, 0

        async def read() -> tuple[int, int, int, int]:
            callback = getattr(self._engine, "get_metrics", None)
            try:
                if callback is None:
                    # AsyncLLM does not expose LLM.get_metrics(), but records
                    # the same counters in the process-wide Prometheus registry.
                    from vllm.v1.metrics.reader import get_metrics_snapshot

                    metrics = get_metrics_snapshot()
                else:
                    metrics = callback()
                    if inspect.isawaitable(metrics):
                        metrics = await metrics
            except (AssertionError, ImportError, RuntimeError):
                return 0, 0, 0, 0
            return self._sum_metric_values(metrics, model_name=self._metric_model_name)

        return self._runner.run(read())

    async def _generate_one(
        self,
        prompt: Any,
        params: Any,
        request: GenerationRequest | None = None,
    ) -> Any:
        submitted_at = time.time()
        epoch_offset = submitted_at - time.monotonic()
        engine_request_id = (
            request.request_id if request is not None else self._next_request_id()
        )
        priority = self._request_priority(request)
        kwargs = {
            "prompt": prompt,
            "sampling_params": params,
            "request_id": engine_request_id,
            "priority": priority,
        }
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        self._engine_requests_started(1)
        final = None
        first_output_at: float | None = None
        common = {
            "schema_version": 1,
            "request_id": None if request is None else request.request_id,
            "engine_request_id": engine_request_id,
            "prefix_tokens": None if request is None else len(request.prefix),
            "max_new_tokens": None if request is None else request.max_new_tokens,
            "seed": None if request is None else request.seed,
            "priority": priority,
        }
        self._observe_request(
            {
                **common,
                "event": "submitted",
                "event_unix_us": int(submitted_at * 1e6),
            }
        )
        try:
            async for output in self._engine.generate(**kwargs):
                if first_output_at is None:
                    first_output_at = time.time()
                    self._observe_request(
                        {
                            **common,
                            "event": "first_output",
                            "event_unix_us": int(first_output_at * 1e6),
                        }
                    )
                final = output
        except BaseException as error:
            self._observe_request(
                {
                    **common,
                    "event": "error",
                    "event_unix_us": int(time.time() * 1e6),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise
        finally:
            self._engine_requests_finished(1)
        if final is None:
            raise RuntimeError("asynchronous vLLM request returned no output")
        finished_at = time.time()
        metrics = getattr(final, "metrics", None)
        timing: dict[str, Any] = {}
        if metrics is not None:
            for name in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts"):
                value = float(getattr(metrics, name, 0.0) or 0.0)
                if value > 0:
                    timing[f"{name[:-3]}_unix_us"] = int((epoch_offset + value) * 1e6)
            queued = float(getattr(metrics, "queued_ts", 0.0) or 0.0)
            scheduled = float(getattr(metrics, "scheduled_ts", 0.0) or 0.0)
            first_token = float(getattr(metrics, "first_token_ts", 0.0) or 0.0)
            last_token = float(getattr(metrics, "last_token_ts", 0.0) or 0.0)
            if queued > 0 and scheduled >= queued:
                timing["queue_us"] = int((scheduled - queued) * 1e6)
            if scheduled > 0 and first_token >= scheduled:
                timing["scheduled_to_first_token_us"] = int(
                    (first_token - scheduled) * 1e6
                )
            if first_token > 0 and last_token >= first_token:
                timing["decode_us"] = int((last_token - first_token) * 1e6)
        completion = self._completion(final)
        self._observe_request(
            {
                **common,
                **timing,
                "event": "finished",
                "event_unix_us": int(finished_at * 1e6),
                "output_tokens": len(completion.token_ids),
                "cached_tokens": int(getattr(final, "num_cached_tokens", 0) or 0),
                "finish_reason": str(
                    getattr(completion, "finish_reason", "length") or "length"
                ),
                "frontend_first_output_unix_us": (
                    None if first_output_at is None else int(first_output_at * 1e6)
                ),
            }
        )
        return final

    async def _generate_many(self, prompts: Sequence[Any], params: Any) -> list[Any]:
        policies = params if isinstance(params, list) else [params] * len(prompts)
        if len(policies) != len(prompts):
            raise ValueError(
                "the number of vLLM sampling policies must match the prompts"
            )
        return list(
            await asyncio.gather(
                *(
                    self._generate_one(prompt, policy)
                    for prompt, policy in zip(prompts, policies, strict=True)
                )
            )
        )

    async def _generate_many_as_completed(
        self,
        prompts: Sequence[Any],
        params: Any,
        requests: Sequence[GenerationRequest],
        on_complete: SampleCompletionCallback,
    ) -> list[tuple[SequenceSample, int, int, int]]:
        policies = params if isinstance(params, list) else [params] * len(prompts)
        if len(policies) != len(prompts) or len(requests) != len(prompts):
            raise ValueError(
                "vLLM prompts, policies, and requests must have equal length"
            )

        async def indexed(index: int, prompt: Any, policy: Any):
            return index, await self._generate_one(prompt, policy, requests[index])

        tasks = [
            asyncio.create_task(indexed(index, prompt, policy))
            for index, (prompt, policy) in enumerate(
                zip(prompts, policies, strict=True)
            )
        ]
        parsed: list[tuple[SequenceSample, int, int, int] | None] = [None] * len(tasks)
        try:
            for completed in asyncio.as_completed(tasks):
                index, output = await completed
                item = self._sample_from_output(requests[index], output)
                parsed[index] = item
                self.observe_draft_samples((item[0],))
                on_complete(index, item[0])
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        if any(item is None for item in parsed):
            raise RuntimeError("asynchronous vLLM omitted a request result")
        return [item for item in parsed if item is not None]

    def sample_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[SequenceSample]:
        """Submit algorithm request IDs directly so service traces retain the CIS tree."""

        if not requests:
            return []
        if self._mh_fused_logprobs:
            return super().sample_batch(requests)
        if self._closed:
            raise RuntimeError("vLLM backend is closed")
        if self._native_parallel_sampling:
            parsed = self._runner.run(
                self._generate_native_parallel(tuple(requests), None)
            )
            samples = [item[0] for item in parsed]
            self._record_sample_batch(
                samples,
                prefill_tokens=sum(item[1] for item in parsed),
                cached_tokens=sum(item[2] for item in parsed),
                forward_slots=sum(item[3] for item in parsed),
            )
            self.observe_draft_samples(samples)
            return samples
        prompts = tuple(self._prompt(request.prefix) for request in requests)
        params = [self._sampling_params(request) for request in requests]
        parsed = self._runner.run(
            self._generate_many_as_completed(
                prompts,
                params,
                tuple(requests),
                lambda _index, _sample: None,
            )
        )
        samples = [item[0] for item in parsed]
        self._record_sample_batch(
            samples,
            prefill_tokens=sum(item[1] for item in parsed),
            cached_tokens=sum(item[2] for item in parsed),
            forward_slots=sum(item[3] for item in parsed),
        )
        return samples

    def _generate(self, prompts: Sequence[Any], params: Any) -> list[Any]:
        if self._closed:
            raise RuntimeError("vLLM backend is closed")
        return self._runner.run(self._generate_many(tuple(prompts), params))

    def sample_batch_with_callback(
        self,
        requests: Sequence[GenerationRequest],
        on_complete: SampleCompletionCallback,
    ) -> list[SequenceSample]:
        """Stream final request outputs from the persistent AsyncLLM engine."""

        if not requests:
            return []
        if self._closed:
            raise RuntimeError("vLLM backend is closed")
        if self._native_parallel_sampling:
            parsed = self._runner.run(
                self._generate_native_parallel(tuple(requests), on_complete)
            )
            samples = [item[0] for item in parsed]
            self._record_sample_batch(
                samples,
                prefill_tokens=sum(item[1] for item in parsed),
                cached_tokens=sum(item[2] for item in parsed),
                forward_slots=sum(item[3] for item in parsed),
            )
            self.observe_draft_samples(samples)
            return samples
        prompts = tuple(self._prompt(request.prefix) for request in requests)
        params = [self._sampling_params(request) for request in requests]
        parsed = self._runner.run(
            self._generate_many_as_completed(
                prompts,
                params,
                tuple(requests),
                on_complete,
            )
        )
        samples = [item[0] for item in parsed]
        self._record_sample_batch(
            samples,
            prefill_tokens=sum(item[1] for item in parsed),
            cached_tokens=sum(item[2] for item in parsed),
            forward_slots=sum(item[3] for item in parsed),
        )
        return samples

    def _beam_score(self, tokens: Sequence[int], cumulative_logprob: float) -> float:
        length = len(tokens)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None and tokens and int(tokens[-1]) == int(eos):
            length -= 1
        return cumulative_logprob / max(1, length)

    async def _beam_search_async(
        self,
        prefix: TokenSequence,
        *,
        max_new_tokens: int,
        num_beams: int,
    ) -> TokenSequence:
        params = self._sampling_params_factory(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            seed=0,
            logprobs=2 * int(num_beams),
            flat_logprobs=False,
            ignore_eos=True,
            detokenize=False,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )
        model_prefix = self._model_prefix(prefix)
        active: list[tuple[TokenSequence, float]] = [(model_prefix, 0.0)]
        completed: list[tuple[TokenSequence, float]] = []
        eos = getattr(self.tokenizer, "eos_token_id", None)
        for _ in range(max_new_tokens):
            outputs = await asyncio.gather(
                *(
                    self._generate_one(self._prompt(tokens), params)
                    for tokens, _ in active
                )
            )
            candidates: list[tuple[TokenSequence, float]] = []
            for (tokens, cumulative), output in zip(active, outputs, strict=True):
                completion = self._completion(output)
                positions = completion.logprobs
                if positions is None or len(positions) != 1:
                    raise RuntimeError(
                        "vLLM beam expansion omitted next-token log-probabilities"
                    )
                for token, value in positions[0].items():
                    token_id = int(token)
                    expanded = tokens + (token_id,)
                    scored = cumulative + float(getattr(value, "logprob", value))
                    if eos is not None and token_id == int(eos):
                        completed.append((expanded, scored))
                    else:
                        candidates.append((expanded, scored))
            active = sorted(
                candidates,
                key=lambda item: self._beam_score(*item),
                reverse=True,
            )[:num_beams]
            if not active:
                break
        finalists = completed + active
        if not finalists:
            raise RuntimeError("vLLM beam search produced no sequence")
        full_tokens = max(finalists, key=lambda item: self._beam_score(*item))[0]
        if full_tokens[: len(model_prefix)] != model_prefix:
            raise RuntimeError("vLLM beam-search output did not preserve the prompt")
        return full_tokens[len(model_prefix) :]

    def direct_generate(
        self,
        prefix: TokenSequence,
        *,
        max_new_tokens: int,
        num_beams: int = 1,
    ) -> TokenSequence:
        if num_beams == 1:
            return super().direct_generate(
                prefix,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        if max_new_tokens <= 0 or num_beams <= 0:
            raise ValueError("generation length and beam count must be positive")
        return self._runner.run(
            self._beam_search_async(
                prefix,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._runner.close()
