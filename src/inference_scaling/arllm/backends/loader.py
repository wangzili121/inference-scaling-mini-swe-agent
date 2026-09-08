"""Configuration-driven construction and cleanup of experiment backends."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inference_scaling.arllm.acceleration import (
    ActiveBatchSpeculationConfig,
    SpeculationTier,
)
from inference_scaling.arllm.backends.transformers_backend import TransformersBackend
from inference_scaling.arllm.backends.vllm_backend import AsyncVLLMBackend, VLLMBackend

BACKEND_CHOICES = ("transformers", "vllm", "vllm-sync")

_VLLM_SETTINGS = {
    "asynchronous",
    "data_parallel_size",
    "download_dir",
    "dtype",
    "enable_prefix_caching",
    "enforce_eager",
    "engine_kwargs",
    "exact_scoring_backend",
    "exact_scoring_device",
    "exact_scoring_dtype",
    "gpu_memory_utilization",
    "max_lora_rank",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "mh_fused_logprobs",
    "parameter_count",
    "quantization",
    "revision",
    "seed",
    "tensor_parallel_size",
    "trust_remote_code",
}
_MODEL_ROLES = ("base", "proposal", "rl")
_EXPLICIT_ENGINE_SETTINGS = {
    "data_parallel_size",
    "download_dir",
    "dtype",
    "enable_lora",
    "enable_prefix_caching",
    "enforce_eager",
    "generation_config",
    "gpu_memory_utilization",
    "logprobs_mode",
    "max_lora_rank",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "model",
    "quantization",
    "revision",
    "seed",
    "tensor_parallel_size",
    "trust_remote_code",
    "worker_cls",
    "async_scheduling",
}


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a table")
    return dict(value)


def _speculation_from_config(
    config: Mapping[str, Any],
) -> tuple[ActiveBatchSpeculationConfig | None, bool]:
    acceleration = _mapping(config.get("acceleration"), name="acceleration")
    table = _mapping(acceleration.get("speculation"), name="acceleration.speculation")
    if not table or not bool(table.pop("enabled", False)):
        return None, False
    dynamic_vllm = bool(table.pop("dynamic_vllm", False))
    raw_tiers = table.pop("tiers", None)
    tiers = None
    if raw_tiers is not None:
        if not isinstance(raw_tiers, (list, tuple)):
            raise TypeError("acceleration.speculation.tiers must be an array")
        try:
            tiers = tuple(
                SpeculationTier(int(item[0]), int(item[1])) for item in raw_tiers
            )
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError(
                "each speculation tier must be [maximum_active_batch, draft_tokens]"
            ) from error
    aliases = {
        "min_context_tokens": "min_context_tokens",
        "min_token_probability": "min_token_probability",
        "tree_max_context_tokens": "tree_max_context_tokens",
        "tree_max_contexts": "tree_max_contexts",
        "vllm_max_cached_requests": "vllm_max_cached_requests",
        "stochastic_tree": "stochastic_tree",
    }
    unknown = sorted(set(table) - set(aliases))
    if unknown:
        raise ValueError("unknown speculation settings: " + ", ".join(unknown))
    kwargs = {aliases[name]: value for name, value in table.items()}
    if tiers is not None:
        kwargs["tiers"] = tiers
    return ActiveBatchSpeculationConfig(**kwargs), dynamic_vllm


def _infer_role(
    path: str,
    config: Mapping[str, Any],
    *,
    adapter_base: str | None,
) -> str:
    models = _mapping(config.get("models"), name="models")
    resolved = Path(path).resolve()
    for role in _MODEL_ROLES:
        configured = models.get(role)
        if configured is not None and Path(str(configured)).resolve() == resolved:
            return role
    return "rl" if adapter_base is not None else "base"


def _vllm_settings(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    table = _mapping(config.get("vllm"), name="vllm")
    settings = {
        key: value
        for key, value in table.items()
        if key not in _MODEL_ROLES
    }
    role_settings = _mapping(table.get(role), name=f"vllm.{role}")
    common_engine = _mapping(settings.pop("engine_kwargs", None), name="vllm.engine_kwargs")
    role_engine = _mapping(
        role_settings.pop("engine_kwargs", None),
        name=f"vllm.{role}.engine_kwargs",
    )
    settings.update(role_settings)
    settings["engine_kwargs"] = {**common_engine, **role_engine}
    unknown = sorted(set(settings) - _VLLM_SETTINGS)
    if unknown:
        raise ValueError("unknown vLLM settings: " + ", ".join(unknown))
    return settings


def configured_backend(config: Mapping[str, Any]) -> str:
    runtime = _mapping(config.get("runtime"), name="runtime")
    backend = str(runtime.get("backend", "transformers"))
    if backend not in BACKEND_CHOICES:
        raise ValueError(
            f"unknown runtime backend {backend!r}; expected one of {BACKEND_CHOICES}"
        )
    return backend


def set_backend_override(config: dict[str, Any], backend: str | None) -> None:
    """Apply a CLI backend override before fingerprinting an experiment."""

    if backend is None:
        return
    if backend not in BACKEND_CHOICES:
        raise ValueError(f"unknown runtime backend {backend!r}")
    config.setdefault("runtime", {})["backend"] = backend


def _transformers_backend(
    model_name_or_path: str,
    adapter_name_or_path: str | None,
    runtime: Mapping[str, Any],
    *,
    device: str | None = None,
    dtype: str | None = None,
    speculation: ActiveBatchSpeculationConfig | None = None,
) -> TransformersBackend:
    kwargs: dict[str, Any] = {}
    if speculation is not None:
        kwargs["speculation"] = speculation
    return TransformersBackend.from_pretrained(
        model_name_or_path,
        adapter_name_or_path=adapter_name_or_path,
        device=device or str(runtime.get("device", "cuda")),
        dtype=dtype or str(runtime.get("dtype", "float32")),
        local_files_only=True,
        trust_remote_code=bool(runtime.get("trust_remote_code", False)),
        max_score_batch_size=int(runtime.get("max_score_batch_size", 8)),
        **kwargs,
    )


def load_backend_from_config(
    path: str,
    config: Mapping[str, Any],
    *,
    adapter_base: str | None = None,
) -> Any:
    """Load one model using the selected runtime without changing its policy.

    ``vllm`` selects the persistent asynchronous engine, while ``vllm-sync``
    selects the offline ``LLM`` frontend.  Per-role tables such as
    ``[vllm.proposal]`` override common vLLM settings, which is useful when a
    base and proposal engine share one GPU.
    """

    runtime = _mapping(config.get("runtime"), name="runtime")
    speculation, dynamic_vllm_speculation = _speculation_from_config(config)
    backend_kind = configured_backend(config)
    model_name_or_path = adapter_base or path
    adapter_name_or_path = path if adapter_base is not None else None
    if backend_kind == "transformers":
        return _transformers_backend(
            model_name_or_path,
            adapter_name_or_path,
            runtime,
            speculation=speculation,
        )

    role = _infer_role(path, config, adapter_base=adapter_base)
    settings = _vllm_settings(config, role)
    exact_kind = str(settings.pop("exact_scoring_backend", "none"))
    exact_backend = None
    if exact_kind == "transformers":
        exact_backend = _transformers_backend(
            model_name_or_path,
            adapter_name_or_path,
            runtime,
            device=str(settings.pop("exact_scoring_device", runtime.get("device", "cuda"))),
            dtype=str(settings.pop("exact_scoring_dtype", runtime.get("dtype", "float32"))),
        )
    elif exact_kind != "none":
        raise ValueError("vllm.exact_scoring_backend must be 'none' or 'transformers'")
    elif "exact_scoring_device" in settings or "exact_scoring_dtype" in settings:
        raise ValueError(
            "exact_scoring_device/dtype require exact_scoring_backend='transformers'"
        )

    asynchronous = bool(settings.pop("asynchronous", True))
    if backend_kind == "vllm-sync":
        asynchronous = False
    mh_fused_logprobs = bool(settings.pop("mh_fused_logprobs", False))
    if mh_fused_logprobs and asynchronous:
        raise ValueError(
            "vllm.mh_fused_logprobs requires runtime.backend='vllm-sync'"
        )
    if mh_fused_logprobs and speculation is not None:
        raise ValueError(
            "vllm.mh_fused_logprobs cannot be combined with speculative decoding"
        )
    engine_kwargs = _mapping(
        settings.pop("engine_kwargs", None),
        name="vllm.engine_kwargs",
    )
    collisions = sorted(_EXPLICIT_ENGINE_SETTINGS.intersection(engine_kwargs))
    if collisions:
        raise ValueError(
            "vLLM engine_kwargs duplicate explicit settings: "
            + ", ".join(collisions)
        )
    beam_width = int(_mapping(config.get("beam"), name="beam").get("num_beams", 1))
    required_logprobs = max(20, 2 * beam_width)
    engine_kwargs["max_logprobs"] = max(
        required_logprobs,
        int(engine_kwargs.get("max_logprobs", required_logprobs)),
    )
    loader = AsyncVLLMBackend if asynchronous else VLLMBackend
    acceleration_kwargs: dict[str, Any] = {}
    if speculation is not None:
        acceleration_kwargs = {
            "speculation": speculation,
            "dynamic_speculation": dynamic_vllm_speculation,
        }
    try:
        return loader.from_pretrained(
            model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dtype=str(settings.pop("dtype", runtime.get("dtype", "bfloat16"))),
            tensor_parallel_size=int(settings.pop("tensor_parallel_size", 1)),
            data_parallel_size=int(settings.pop("data_parallel_size", 1)),
            gpu_memory_utilization=float(settings.pop("gpu_memory_utilization", 0.9)),
            max_model_len=settings.pop("max_model_len", None),
            max_num_seqs=settings.pop("max_num_seqs", None),
            max_num_batched_tokens=settings.pop("max_num_batched_tokens", None),
            quantization=settings.pop("quantization", None),
            enforce_eager=bool(settings.pop("enforce_eager", False)),
            trust_remote_code=bool(
                settings.pop("trust_remote_code", runtime.get("trust_remote_code", False))
            ),
            revision=settings.pop("revision", None),
            download_dir=settings.pop("download_dir", None),
            seed=int(settings.pop("seed", config.get("run", {}).get("seed", 0))),
            parameter_count=settings.pop("parameter_count", None),
            scoring_backend=exact_backend,
            enable_prefix_caching=bool(settings.pop("enable_prefix_caching", True)),
            max_lora_rank=int(settings.pop("max_lora_rank", 16)),
            enable_mh_fused_logprobs=mh_fused_logprobs,
            engine_kwargs=engine_kwargs,
            **acceleration_kwargs,
        )
    except BaseException:
        close_backend(exact_backend)
        raise


def close_backend(backend: Any | None) -> None:
    """Close vLLM processes/threads when present; safe for other backends."""

    if backend is None:
        return
    callback = getattr(backend, "close", None)
    if callback is not None:
        callback()
    scoring_backend = getattr(backend, "scoring_backend", None)
    if scoring_backend is not None and scoring_backend is not backend:
        nested = getattr(scoring_backend, "close", None)
        if nested is not None:
            nested()
