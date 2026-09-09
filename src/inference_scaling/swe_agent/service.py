"""Persistent ordinary Conditional IS runner for agent model calls."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import tomllib
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, is_dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from inference_scaling.arllm.algorithms import run_conditional_is
from inference_scaling.arllm.backends import close_backend, load_backend_from_config
from inference_scaling.arllm.config import ConditionalISConfig, SamplingConfig
from inference_scaling.arllm.rewards import (
    ConsilienceReward,
    SequenceLogProbabilityReward,
)
from inference_scaling.shared.metrics import importance_effective_sample_size
from inference_scaling.shared.rng import SeedStream
from inference_scaling.swe_agent.messages import BASH_TOOL, public_messages
from inference_scaling.swe_agent.tool_calls import (
    ParsedAssistant,
    ToolCallParseError,
    parse_assistant_text,
)


_ENVIRONMENT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"required environment variable {name!r} is unset")
            return os.environ[name]

        return _ENVIRONMENT.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _apply_config_overrides(
    config: dict[str, Any], overrides: Sequence[str]
) -> dict[str, Any]:
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"configuration override requires path=value: {override!r}"
            )
        path, raw_value = override.split("=", 1)
        keys = path.split(".")
        if not keys or any(not key for key in keys):
            raise ValueError(f"invalid configuration path: {path!r}")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"configuration override values must be JSON: {override!r}"
            ) from error
        table: dict[str, Any] = config
        for key in keys[:-1]:
            nested = table.get(key)
            if not isinstance(nested, dict):
                raise ValueError(
                    f"configuration override path is not a table: {path!r}"
                )
            table = nested
        if keys[-1] not in table:
            raise ValueError(f"configuration override path does not exist: {path!r}")
        table[keys[-1]] = value
    return config


def load_service_config(
    path: str | Path, *, overrides: Sequence[str] = ()
) -> dict[str, Any]:
    source = Path(path)
    with source.open("rb") as stream:
        config = _expand_environment(tomllib.load(stream))
    _apply_config_overrides(config, overrides)
    if not isinstance(config.get("models", {}).get("base"), str):
        raise ValueError("service config requires models.base")
    for table in ("generation", "sampling", "conditional_is", "reward"):
        if not isinstance(config.get(table), dict):
            raise ValueError(f"service config requires [{table}]")
    return config


_public_messages = public_messages


def _snapshot(backend: Any) -> dict[str, Any]:
    callback = getattr(backend, "snapshot", None)
    if callback is None:
        return {}
    value = callback()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _counter_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, value in after.items():
        previous = before.get(key)
        if isinstance(value, (int, float)) and isinstance(previous, (int, float)):
            delta[key] = value - previous
    return delta


@dataclass(frozen=True, slots=True)
class QueryResult:
    message: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CISExecution:
    prompt: tuple[int, ...]
    result: Any
    algorithm_seconds: float
    total_seconds: float
    backend_delta: dict[str, Any]
    stage_events: tuple[dict[str, Any], ...]
    conditional: dict[str, int]
    started_at: float
    finished_at: float


@dataclass(slots=True)
class _PendingQuery:
    fingerprint: str
    ready: threading.Event
    result: QueryResult | None = None
    error: BaseException | None = None


class IdempotentQueryCache:
    """Coalesce retry-equivalent jobs and retain a bounded completed-result cache."""

    def __init__(self, maximum_entries: int = 256) -> None:
        if maximum_entries < 0:
            raise ValueError("idempotency cache size must be non-negative")
        self.maximum_entries = maximum_entries
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingQuery] = {}
        self._completed: OrderedDict[str, tuple[str, QueryResult]] = OrderedDict()

    def execute(
        self,
        request_id: str,
        fingerprint: str,
        callback: Callable[[], QueryResult],
    ) -> QueryResult:
        with self._lock:
            cached = self._completed.get(request_id)
            if cached is not None:
                if cached[0] != fingerprint:
                    raise ValueError(
                        "request_id was already used with a different payload"
                    )
                self._completed.move_to_end(request_id)
                return cached[1]
            pending = self._pending.get(request_id)
            if pending is not None:
                if pending.fingerprint != fingerprint:
                    raise ValueError("request_id is in flight with a different payload")
                owner = False
            else:
                pending = _PendingQuery(fingerprint, threading.Event())
                self._pending[request_id] = pending
                owner = True

        if not owner:
            pending.ready.wait()
            if pending.error is not None:
                raise pending.error
            if pending.result is None:
                raise RuntimeError(
                    "coalesced Conditional IS request produced no result"
                )
            return pending.result

        try:
            result = callback()
        except BaseException as error:
            pending.error = error
            raise
        else:
            pending.result = result
            with self._lock:
                if self.maximum_entries:
                    self._completed[request_id] = (fingerprint, result)
                    self._completed.move_to_end(request_id)
                    while len(self._completed) > self.maximum_entries:
                        self._completed.popitem(last=False)
            return result
        finally:
            with self._lock:
                self._pending.pop(request_id, None)
                pending.ready.set()


class JsonlTraceWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(payload + "\n")


class ConditionalISRunner:
    """One persistent backend; each query is one complete ordinary CIS job."""

    def __init__(self, backend: Any, config: Mapping[str, Any]) -> None:
        self.backend = backend
        self.config = dict(config)
        generation = dict(config["generation"])
        conditional = dict(config["conditional_is"])
        sampling = dict(config["sampling"])
        reward = dict(config["reward"])
        self.maximum = int(generation["max_new_tokens"])
        self.sampling = SamplingConfig(
            temperature=float(sampling.get("temperature", 1.0)),
            top_p=float(sampling.get("top_p", 1.0)),
            top_k=(None if sampling.get("top_k") is None else int(sampling["top_k"])),
            eos_token_id=getattr(backend.tokenizer, "eos_token_id", None),
        )
        self.conditional = ConditionalISConfig(
            candidate_count=int(conditional["candidate_count"]),
            rollout_count=int(conditional["rollout_count"]),
            block_size=int(conditional["block_size"]),
            total_length=self.maximum,
            reward_temperature=float(conditional.get("reward_temperature", 1.0)),
            rollout_submission_batch_size=(
                None
                if conditional.get("rollout_submission_batch_size") is None
                else int(conditional["rollout_submission_batch_size"])
            ),
        )
        reward_kind = str(reward.get("kind", "sequence_log_probability"))
        if reward_kind == "sequence_log_probability":
            self.reward = SequenceLogProbabilityReward(
                backend,
                self.sampling,
                scale=float(reward.get("scale", 1.0)),
            )
        elif reward_kind == "consilience":
            self.reward = ConsilienceReward(
                backend,
                self.sampling,
                top_k=int(reward.get("top_k", 5)),
                window_fraction=float(reward.get("window_fraction", 0.2)),
                skip_fraction=float(reward.get("skip_fraction", 0.05)),
                initial_penalty=float(reward.get("initial_penalty", 3.0)),
                scale=float(reward.get("scale", 1.0)),
            )
        else:
            raise ValueError(f"unsupported agent reward {reward_kind!r}")
        service = dict(config.get("service", {}))
        self.instance_id = str(
            service.get("instance_id", os.environ.get("CIS_INSTANCE_ID", "instance-0"))
        )
        trace_path = service.get("trace_path")
        self.trace_writer = JsonlTraceWriter(trace_path) if trace_path else None
        self.query_cache = IdempotentQueryCache(
            int(service.get("idempotency_cache_size", 256))
        )

    @classmethod
    def from_toml(
        cls, path: str | Path, *, overrides: Sequence[str] = ()
    ) -> "ConditionalISRunner":
        config = load_service_config(path, overrides=overrides)
        backend = load_backend_from_config(str(config["models"]["base"]), config)
        return cls(backend, config)

    def _prompt_tokens(self, messages: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
        public = _public_messages(messages)
        rendered = self.backend.tokenizer.apply_chat_template(
            public,
            tools=[BASH_TOOL],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tuple(self.backend.encode(str(rendered), add_special_tokens=False))

    def query(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_id: str,
        seed: int,
        conditional_overrides: Mapping[str, int] | None = None,
    ) -> QueryResult:
        fingerprint = sha256(
            json.dumps(
                {
                    "messages": list(messages),
                    "seed": seed,
                    "conditional_is": dict(conditional_overrides or {}),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return self.query_cache.execute(
            request_id,
            fingerprint,
            lambda: self._query_uncached(
                messages,
                request_id=request_id,
                seed=seed,
                conditional_overrides=conditional_overrides,
            ),
        )

    def _query_uncached(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        request_id: str,
        seed: int,
        conditional_overrides: Mapping[str, int] | None = None,
    ) -> QueryResult:
        execution = self.execute(
            messages,
            seed=seed,
            request_namespace=request_id,
            conditional_overrides=conditional_overrides,
        )
        result = execution.result
        text = self.backend.decode(result.token_ids, skip_special_tokens=False)
        parse_error = None
        try:
            parsed = parse_assistant_text(text, request_id=request_id)
        except ToolCallParseError as error:
            parse_error = str(error)
            parsed = ParsedAssistant(text or None, (), ())
        candidate_ess = [
            importance_effective_sample_size(
                [candidate.log_weight for candidate in step.candidates]
            )
            for step in result.steps
        ]
        stage_seconds: dict[str, float] = defaultdict(float)
        for event in execution.stage_events:
            stage_seconds[str(event["name"])] += float(event["duration_us"]) / 1e6
        diagnostics = {
            "request_id": request_id,
            "seed": seed,
            "prompt_tokens": len(execution.prompt),
            "completion_tokens": len(result.token_ids),
            "algorithm_seconds": execution.algorithm_seconds,
            "total_seconds": execution.total_seconds,
            "started_at": execution.started_at,
            "finished_at": execution.finished_at,
            "steps": len(result.steps),
            **execution.conditional,
            "candidate_ess": candidate_ess,
            "instance_id": self.instance_id,
            "stage_seconds": dict(stage_seconds),
            "backend_delta": execution.backend_delta,
            "backend_delta_scope": "process_window_not_concurrency_safe",
            "tool_call_parse_error": parse_error,
            "finish_reason": (
                "eos" if self.sampling.eos_token_id in result.token_ids else "length"
            ),
        }
        message = {
            "role": "assistant",
            "content": parsed.content,
            "tool_calls": list(parsed.tool_calls),
            "extra": {
                "actions": list(parsed.actions),
                "cost": 0.0,
                "timestamp": time.time(),
                "conditional_is": diagnostics,
                "raw_completion": text,
            },
        }
        if self.trace_writer is not None:
            self.trace_writer.append(
                {
                    "schema_version": 1,
                    "request_id": request_id,
                    "messages": list(messages),
                    "prompt_token_ids": execution.prompt,
                    "message": message,
                    "diagnostics": diagnostics,
                    "stage_events": list(execution.stage_events),
                }
            )
        return QueryResult(message=message, diagnostics=diagnostics)

    def backend_snapshot(self) -> dict[str, Any]:
        return _snapshot(self.backend)

    def start_profile(self, profile_prefix: str | None = None) -> None:
        callback = getattr(self.backend, "start_profile", None)
        if callback is None:
            raise RuntimeError("configured backend does not support profiling")
        callback(profile_prefix)

    def stop_profile(self) -> None:
        callback = getattr(self.backend, "stop_profile", None)
        if callback is None:
            raise RuntimeError("configured backend does not support profiling")
        callback()

    def execute(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        request_namespace: str = "conditional-is",
        conditional_overrides: Mapping[str, int] | None = None,
    ) -> CISExecution:
        """Run one complete CIS job without parsing its selected assistant text."""

        started = time.perf_counter()
        started_at = time.time()
        started_ns = time.perf_counter_ns()
        stage_events: list[dict[str, Any]] = []

        def observe_stage(
            name: str,
            step_index: int,
            seconds: float,
            metadata: Mapping[str, Any],
        ) -> None:
            ended_ns = time.perf_counter_ns()
            duration_ns = max(0, int(seconds * 1e9))
            stage_events.append(
                {
                    "name": name,
                    "step": step_index,
                    "job_id": request_namespace,
                    "block_id": step_index,
                    "instance_id": self.instance_id,
                    "rank": int(
                        os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
                    ),
                    "start_unix_us": int(
                        (started_at + max(0, ended_ns - duration_ns - started_ns) / 1e9)
                        * 1e6
                    ),
                    "start_us": max(0, ended_ns - duration_ns - started_ns) / 1000,
                    "duration_us": duration_ns / 1000,
                    **dict(metadata),
                }
            )

        prompt = self._prompt_tokens(messages)
        configured_max = self.config.get("vllm", {}).get("max_model_len")
        if configured_max is not None and len(prompt) + self.maximum > int(
            configured_max
        ):
            raise ValueError(
                f"prompt ({len(prompt)}) plus generation ({self.maximum}) exceeds "
                f"max_model_len={configured_max}"
            )
        before = _snapshot(self.backend)
        algorithm_started = time.perf_counter()
        conditional = self.conditional
        if conditional_overrides:
            allowed = {"candidate_count", "rollout_count", "block_size"}
            unknown = sorted(set(conditional_overrides) - allowed)
            if unknown:
                raise ValueError(
                    "unsupported Conditional IS query overrides: " + ", ".join(unknown)
                )
            conditional = replace(
                conditional,
                **{key: int(value) for key, value in conditional_overrides.items()},
            )
        result = run_conditional_is(
            self.backend,
            prompt,
            conditional,
            self.reward,
            SeedStream(seed),
            base_sampling=self.sampling,
            rollout_backend=self.backend,
            rollout_sampling=self.sampling,
            request_namespace=request_namespace,
            stage_observer=observe_stage,
        )
        algorithm_seconds = time.perf_counter() - algorithm_started
        after = _snapshot(self.backend)
        finished_at = time.time()
        return CISExecution(
            prompt=prompt,
            result=result,
            algorithm_seconds=algorithm_seconds,
            total_seconds=time.perf_counter() - started,
            backend_delta=_counter_delta(before, after),
            stage_events=tuple(stage_events),
            conditional={
                "candidate_count": conditional.candidate_count,
                "rollout_count": conditional.rollout_count,
                "block_size": conditional.block_size,
            },
            started_at=started_at,
            finished_at=finished_at,
        )

    def close(self) -> None:
        close_backend(self.backend)


__all__ = [
    "BASH_TOOL",
    "CISExecution",
    "ConditionalISRunner",
    "JsonlTraceWriter",
    "QueryResult",
    "_apply_config_overrides",
    "load_service_config",
]
