"""Persistent ordinary Conditional IS runner for agent model calls."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import tomllib
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from inference_scaling.arllm.algorithms import run_conditional_is
from inference_scaling.arllm.backends import close_backend, load_backend_from_config
from inference_scaling.arllm.config import ConditionalISConfig, SamplingConfig
from inference_scaling.arllm.rewards import (
    ConsilienceReward,
    SequenceLogProbabilityReward,
)
from inference_scaling.shared.metrics import importance_effective_sample_size
from inference_scaling.shared.rng import SeedStream
from inference_scaling.swe_agent.tool_calls import parse_assistant_text


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

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


def load_service_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("rb") as stream:
        config = _expand_environment(tomllib.load(stream))
    if not isinstance(config.get("models", {}).get("base"), str):
        raise ValueError("service config requires models.base")
    for table in ("generation", "sampling", "conditional_is", "reward"):
        if not isinstance(config.get(table), dict):
            raise ValueError(f"service config requires [{table}]")
    return config


def _public_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    return [
        {key: value for key, value in message.items() if key in allowed}
        for message in messages
    ]


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
        trace_path = service.get("trace_path")
        self.trace_writer = JsonlTraceWriter(trace_path) if trace_path else None

    @classmethod
    def from_toml(cls, path: str | Path) -> "ConditionalISRunner":
        config = load_service_config(path)
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
    ) -> QueryResult:
        started = time.perf_counter()
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
        result = run_conditional_is(
            self.backend,
            prompt,
            self.conditional,
            self.reward,
            SeedStream(seed),
            base_sampling=self.sampling,
            rollout_backend=self.backend,
            rollout_sampling=self.sampling,
        )
        algorithm_seconds = time.perf_counter() - algorithm_started
        after = _snapshot(self.backend)
        text = self.backend.decode(result.token_ids, skip_special_tokens=False)
        parsed = parse_assistant_text(text, request_id=request_id)
        candidate_ess = [
            importance_effective_sample_size(
                [candidate.log_weight for candidate in step.candidates]
            )
            for step in result.steps
        ]
        diagnostics = {
            "request_id": request_id,
            "seed": seed,
            "prompt_tokens": len(prompt),
            "completion_tokens": len(result.token_ids),
            "algorithm_seconds": algorithm_seconds,
            "total_seconds": time.perf_counter() - started,
            "steps": len(result.steps),
            "candidate_count": self.conditional.candidate_count,
            "rollout_count": self.conditional.rollout_count,
            "block_size": self.conditional.block_size,
            "candidate_ess": candidate_ess,
            "backend_delta": _counter_delta(before, after),
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
                    "prompt_token_ids": prompt,
                    "message": message,
                    "diagnostics": diagnostics,
                }
            )
        return QueryResult(message=message, diagnostics=diagnostics)

    def close(self) -> None:
        close_backend(self.backend)


__all__ = [
    "BASH_TOOL",
    "ConditionalISRunner",
    "JsonlTraceWriter",
    "QueryResult",
    "load_service_config",
]
