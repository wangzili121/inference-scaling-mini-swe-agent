from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.swe_agent.model import ConditionalISModel
from inference_scaling.swe_agent.service import (
    ConditionalISRunner,
    load_service_config,
)
from inference_scaling.swe_agent.tool_calls import (
    ToolCallParseError,
    parse_assistant_text,
)


def test_qwen_tool_call_is_converted_to_one_bash_action() -> None:
    parsed = parse_assistant_text(
        'THOUGHT: inspect first\n<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call><|im_end|>',
        request_id="request-1",
    )

    assert parsed.content == "THOUGHT: inspect first"
    assert parsed.actions[0]["command"] == "pwd"
    assert json.loads(parsed.tool_calls[0]["function"]["arguments"]) == {
        "command": "pwd"
    }


def test_qwen_tool_call_rejects_non_bash_tool() -> None:
    with pytest.raises(ToolCallParseError, match="expected 'bash'"):
        parse_assistant_text(
            '<tool_call>{"name":"python","arguments":{"command":"pwd"}}</tool_call>',
            request_id="request-2",
        )


class _Tokenizer:
    eos_token_id = 2

    def __init__(self) -> None:
        self.messages = None
        self.tools = None

    def apply_chat_template(self, messages, *, tools, tokenize, add_generation_prompt):
        self.messages = messages
        self.tools = tools
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered prompt"


class _AgentBackend(TabularAutoregressiveBackend):
    def __init__(self) -> None:
        super().__init__({}, fallback=(0.6, 0.4))
        self.tokenizer = _Tokenizer()

    def encode(self, text, *, add_special_tokens=True):
        assert text == "rendered prompt"
        assert add_special_tokens is False
        return (7, 8, 9)

    def decode(self, tokens, *, skip_special_tokens=True):
        assert skip_special_tokens is False
        return '<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>'


def _runner_config(trace_path: Path) -> dict:
    return {
        "models": {"base": "unused"},
        "generation": {"max_new_tokens": 2},
        "sampling": {"temperature": 1.0, "top_p": 1.0},
        "conditional_is": {
            "candidate_count": 2,
            "rollout_count": 2,
            "block_size": 1,
            "reward_temperature": 1.0,
        },
        "reward": {"kind": "sequence_log_probability", "scale": 1.0},
        "vllm": {"max_model_len": 16},
        "service": {"trace_path": str(trace_path)},
    }


def test_runner_executes_one_complete_cis_job_and_records_trace(tmp_path: Path) -> None:
    trace = tmp_path / "calls.jsonl"
    backend = _AgentBackend()
    runner = ConditionalISRunner(backend, _runner_config(trace))

    result = runner.query(
        [{"role": "user", "content": "fix it", "extra": {"private": True}}],
        request_id="agent-call-1",
        seed=17,
    )

    assert result.message["extra"]["actions"][0]["command"] == "ls"
    assert result.diagnostics["prompt_tokens"] == 3
    assert result.diagnostics["completion_tokens"] == 2
    assert backend.tokenizer.messages == [{"role": "user", "content": "fix it"}]
    record = json.loads(trace.read_text())
    assert record["request_id"] == "agent-call-1"
    assert record["messages"][0]["extra"] == {"private": True}


def test_runner_coalesces_retry_equivalent_request_ids(tmp_path: Path) -> None:
    trace = tmp_path / "calls.jsonl"
    backend = _AgentBackend()
    runner = ConditionalISRunner(backend, _runner_config(trace))
    messages = [{"role": "user", "content": "fix it"}]

    first = runner.query(messages, request_id="retry-1", seed=17)
    second = runner.query(messages, request_id="retry-1", seed=17)

    assert first == second
    assert len(trace.read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="different payload"):
        runner.query(messages, request_id="retry-1", seed=18)


def test_runner_forwards_profile_control_to_backend(tmp_path: Path) -> None:
    backend = _AgentBackend()
    events = []
    backend.start_profile = lambda prefix=None: events.append(("start", prefix))
    backend.stop_profile = lambda: events.append(("stop", None))
    runner = ConditionalISRunner(backend, _runner_config(tmp_path / "trace.jsonl"))

    runner.start_profile("tp2-best")
    runner.stop_profile()

    assert events == [("start", "tp2-best"), ("stop", None)]


def test_runner_returns_malformed_tool_call_for_agent_format_recovery(
    tmp_path: Path,
) -> None:
    backend = _AgentBackend()
    backend.decode = lambda tokens, skip_special_tokens=False: (
        '<tool_call>{"name":"bash","arguments":broken}</tool_call>'
    )
    runner = ConditionalISRunner(backend, _runner_config(tmp_path / "trace.jsonl"))

    result = runner.query(
        [{"role": "user", "content": "fix it"}], request_id="malformed", seed=3
    )

    assert result.message["extra"]["actions"] == []
    assert "invalid tool-call JSON" in result.diagnostics["tool_call_parse_error"]


def test_model_client_uses_stable_request_identity() -> None:
    class StubModel(ConditionalISModel):
        def __init__(self):
            super().__init__(seed=11)
            self.payloads = []

        def _post(self, payload):
            self.payloads.append(payload)
            return {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [],
                    "extra": {
                        "actions": [{"command": "pwd", "tool_call_id": "call_1"}],
                        "cost": 0.0,
                    },
                }
            }

    model = StubModel()
    messages = [{"role": "user", "content": "task"}]
    model.query(messages)
    model.query(messages)

    assert model.config.model_name.startswith("conditional-is/")
    assert model.payloads[0]["request_id"] == model.payloads[1]["request_id"]
    assert model.payloads[0]["seed"] == model.payloads[1]["seed"]


def test_service_config_expands_required_model_path(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "service.toml"
    source.write_text(
        "\n".join(
            [
                "[models]",
                'base = "${MODEL_ROOT}/model"',
                "[generation]",
                "max_new_tokens = 2",
                "[sampling]",
                "temperature = 1.0",
                "[conditional_is]",
                "candidate_count = 2",
                "[reward]",
                'kind = "sequence_log_probability"',
            ]
        )
    )
    monkeypatch.setenv("MODEL_ROOT", "/models")

    assert load_service_config(source)["models"]["base"] == "/models/model"


def test_service_config_applies_existing_typed_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "service.toml"
    source.write_text(
        """
[models]
base = "${MODEL_ROOT}/model"
[generation]
max_new_tokens = 2
[sampling]
temperature = 1.0
[conditional_is]
candidate_count = 2
[reward]
kind = "sequence_log_probability"
[vllm]
max_num_seqs = 128
""".strip()
    )
    monkeypatch.setenv("MODEL_ROOT", "/models")

    config = load_service_config(
        source,
        overrides=("vllm.max_num_seqs=256", "sampling.temperature=0.7"),
    )

    assert config["vllm"]["max_num_seqs"] == 256
    assert config["sampling"]["temperature"] == 0.7
    with pytest.raises(ValueError, match="does not exist"):
        load_service_config(source, overrides=("vllm.unknown=1",))
