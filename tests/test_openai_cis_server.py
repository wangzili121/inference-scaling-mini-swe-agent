from __future__ import annotations

import json
import threading
import urllib.request
from types import SimpleNamespace

from inference_scaling.swe_agent.server import ConditionalISHTTPServer, _handler
from inference_scaling.swe_agent.service import QueryResult


class _Runner:
    instance_id = "test-instance"
    default_seed = 17
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0)

    def __init__(self) -> None:
        self.calls = []

    def backend_snapshot(self):
        return {"engine_requests": 40}

    def query(self, messages, *, request_id, seed, conditional_overrides=None):
        self.calls.append((messages, request_id, seed, conditional_overrides))
        diagnostics = {
            "prompt_tokens": 12,
            "completion_tokens": 64,
            "finish_reason": "length",
            "candidate_count": 8,
            "rollout_count": 3,
            "steps": 1,
            "stage_seconds": {"candidate": 1.0, "rollout": 2.0},
        }
        return QueryResult(
            message={
                "role": "assistant",
                "content": "answer",
                "tool_calls": [],
                "extra": {"raw_completion": "answer"},
            },
            diagnostics=diagnostics,
        )

    def query_direct(self, messages, *, request_id, seed, max_new_tokens):
        self.calls.append((messages, request_id, seed, {"direct": max_new_tokens}))
        return QueryResult(
            message={
                "role": "assistant",
                "content": "direct answer",
                "tool_calls": [],
                "extra": {"raw_completion": "direct answer"},
            },
            diagnostics={
                "mode": "direct_ar",
                "prompt_tokens": 12,
                "completion_tokens": max_new_tokens,
                "finish_reason": "length",
            },
        )


def test_openai_chat_completion_wraps_a_complete_cis_job() -> None:
    runner = _Runner()
    server = ConditionalISHTTPServer(("127.0.0.1", 0), _handler(runner))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        payload = {
            "model": "dsv4-cis",
            "messages": [{"role": "user", "content": "fix it"}],
            "max_tokens": 64,
            "temperature": 1.0,
            "top_p": 1.0,
            "stream": False,
            "request_id": "openai-test",
        }
        request = urllib.request.Request(
            endpoint + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read())
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "answer"
        assert result["usage"]["total_tokens"] == 76
        assert result["conditional_is"]["candidate_count"] == 8
        assert runner.calls[0][3] == {"total_length": 64}
        with urllib.request.urlopen(endpoint + "/v1/models") as response:
            models = json.loads(response.read())
        assert models["data"][0]["id"] == "dsv4-cis"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_openai_benchmark_payload_accepts_nullable_n_and_bounded_seed() -> None:
    runner = _Runner()
    server = ConditionalISHTTPServer(("127.0.0.1", 0), _handler(runner))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        payload = {
            "model": "dsv4-cis",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "fix it"}],
                }
            ],
            "max_completion_tokens": 64,
            "temperature": 1.0,
            "top_p": 1.0,
            "n": None,
            "stream": False,
            "stream_options": {"include_usage": True},
        }
        request = urllib.request.Request(
            endpoint + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Request-Id": "bench-test-0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read())
        assert result["object"] == "chat.completion"
        assert runner.calls[0][1] == "bench-test-0"
        assert 0 <= runner.calls[0][2] <= (1 << 63) - 1
        assert runner.calls[0][3] == {"total_length": 64}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_openai_direct_endpoint_bypasses_conditional_is() -> None:
    runner = _Runner()
    server = ConditionalISHTTPServer(("127.0.0.1", 0), _handler(runner))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        payload = {
            "model": "dsv4-cis",
            "messages": [{"role": "user", "content": "fix it"}],
            "max_completion_tokens": 32,
            "temperature": 1.0,
            "top_p": 1.0,
            "stream": False,
            "request_id": "direct-test",
        }
        request = urllib.request.Request(
            endpoint + "/v1/direct/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read())
        assert result["choices"][0]["message"]["content"] == "direct answer"
        assert result["conditional_is"]["mode"] == "direct_ar"
        assert runner.calls[0][3] == {"direct": 32}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
