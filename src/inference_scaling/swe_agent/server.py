"""HTTP entry point for one persistent Conditional IS backend."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import signal
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from inference_scaling.swe_agent.service import ConditionalISRunner


def _verify_requested_runtime_features() -> None:
    if os.getenv("VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE") != "1":
        return
    try:
        import torch
        import vllm_ascend.sample.sampler as sampler
        importlib.import_module("vllm_ascend.vllm_ascend_C")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "native categorical sampling was requested but its runtime failed to load"
        ) from error
    if not getattr(sampler, "_CATEGORICAL_SAMPLE_ENABLED", False):
        raise RuntimeError(
            "native categorical sampling was requested but the patched sampler is absent"
        )
    if not hasattr(torch.ops._C_ascend, "npu_categorical_sample"):
        raise RuntimeError(
            "native categorical sampling was requested but the Ascend operator is absent"
        )
    print("native categorical sampler preflight passed", flush=True)


class ConditionalISHTTPServer(ThreadingHTTPServer):
    """Keep saturated bursts in the socket queue while the engine admits jobs."""

    request_queue_size = 1024
    daemon_threads = True
    block_on_close = False


def _handler(runner: ConditionalISRunner) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _event_stream(self, payloads: list[dict[str, Any]]) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for payload in payloads:
                self.wfile.write(
                    b"data: "
                    + json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    + b"\n\n"
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._json(HTTPStatus.OK, {"status": "ok"})
            elif self.path == "/v1/diagnostics":
                self._json(
                    HTTPStatus.OK,
                    {
                        "instance_id": runner.instance_id,
                        "backend": runner.backend_snapshot(),
                    },
                )
            elif self.path == "/v1/models":
                self._json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": "dsv4-cis",
                                "object": "model",
                                "created": 0,
                                "owned_by": "inference-scaling",
                            }
                        ],
                    },
                )
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path in {"/v1/profile/start", "/v1/profile/stop"}:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = (
                        json.loads(self.rfile.read(length).decode("utf-8"))
                        if length
                        else {}
                    )
                    if not isinstance(payload, dict):
                        raise TypeError("profile payload must be an object")
                    if self.path.endswith("/start"):
                        prefix = payload.get("prefix")
                        runner.start_profile(None if prefix is None else str(prefix))
                        status = "started"
                    else:
                        runner.stop_profile()
                        status = "stopped"
                except (TypeError, ValueError, RuntimeError) as error:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                self._json(HTTPStatus.OK, {"status": status})
                return
            if self.path not in {
                "/v1/query",
                "/v1/chat/completions",
                "/v1/direct/chat/completions",
            }:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = payload["messages"]
                openai_request = self.path in {
                    "/v1/chat/completions",
                    "/v1/direct/chat/completions",
                }
                direct_request = self.path == "/v1/direct/chat/completions"
                if openai_request:
                    request_id = str(
                        payload.get("request_id")
                        or self.headers.get("X-Request-Id")
                        or f"chatcmpl-{uuid.uuid4().hex}"
                    )
                    configured_seed = int(runner.default_seed)
                    seed_material = json.dumps(
                        {
                            "messages": messages,
                            "max_tokens": payload.get(
                                "max_completion_tokens", payload.get("max_tokens")
                            ),
                            "model": payload.get("model", "dsv4-cis"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    default_seed = (
                        int.from_bytes(
                            hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
                            "big",
                        )
                        ^ configured_seed
                    ) & ((1 << 63) - 1)
                    seed = int(payload.get("seed", default_seed))
                    requested_n = payload.get("n")
                    if requested_n is not None and int(requested_n) != 1:
                        raise ValueError("Conditional IS supports only n=1")
                    for name, configured in (
                        ("temperature", runner.sampling.temperature),
                        ("top_p", runner.sampling.top_p),
                    ):
                        if name in payload and payload[name] is not None:
                            if abs(float(payload[name]) - float(configured)) > 1e-9:
                                raise ValueError(
                                    f"{name} is fixed by the CIS service at {configured}"
                                )
                else:
                    request_id = str(payload["request_id"])
                    seed = int(payload["seed"])
                conditional_overrides = payload.get("conditional_is")
                if conditional_overrides is not None and not isinstance(
                    conditional_overrides, dict
                ):
                    raise TypeError("conditional_is must be an object")
                if openai_request:
                    maximum = payload.get("max_completion_tokens", payload.get("max_tokens"))
                    if maximum is not None:
                        conditional_overrides = dict(conditional_overrides or {})
                        conditional_overrides["total_length"] = int(maximum)
                if direct_request:
                    direct_maximum = int(
                        maximum
                        if maximum is not None
                        else getattr(runner, "maximum", 512)
                    )
                    result = runner.query_direct(
                        messages,
                        request_id=request_id,
                        seed=seed,
                        max_new_tokens=direct_maximum,
                    )
                else:
                    result = runner.query(
                        messages,
                        request_id=request_id,
                        seed=seed,
                        conditional_overrides=conditional_overrides,
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                request_id = None
                payload_keys: list[str] = []
                if isinstance(locals().get("payload"), dict):
                    request_id = payload.get("request_id") or self.headers.get(
                        "X-Request-Id"
                    )
                    payload_keys = sorted(str(key) for key in payload)
                print(
                    "CIS request rejected: "
                    f"request_id={request_id!r} "
                    f"error={type(error).__name__}: {error}; "
                    f"payload_keys={payload_keys}",
                    file=sys.stderr,
                    flush=True,
                )
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except Exception as error:
                traceback.print_exc()
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"{type(error).__name__}: {error}"},
                )
                return
            if self.path == "/v1/query":
                self._json(
                    HTTPStatus.OK,
                    {"message": result.message, "diagnostics": result.diagnostics},
                )
                return
            created = int(time.time())
            completion_id = request_id if request_id.startswith("chatcmpl-") else f"chatcmpl-{request_id}"
            raw = result.message.get("extra", {}).get("raw_completion")
            content = result.message.get("content")
            if content is None and isinstance(raw, str):
                content = raw
            finish_reason = result.diagnostics.get("finish_reason", "stop")
            if finish_reason == "eos":
                finish_reason = "stop"
            usage = {
                "prompt_tokens": int(result.diagnostics["prompt_tokens"]),
                "completion_tokens": int(result.diagnostics["completion_tokens"]),
                "total_tokens": int(result.diagnostics["prompt_tokens"])
                + int(result.diagnostics["completion_tokens"]),
            }
            if bool(payload.get("stream", False)):
                base = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": str(payload.get("model", "dsv4-cis")),
                }
                self._event_stream(
                    [
                        {
                            **base,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": content or "",
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        },
                        {
                            **base,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }
                            ],
                            "usage": usage,
                            "conditional_is": result.diagnostics,
                        },
                    ]
                )
                return
            self._json(
                HTTPStatus.OK,
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": str(payload.get("model", "dsv4-cis")),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": result.message.get("tool_calls", []),
                            },
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": usage,
                    "conditional_is": result.diagnostics,
                },
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    _verify_requested_runtime_features()
    runner = ConditionalISRunner.from_toml(args.config, overrides=args.overrides)
    server = ConditionalISHTTPServer((args.host, args.port), _handler(runner))

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        runner.close()


if __name__ == "__main__":
    main()
