"""HTTP entry point for one persistent Conditional IS backend."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
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
            if self.path != "/v1/query":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = payload["messages"]
                request_id = str(payload["request_id"])
                seed = int(payload["seed"])
                conditional_overrides = payload.get("conditional_is")
                if conditional_overrides is not None and not isinstance(
                    conditional_overrides, dict
                ):
                    raise TypeError("conditional_is must be an object")
                result = runner.query(
                    messages,
                    request_id=request_id,
                    seed=seed,
                    conditional_overrides=conditional_overrides,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except Exception as error:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"{type(error).__name__}: {error}"},
                )
                return
            self._json(
                HTTPStatus.OK,
                {"message": result.message, "diagnostics": result.diagnostics},
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
