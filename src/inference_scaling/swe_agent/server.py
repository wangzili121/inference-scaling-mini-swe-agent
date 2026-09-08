"""HTTP entry point for one persistent Conditional IS backend."""

from __future__ import annotations

import argparse
import json
import signal
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from inference_scaling.swe_agent.service import ConditionalISRunner


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
    runner = ConditionalISRunner.from_toml(args.config, overrides=args.overrides)
    server = ThreadingHTTPServer((args.host, args.port), _handler(runner))

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
