"""Issue one ordinary CIS query and check the DeepSeek V4 bash-action path."""

from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8123")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    payload = {
        "request_id": "dsv4-0731-smoke",
        "seed": 20260908,
        "messages": [
            {"role": "system", "content": "You are a coding agent. Use the bash tool."},
            {"role": "user", "content": "Call bash with command: echo cis-ready"},
        ],
    }
    request = urllib.request.Request(
        args.endpoint.rstrip("/") + "/v1/query",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        result = json.load(response)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    actions = result.get("diagnostics", {}).get("actions", [])
    if not actions and not result.get("message", {}).get("tool_calls"):
        raise SystemExit("No bash tool call; inspect model text/parser before benchmarking")


if __name__ == "__main__":
    main()
