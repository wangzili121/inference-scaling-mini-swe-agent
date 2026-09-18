#!/usr/bin/env python3
"""Verify the OpenAI-compatible wrapper still executes one complete CIS job."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8123")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    payload = {
        "model": "dsv4-cis",
        "messages": [
            {
                "role": "user",
                "content": "Write a Python function that returns the larger of two integers.",
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 20260908,
        "stream": False,
        "request_id": "openai-smoke-20260908",
    }
    request = urllib.request.Request(
        args.endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {error.code} from CIS service: {body}") from error
    if result.get("object") != "chat.completion" or not result.get("choices"):
        raise SystemExit("invalid OpenAI-compatible response")
    diagnostics = result.get("conditional_is", {})
    required = {"candidate_count", "rollout_count", "steps", "stage_seconds"}
    missing = sorted(required - set(diagnostics))
    if missing:
        raise SystemExit(f"response lacks CIS diagnostics: {missing}")
    sampled = int(diagnostics.get("backend_delta", {}).get("sampled_sequences", 0))
    if diagnostics["steps"] < 2 or sampled <= int(diagnostics["candidate_count"]):
        raise SystemExit(
            "smoke did not exercise a candidate-to-rollout transition; "
            "increase --max-tokens"
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "model": result.get("model"),
                "usage": result.get("usage"),
                "candidate_count": diagnostics["candidate_count"],
                "rollout_count": diagnostics["rollout_count"],
                "steps": diagnostics["steps"],
                "reward": diagnostics.get("reward"),
                "reward_execution": diagnostics.get("reward_execution"),
                "total_seconds": diagnostics.get("total_seconds"),
                "sampled_sequences": sampled,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
