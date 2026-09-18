#!/usr/bin/env python3
"""Generate a deterministic mixed-length coding capacity workload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASKS = (
    "Find the concurrency bug and provide a minimal safe patch.",
    "Implement the missing cache invalidation path and explain its complexity.",
    "Review this parser for correctness and write the corrected implementation.",
    "Refactor this request router without changing externally visible behavior.",
    "Diagnose the flaky test and propose a deterministic fix.",
    "Add bounded retries while preserving idempotency and cancellation.",
    "Repair the transaction logic and state the invariant your patch maintains.",
    "Optimize this hot path without changing its return values or exceptions.",
)


def source_line(case: int, index: int) -> str:
    return (
        f"def repository_case_{case}_{index}(state, key, value):\n"
        f"    current = state.get(key, {index % 11})\n"
        f"    if (current + {case}) % {index % 17 + 2} == 0:\n"
        f"        state[key] = current + value\n"
        f"    return state.get(key), len(state)\n\n"
    )


def prompt(case: int, target_tokens: int) -> str:
    header = (
        "You are reviewing a real Python service repository. "
        + TASKS[case % len(TASKS)]
        + " Return a complete patch, the reasoning behind it, and focused tests.\n\n"
        + f"Repository snapshot {case}:\n```python\n"
    )
    target_chars = target_tokens * 4
    parts = [header]
    index = 0
    while sum(map(len, parts)) < target_chars:
        parts.append(source_line(case, index))
        index += 1
    parts.append("```\n")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument(
        "--input-buckets",
        default="4096,4096,4096,4096,8192,8192,8192,16384",
        help="comma-separated target prompt-token buckets",
    )
    parser.add_argument("--profile-name", default="coding-mixed")
    args = parser.parse_args()
    if args.count <= 0 or args.output_tokens <= 0:
        raise SystemExit("count and output-tokens must be positive")
    try:
        buckets = tuple(int(value) for value in args.input_buckets.split(","))
    except ValueError as error:
        raise SystemExit("input buckets must be comma-separated integers") from error
    if not buckets or any(value <= 0 for value in buckets):
        raise SystemExit("input buckets must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    lengths = []
    with args.output.open("wb") as stream:
        for index in range(args.count):
            target = buckets[index % len(buckets)]
            value = {
                "prompt": prompt(index, target),
                "output_tokens": args.output_tokens,
            }
            payload = json.dumps(
                value, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            stream.write(payload + b"\n")
            digest.update(payload + b"\n")
            lengths.append(target)
    manifest = {
        "schema_version": 1,
        "kind": "synthetic_coding_capacity",
        "profile_name": args.profile_name,
        "count": args.count,
        "target_input_tokens": lengths,
        "output_tokens": args.output_tokens,
        "sha256": digest.hexdigest(),
        "warning": "Capacity workload only; use frozen agent calls for final claims.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
