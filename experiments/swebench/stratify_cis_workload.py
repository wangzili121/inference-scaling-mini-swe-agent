#!/usr/bin/env python3
"""Freeze deterministic prompt-length strata from a CIS JSONL workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LengthRange:
    name: str
    minimum: int
    maximum: int | None

    def contains(self, tokens: int) -> bool:
        return tokens >= self.minimum and (
            self.maximum is None or tokens < self.maximum
        )


def _parse_range(value: str) -> LengthRange:
    parts = value.split(":")
    if len(parts) != 3 or not parts[0]:
        raise argparse.ArgumentTypeError("range must be NAME:MIN:MAX")
    try:
        minimum = int(parts[1])
        maximum = None if not parts[2] else int(parts[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError("range bounds must be integers") from error
    if minimum < 0 or (maximum is not None and maximum <= minimum):
        raise argparse.ArgumentTypeError("range must satisfy 0 <= MIN < MAX")
    return LengthRange(parts[0], minimum, maximum)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_tokens(row: dict[str, Any]) -> int:
    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("workload row is missing diagnostics")
    tokens = diagnostics.get("prompt_tokens")
    if not isinstance(tokens, int) or tokens <= 0:
        raise ValueError("workload row has invalid diagnostics.prompt_tokens")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--range",
        dest="ranges",
        action="append",
        type=_parse_range,
        required=True,
        help="NAME:MIN:MAX, with an empty MAX for an open upper bound",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    names = [length_range.name for length_range in args.ranges]
    if len(names) != len(set(names)):
        parser.error("range names must be unique")

    for range_index, length_range in enumerate(args.ranges):
        eligible = [
            (index, row)
            for index, row in enumerate(rows)
            if length_range.contains(_prompt_tokens(row))
        ]
        if len(eligible) < args.count:
            parser.error(
                f"range {length_range.name!r} has {len(eligible)} rows, "
                f"fewer than requested {args.count}"
            )
        rng = random.Random(args.seed + range_index)
        selected = sorted(rng.sample(eligible, args.count), key=lambda item: item[0])
        output_dir = args.output_root / length_range.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "public-256.jsonl"
        with output_path.open("w", encoding="utf-8") as stream:
            for _, row in selected:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        lengths = sorted(_prompt_tokens(row) for _, row in selected)
        manifest = {
            "schema_version": 1,
            "source": str(args.input),
            "source_sha256": _sha256(args.input),
            "workload_sha256": _sha256(output_path),
            "seed": args.seed + range_index,
            "range": {
                "name": length_range.name,
                "minimum": length_range.minimum,
                "maximum_exclusive": length_range.maximum,
            },
            "requests": len(selected),
            "prompt_tokens": {
                "minimum": lengths[0],
                "median": (
                    lengths[(len(lengths) - 1) // 2] + lengths[len(lengths) // 2]
                )
                / 2,
                "maximum": lengths[-1],
            },
            "source_indices": [index for index, _ in selected],
        }
        (output_dir / "public-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
