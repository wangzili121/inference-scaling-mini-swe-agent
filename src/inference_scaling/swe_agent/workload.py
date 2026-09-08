"""Freeze real mini-SWE-agent model calls into deterministic tuning manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as stream:
        for record in records:
            payload = json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            stream.write(payload + b"\n")
            digest.update(payload + b"\n")
    return digest.hexdigest()


def freeze_workload(
    trace_path: str | Path,
    output_directory: str | Path,
    *,
    seed: int = 20260908,
    total: int = 128,
) -> dict[str, Any]:
    if total <= 0 or total % 2:
        raise ValueError("total must be a positive even number")
    records = _load_jsonl(Path(trace_path))
    successful: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        request_id = str(record.get("request_id", ""))
        actions = record.get("message", {}).get("extra", {}).get("actions", [])
        if not request_id or request_id in seen or not actions:
            continue
        if not isinstance(record.get("messages"), list):
            continue
        seen.add(request_id)
        successful.append(record)
    if len(successful) < total:
        raise ValueError(
            f"need {total} unique successful calls, found {len(successful)}"
        )
    rng = random.Random(seed)
    rng.shuffle(successful)
    selected = successful[:total]
    midpoint = total // 2
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tune_path = output / f"tune-{midpoint}.jsonl"
    holdout_path = output / f"holdout-{midpoint}.jsonl"
    tune_sha = _write_jsonl(tune_path, selected[:midpoint])
    holdout_sha = _write_jsonl(holdout_path, selected[midpoint:])
    metadata = {
        "schema_version": 1,
        "source": str(Path(trace_path).resolve()),
        "seed": seed,
        "total": total,
        "tune": {"path": tune_path.name, "count": midpoint, "sha256": tune_sha},
        "holdout": {
            "path": holdout_path.name,
            "count": midpoint,
            "sha256": holdout_sha,
        },
        "prompt_tokens": {
            "minimum": min(
                record["diagnostics"]["prompt_tokens"] for record in selected
            ),
            "maximum": max(
                record["diagnostics"]["prompt_tokens"] for record in selected
            ),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--total", type=int, default=128)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_workload(
                args.trace,
                args.output_directory,
                seed=args.seed,
                total=args.total,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
