#!/usr/bin/env python3
"""Summarize vLLM capacity JSON files into stable JSON, CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "max_concurrency",
    "completed",
    "failed",
    "duration",
    "request_throughput",
    "mean_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
)


def show(row: dict[str, object], name: str) -> str:
    value = row.get(name)
    if value is None:
        return ""
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(value, dict) or "request_throughput" not in value:
            continue
        row = {field: value.get(field) for field in FIELDS}
        row["source"] = str(path.relative_to(args.directory))
        rows.append(row)
    rows.sort(key=lambda row: (row.get("max_concurrency") or 0, row["source"]))
    if not rows:
        raise SystemExit(f"no vLLM benchmark result JSON found below {args.directory}")
    (args.directory / "capacity-summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.directory / "capacity-summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=(*FIELDS, "source"))
        writer.writeheader()
        writer.writerows(rows)
    header = "| concurrency | completed | failed | jobs/s | mean E2E ms | P95 ms | P99 ms |"
    separator = "|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, separator]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                show(row, name)
                for name in (
                    "max_concurrency",
                    "completed",
                    "failed",
                    "request_throughput",
                    "mean_e2el_ms",
                    "p95_e2el_ms",
                    "p99_e2el_ms",
                )
            )
            + " |"
        )
    (args.directory / "capacity-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
