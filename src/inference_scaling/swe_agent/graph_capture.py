"""Derive an ACL-graph capture-size candidate from real decode batch shapes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from inference_scaling.swe_agent.profile_analysis import analyze_service_profile


def _round_up(value: float) -> int:
    integer = max(1, int(math.ceil(value)))
    if integer <= 8:
        return 1 << (integer - 1).bit_length()
    quantum = 8 if integer <= 128 else 16 if integer <= 512 else 32
    return int(math.ceil(integer / quantum) * quantum)


def graph_capture_candidates(service: dict[str, Any]) -> dict[str, Any]:
    columns = service.get("batch_shape_quantiles") or {}
    ranked = sorted(
        columns.items(),
        key=lambda item: (
            "scheduled_token" not in item[0].lower().replace(" ", "_"),
            "batch_size" not in item[0].lower().replace(" ", "_"),
            item[0],
        ),
    )
    if not ranked:
        return {
            "status": "insufficient_service_profile",
            "reason": "batch.csv has no recognized scheduled-token or batch-size column",
        }
    column, summary = ranked[0]
    sizes = sorted(
        {
            _round_up(float(summary[name]))
            for name in ("p50", "p75", "p90", "p95", "p99", "maximum")
            if name in summary
        }
    )
    return {
        "status": "candidate_only_requires_ab",
        "source_column": column,
        "source_quantiles": summary,
        "capture_sizes": sizes,
        "override": "vllm.engine_kwargs.compilation_config="
        + json.dumps(
            {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": sizes,
            },
            separators=(",", ":"),
        ),
        "warning": "Do not retain until an end-to-end A/B beats vLLM's default buckets.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = graph_capture_candidates(
        analyze_service_profile(Path(args.service_profile))
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
