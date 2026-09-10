"""Embed an exported CIS request profile into the interactive HTML fragment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MARKER = "__CIS_REQUEST_PROFILE_DATA__"


def _compact_npu(payload: dict[str, Any]) -> dict[str, Any]:
    timeline = payload["detailed_timeline"]["kernel"]
    ascend = payload["ascend"]
    top_kernels: dict[str, float] = {}
    for rank in ascend.get("ranks", ()):
        for name, duration in rank.get("top_kernel_types_us", {}).items():
            top_kernels[name] = top_kernels.get(name, 0.0) + float(duration)
    busy_us = sum(float(rank.get("device_busy_us", 0.0)) for rank in ascend["ranks"])
    attention_us = sum(
        duration
        for name, duration in top_kernels.items()
        if "attention" in name.lower()
    )
    return {
        "source": payload.get("torch_source"),
        "duration_ms": timeline["duration_ms"],
        "raw_event_count": timeline["raw_event_count"],
        "categories": timeline["categories"],
        "ranks": [
            {
                "rank": rank["rank"],
                "bin_ms": rank["overview_bin_ms"],
                "busy": rank["busy_q"],
                "hccl": rank["hccl_q"],
                "dominant": rank["dominant_category"],
            }
            for rank in timeline["ranks"]
        ],
        "phase_summary": payload["detailed_timeline"]["phase_summary"],
        "active_stages": payload["detailed_timeline"]["active_stages"],
        "rank_summary": [
            {
                "busy_ratio": rank["device_busy_ratio"],
                "exposed_hccl_profile_ratio": rank[
                    "exposed_communication_profile_ratio"
                ],
            }
            for rank in ascend["ranks"]
        ],
        "attention_busy_ratio": attention_us / busy_us if busy_us else None,
        "top_kernels": sorted(
            ([name, duration] for name, duration in top_kernels.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--npu-data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = args.template.read_text(encoding="utf-8")
    if template.count(MARKER) != 1:
        raise RuntimeError("template must contain exactly one data marker")
    data = json.loads(args.data.read_text(encoding="utf-8"))
    if args.npu_data is not None:
        npu = json.loads(args.npu_data.read_text(encoding="utf-8"))
        data["npu"] = _compact_npu(npu)
    rendered = template.replace(MARKER, json.dumps(data, separators=(",", ":")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
