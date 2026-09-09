"""Render reproducible Chinese reports and a unified CIS profiling timeline."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any, Iterable

from inference_scaling.swe_agent.profile_analysis import (
    analyze_algorithm_trace,
    analyze_ascend_profile,
    analyze_benchmark,
    analyze_service_profile,
    recommendations,
)
from inference_scaling.swe_agent.graph_capture import graph_capture_candidates
from inference_scaling.swe_agent.artifacts import refresh_artifact_manifest


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def _bar_chart(path: Path, title: str, values: dict[str, float]) -> None:
    width, row_height = 960, 34
    height = 70 + row_height * max(1, len(values))
    maximum = max(values.values(), default=1.0) or 1.0
    rows = []
    for index, (name, value) in enumerate(values.items()):
        y = 52 + index * row_height
        bar_width = 620 * value / maximum
        rows.append(
            f'<text x="12" y="{y + 16}" font-size="14">{html.escape(name)}</text>'
            f'<rect x="260" y="{y}" width="{bar_width:.2f}" height="20" fill="#087e8b"/>'
            f'<text x="{270 + bar_width:.2f}" y="{y + 16}" font-size="13">{value:.4f}</text>'
        )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="12" y="28" font-size="20" font-weight="600">{html.escape(title)}</text>'
        + "".join(rows)
        + "</svg>",
        encoding="utf-8",
    )


def _algorithm_events(trace_directory: Path) -> list[dict[str, Any]]:
    events = []
    for path in sorted(trace_directory.rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            request_id = str(record.get("request_id", "unknown"))
            for stage in record.get("stage_events", ()):
                if "start_unix_us" not in stage:
                    continue
                events.append(
                    {
                        "name": f"CIS/{stage.get('name', 'unknown')}",
                        "cat": "conditional-is",
                        "ph": "X",
                        "ts": float(stage["start_unix_us"]),
                        "dur": float(stage.get("duration_us", 0.0)),
                        "pid": f"cis:{stage.get('instance_id', 'unknown')}",
                        "tid": f"job:{request_id}",
                        "args": dict(stage),
                    }
                )
    return events


def _kernel_events(
    profile_directory: Path, window_start_us: float
) -> list[dict[str, Any]]:
    events = []
    for rank_index, path in enumerate(
        sorted(profile_directory.rglob("kernel_details.csv"))
    ):
        rows = []
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                try:
                    start = float(row.get("Start Time(us)", "0"))
                    duration = float(row.get("Duration(us)", "0"))
                except ValueError:
                    continue
                if duration > 0:
                    rows.append((start, duration, row))
        if not rows:
            continue
        origin = min(start for start, _, _ in rows)
        for start, duration, row in rows:
            name = str(row.get("Name") or row.get("Type") or "kernel")
            lowered = name.lower()
            events.append(
                {
                    "name": name,
                    "cat": "hccl"
                    if "hcom" in lowered or "hccl" in lowered
                    else "npu-kernel",
                    "ph": "X",
                    "ts": window_start_us + start - origin,
                    "dur": duration,
                    "pid": f"npu-rank-{rank_index}",
                    "tid": str(row.get("Accelerator Core") or "device"),
                    "args": {
                        "source": str(path),
                        "clock_alignment": "first-kernel-to-profile-window",
                    },
                }
            )
    return events


def _service_events(profile_directory: Path) -> list[dict[str, Any]]:
    events = []
    for path in sorted(profile_directory.rglob("batch.csv")):
        instance = next(
            (part for part in path.parts if part.startswith("profile-")),
            path.parent.name,
        )
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                try:
                    start = float(row.get("start_time(ms)", "0")) * 1_000
                    duration = float(row.get("during_time(ms)", "0")) * 1_000
                except ValueError:
                    continue
                if start <= 0 or duration <= 0:
                    continue
                events.append(
                    {
                        "name": f"vLLM/{row.get('name') or 'batch'}",
                        "cat": "vllm-service",
                        "ph": "X",
                        "ts": start,
                        "dur": duration,
                        "pid": f"service:{instance}",
                        "tid": str(row.get("batch_type") or "batch"),
                        "args": {
                            "batch_size": row.get("batch_size"),
                            "source": str(path),
                        },
                    }
                )
    return events


def build_unified_timeline(profile_directory: Path, output: Path) -> dict[str, Any]:
    benchmark = json.loads((profile_directory / "benchmark.json").read_text())
    window = benchmark.get("profile", {}).get("window", {})
    window_start_us = (
        float(window.get("started_at", benchmark.get("started_at", 0.0))) * 1e6
    )
    events = _algorithm_events(profile_directory / "algorithm-traces")
    events.extend(_service_events(profile_directory))
    events.extend(_kernel_events(profile_directory, window_start_us))
    events.sort(key=lambda item: float(item.get("ts", 0.0)))
    payload = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "algorithm_clock": "unix-us",
            "kernel_clock_alignment": "each rank's first exported kernel is aligned to the measured profile window start",
            "warning": "Use raw trace_view.json for exact device timing; this merged view is for cross-layer orientation.",
        },
    }
    output.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return {"events": len(events), "path": str(output)}


def render_profile_report(profile_directory: str | Path) -> dict[str, Any]:
    root = Path(profile_directory).resolve()
    derived = root / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    benchmark = analyze_benchmark(root / "benchmark.json")
    algorithm = analyze_algorithm_trace(root / "algorithm-traces")
    ascend = analyze_ascend_profile(root)
    service = analyze_service_profile(root)
    actions = recommendations(benchmark, algorithm, ascend)
    graph_capture = graph_capture_candidates(service)
    analysis = {
        "schema_version": 1,
        "benchmark": benchmark,
        "algorithm": algorithm,
        "ascend": ascend,
        "service": service,
        "recommendations": actions,
        "graph_capture": graph_capture,
    }
    (derived / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    (derived / "graph-capture-candidates.json").write_text(
        json.dumps(graph_capture, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        derived / "stage-summary.csv",
        (
            {
                "stage": stage,
                "seconds": seconds,
                "share": algorithm["stage_share"].get(stage, 0.0),
                "p95_seconds": algorithm["stage_p95_seconds"].get(stage, 0.0),
            }
            for stage, seconds in algorithm["stage_seconds"].items()
            if stage != "block"
        ),
    )
    _write_csv(
        derived / "rank-summary.csv",
        (
            {
                "rank": index,
                "device_busy_ratio": item["device_busy_ratio"],
                "communication_us": item["communication_us"],
                "exposed_communication_ratio": item["exposed_communication_ratio"],
            }
            for index, item in enumerate(ascend["ranks"])
        ),
    )
    _bar_chart(
        derived / "stage-share.svg",
        "Conditional IS stage share",
        algorithm["stage_share"],
    )
    _bar_chart(
        derived / "rank-utilization.svg",
        "NPU busy ratio by rank",
        {
            f"rank-{index}": item["device_busy_ratio"]
            for index, item in enumerate(ascend["ranks"])
        },
    )
    timeline = build_unified_timeline(root, derived / "unified-timeline.json")
    report_lines = [
        "# Conditional IS Profiling 报告",
        "",
        "## 运行结论",
        "",
        f"- 成功率：{benchmark['success_rate']:.2%}",
        f"- 完整 CIS jobs/s：{benchmark['jobs_per_second']:.4f}",
        f"- P95：{float(benchmark['latency_seconds'].get('p95', 0.0)):.3f}s",
        f"- APC token hit ratio：{benchmark['apc_token_hit_ratio']:.2%}",
        f"- block 内未归入叶子阶段的间隙：{algorithm['block_gap_share']:.2%}",
        f"- NPU busy ratio 中位数：{ascend['device_busy_ratio']['median']:.2%}",
        f"- 暴露通信占 profile 窗口：{ascend['exposed_communication_profile_ratio']:.2%}",
        f"- MS Service batch/scheduler 证据：{'有' if service['has_batch_scheduler_evidence'] else '无'}",
        "",
        "## 算法阶段",
        "",
        "| 阶段 | 总时间(s) | 占比 | P95(s) |",
        "|---|---:|---:|---:|",
    ]
    for stage, seconds in sorted(
        algorithm["stage_seconds"].items(), key=lambda item: item[1], reverse=True
    ):
        if stage == "block":
            continue
        report_lines.append(
            f"| {stage} | {seconds:.3f} | {algorithm['stage_share'].get(stage, 0.0):.2%} | "
            f"{algorithm['stage_p95_seconds'].get(stage, 0.0):.3f} |"
        )
    report_lines.extend(["", "## 证据驱动的后续建议", ""])
    if actions:
        for item in actions:
            report_lines.append(
                f"- `{item['trigger']}`（{float(item['value']):.2%}）：{item['action']}。"
            )
    else:
        report_lines.append(
            "- 当前阈值未触发定向优化建议；需结合其他算法配置判断瓶颈是否迁移。"
        )
    report_lines.extend(
        [
            "",
            "## 可复查数据",
            "",
            "- `benchmark.json`：无损请求级结果和部署参数。",
            "- `algorithm-traces/`：job/block/stage 原始事件。",
            "- `torch/` 或 `service/`：未修改的 profiler 原始目录。",
            "- `derived/analysis.json`：可重算汇总。",
            "- `derived/graph-capture-candidates.json`：由真实 batch 分位数生成、仍需 A/B 的 graph buckets。",
            f"- `derived/unified-timeline.json`：{timeline['events']} 个跨层事件，可用 Perfetto/Chrome 打开。",
            "- `artifact-manifest.json`：版本、命令、workload/config 哈希和全部文件校验值。",
            "",
            "> 合并时间线按 profile 窗口对齐各 rank 的首个 kernel，用于跨层定位；精确设备时序以原始 `trace_view.json`/`analysis.db` 为准。",
        ]
    )
    report_path = derived / "REPORT.zh-CN.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest_path = root / "artifact-manifest.json"
    if manifest_path.exists():
        refresh_artifact_manifest(root)
    return {
        "analysis": str(derived / "analysis.json"),
        "report": str(report_path),
        "timeline": timeline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-directory", required=True)
    args = parser.parse_args()
    print(json.dumps(render_profile_report(args.profile_directory), indent=2))


if __name__ == "__main__":
    main()
