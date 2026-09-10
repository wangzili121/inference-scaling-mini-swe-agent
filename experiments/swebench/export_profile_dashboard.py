"""Export compact, reproducible data for the CIS profiling dashboard."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from inference_scaling.swe_agent.profile_analysis import (
    analyze_algorithm_trace,
    analyze_ascend_profile,
    analyze_benchmark,
)


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0").strip())
    except (AttributeError, ValueError):
        return 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _merge(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not result or start > result[-1][1]:
            result.append((start, end))
        else:
            result[-1] = (result[-1][0], max(result[-1][1], end))
    return result


def _kernel_category(name: str, kind: str) -> str:
    lowered = f"{name} {kind}".lower()
    if "hcom" in lowered or "hccl" in lowered:
        return "hccl"
    if "inferattentionscore" in lowered or "attention" in lowered:
        return "attention"
    if "moe" in lowered or "groupedmatmul" in lowered:
        return "moe"
    if "matmul" in lowered:
        return "gemm"
    if any(
        marker in lowered
        for marker in (
            "categorical",
            "random",
            "topk",
            "maskedfill",
            "masked_fill",
            "greater",
            "uniform",
            "exponential",
        )
    ):
        return "sampling"
    if "norm" in lowered or "rope" in lowered:
        return "norm_rope"
    return "other"


def _rank_identity(path: Path) -> tuple[int, int, int]:
    rendered = str(path)
    instance_match = re.search(r"/profile-(\d+)/", rendered)
    rank_match = re.search(r"_rank(\d+)_", rendered)
    instance = int(instance_match.group(1)) if instance_match else 0
    local_rank = int(rank_match.group(1)) if rank_match else 0
    return instance * 2 + local_rank, instance, local_rank


def _relative_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    origin_us: float,
    finish_us: float,
) -> list[list[float]]:
    clipped = (
        (max(start, origin_us), min(end, finish_us))
        for start, end in intervals
        if end > origin_us and start < finish_us
    )
    return [
        [round((start - origin_us) / 1_000.0, 4), round((end - start) / 1_000.0, 4)]
        for start, end in _merge(clipped)
    ]


def _active_stage_segments(
    root: Path,
    *,
    origin_us: float,
    finish_us: float,
) -> dict[str, dict[str, list[list[float]]]]:
    boundaries: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for path in sorted((root / "algorithm-traces").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for event in record.get("stage_events", ()):
                stage = str(event.get("name", "unknown"))
                if stage not in {"candidate", "rollout", "reward", "weight"}:
                    continue
                start = float(event.get("start_unix_us", 0.0))
                end = start + float(event.get("duration_us", 0.0))
                start = max(start, origin_us)
                end = min(end, finish_us)
                if end <= start:
                    continue
                instance = str(event.get("instance_id", "unknown"))
                boundaries[(instance, stage)].extend(((start, 1), (end, -1)))
    result: dict[str, dict[str, list[list[float]]]] = defaultdict(dict)
    for (instance, stage), values in boundaries.items():
        count = 0
        previous = origin_us
        segments = []
        for timestamp, delta in sorted(values, key=lambda item: (item[0], -item[1])):
            if timestamp > previous and count:
                segments.append(
                    [
                        round((previous - origin_us) / 1_000.0, 4),
                        round((timestamp - previous) / 1_000.0, 4),
                        count,
                    ]
                )
            count += delta
            previous = timestamp
        result[instance][stage] = segments
    return dict(result)


def _kernel_detail(
    root: Path,
    *,
    origin_us: float,
    finish_us: float,
    active_stages: dict[str, dict[str, list[list[float]]]],
    focus_ms: float = 120.0,
    maximum_focus_events: int = 6_000,
) -> dict[str, Any]:
    categories = (
        "attention",
        "moe",
        "gemm",
        "norm_rope",
        "sampling",
        "other",
        "hccl",
    )
    rank_intervals: dict[int, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rank_meta: dict[int, dict[str, Any]] = {}
    raw_events: list[tuple[int, int, str, str, float, float]] = []
    focus_bins: dict[int, float] = defaultdict(float)
    focus_width_us = focus_ms * 1_000.0
    raw_count = 0
    for path in sorted(root.rglob("kernel_details.csv"), key=str):
        rank, instance, local_rank = _rank_identity(path)
        rank_meta[rank] = {
            "rank": rank,
            "instance": instance,
            "local_rank": local_rank,
            "source": str(path),
        }
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                start = _float(row, "Start Time(us)")
                duration = _float(row, "Duration(us)")
                end = start + duration
                if duration <= 0 or end <= origin_us or start >= finish_us:
                    continue
                raw_count += 1
                start = max(start, origin_us)
                end = min(end, finish_us)
                name = str(row.get("Name") or row.get("Type") or "unknown")
                kind = str(row.get("Type") or "unknown")
                category = _kernel_category(name, kind)
                rank_intervals[rank][category].append((start, end))
                raw_events.append(
                    (
                        rank,
                        int(_float(row, "Stream ID")),
                        name,
                        category,
                        start,
                        end - start,
                    )
                )
                first = int((start - origin_us) // focus_width_us)
                last = int((max(start, end - 1e-9) - origin_us) // focus_width_us)
                weight = 2.0 if category in {"attention", "hccl"} else 0.25
                for index in range(first, last + 1):
                    left = origin_us + index * focus_width_us
                    right = left + focus_width_us
                    focus_bins[index] += weight * max(
                        0.0, min(end, right) - max(start, left)
                    )
    def mixed_instances(index: int) -> int:
        midpoint_ms = (index + 0.5) * focus_ms
        result = 0
        for by_stage in active_stages.values():
            active = []
            for stage in ("candidate", "rollout"):
                active.append(
                    any(
                        start <= midpoint_ms < start + duration and count > 0
                        for start, duration, count in by_stage.get(stage, ())
                    )
                )
            result += int(all(active))
        return result

    focus_index = max(
        focus_bins,
        key=lambda index: (mixed_instances(index), focus_bins[index]),
        default=0,
    )
    focus_start = origin_us + focus_index * focus_width_us
    focus_end = min(finish_us, focus_start + focus_width_us)
    focus_events = [
        event
        for event in raw_events
        if event[4] + event[5] > focus_start and event[4] < focus_end
    ]
    total_focus_events = len(focus_events)
    if total_focus_events > maximum_focus_events:
        focus_events = sorted(
            focus_events,
            key=lambda item: (
                item[3] in {"attention", "hccl"},
                item[5],
            ),
            reverse=True,
        )[:maximum_focus_events]
    focus_events.sort(key=lambda item: (item[4], item[0], item[1]))
    names = sorted({event[2] for event in focus_events})
    name_index = {name: index for index, name in enumerate(names)}
    category_index = {name: index for index, name in enumerate(categories)}
    ranks = []
    overview_bin_us = 1_000.0
    overview_bins = max(
        1,
        int((finish_us - origin_us + overview_bin_us - 1) // overview_bin_us),
    )
    for rank in sorted(rank_meta):
        by_category = rank_intervals[rank]
        compute = [
            interval
            for category, intervals in by_category.items()
            if category != "hccl"
            for interval in intervals
        ]
        item = dict(rank_meta[rank])
        busy = _bin_intervals(
            compute,
            origin=origin_us,
            bins=overview_bins,
            width=overview_bin_us,
        )
        hccl = _bin_intervals(
            by_category.get("hccl", ()),
            origin=origin_us,
            bins=overview_bins,
            width=overview_bin_us,
        )
        category_occupancy = [
            _bin_intervals(
                by_category.get(category, ()),
                origin=origin_us,
                bins=overview_bins,
                width=overview_bin_us,
            )
            for category in categories
            if category != "hccl"
        ]
        item["overview_bin_ms"] = overview_bin_us / 1_000.0
        item["busy_q"] = [round(value * 100) for value in busy]
        item["hccl_q"] = [round(value * 100) for value in hccl]
        item["dominant_category"] = [
            (
                max(
                    range(len(category_occupancy)),
                    key=lambda category: category_occupancy[category][index],
                )
                if busy[index] > 0
                else -1
            )
            for index in range(overview_bins)
        ]
        ranks.append(item)
    return {
        "origin_unix_us": origin_us,
        "duration_ms": round((finish_us - origin_us) / 1_000.0, 4),
        "raw_event_count": raw_count,
        "categories": list(categories),
        "ranks": ranks,
        "focus": {
            "start_ms": round((focus_start - origin_us) / 1_000.0, 4),
            "duration_ms": round((focus_end - focus_start) / 1_000.0, 4),
            "total_events": total_focus_events,
            "rendered_events": len(focus_events),
            "names": names,
            "events": [
                [
                    rank,
                    stream,
                    name_index[name],
                    category_index[category],
                    round((start - origin_us) / 1_000.0, 4),
                    round(duration, 3),
                ]
                for rank, stream, name, category, start, duration in focus_events
            ],
        },
    }


def _algorithm_gantt(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for event in record.get("stage_events", ()):
                stage = str(event.get("name", "unknown"))
                if stage not in {"candidate", "rollout", "reward", "weight", "resample"}:
                    continue
                duration = float(event.get("duration_us", 0.0))
                if duration <= 0:
                    continue
                records.append(
                    {
                        "job": str(event.get("job_id", record.get("request_id", "unknown"))),
                        "instance": str(event.get("instance_id", "unknown")),
                        "block": int(event.get("block_id", event.get("step", 0))),
                        "stage": stage,
                        "start_us": float(event.get("start_unix_us", 0.0)),
                        "duration_us": duration,
                        "sequences": int(event.get("sequence_count", 0)),
                        "prefix_tokens": int(event.get("prefix_tokens", 0)),
                    }
                )
    if not records:
        return {"events": [], "jobs": [], "duration_ms": 0.0}
    origin = min(item["start_us"] for item in records)
    jobs = sorted({item["job"] for item in records})
    job_index = {job: index for index, job in enumerate(jobs)}
    events = [
        [
            job_index[item["job"]],
            item["instance"],
            item["block"],
            item["stage"],
            round((item["start_us"] - origin) / 1_000.0, 4),
            round(item["duration_us"] / 1_000.0, 4),
            item["sequences"],
            item["prefix_tokens"],
        ]
        for item in records
    ]
    return {
        "origin_unix_us": origin,
        "duration_ms": round(
            max(item["start_us"] + item["duration_us"] for item in records)
            / 1_000.0
            - origin / 1_000.0,
            4,
        ),
        "jobs": jobs,
        "events": sorted(events, key=lambda item: (item[4], item[0])),
    }


def _phase_summary(
    kernel: dict[str, Any],
    active_stages: dict[str, dict[str, list[list[float]]]],
) -> list[dict[str, Any]]:
    """Join 1 ms device occupancy with exact candidate/rollout intervals."""
    rows = []
    for rank in kernel["ranks"]:
        bins = len(rank["busy_q"])
        bin_ms = float(rank["overview_bin_ms"])
        by_stage = active_stages.get(f"profile-{rank['instance']}", {})
        counts = {"candidate": [0] * bins, "rollout": [0] * bins}
        for stage, values in counts.items():
            for start, duration, count in by_stage.get(stage, ()):
                first = max(0, int(start // bin_ms))
                last = min(bins, int((start + duration + bin_ms - 1e-9) // bin_ms))
                for index in range(first, last):
                    values[index] = max(values[index], int(count))
        phases: dict[str, list[int]] = defaultdict(list)
        for index in range(bins):
            candidate = counts["candidate"][index] > 0
            rollout = counts["rollout"][index] > 0
            phase = (
                "mixed"
                if candidate and rollout
                else "candidate_only"
                if candidate
                else "rollout_only"
                if rollout
                else "unattributed"
            )
            phases[phase].append(index)
        summaries = {}
        for phase in ("candidate_only", "mixed", "rollout_only", "unattributed"):
            indices = phases.get(phase, [])
            summaries[phase] = {
                "duration_ms": round(len(indices) * bin_ms, 3),
                "busy_ratio": round(
                    sum(rank["busy_q"][index] for index in indices)
                    / (100 * len(indices)),
                    5,
                )
                if indices
                else None,
                "hccl_ratio": round(
                    sum(rank["hccl_q"][index] for index in indices)
                    / (100 * len(indices)),
                    5,
                )
                if indices
                else None,
            }
        rows.append(
            {
                "rank": rank["rank"],
                "instance": rank["instance"],
                "phases": summaries,
            }
        )
    return rows


def _service_detail(root: Path) -> dict[str, Any]:
    records = []
    for instance, path in enumerate(sorted(root.rglob("batch.csv"), key=str)):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row.get("name") != "batchFrameworkProcessing":
                    continue
                try:
                    resources = ast.literal_eval(row.get("res_list", "[]"))
                except (SyntaxError, ValueError):
                    resources = []
                types = [int(item.get("type", -1)) for item in resources]
                records.append(
                    {
                        "instance": instance,
                        "start_ms": _float(row, "start_time(ms)"),
                        "duration_ms": _float(row, "during_time(ms)"),
                        "batch_size": int(_float(row, "batch_size")),
                        "prefill": types.count(0),
                        "decode": types.count(1),
                        "scheduled_tokens": sum(
                            int(item.get("num_scheduled_tokens", 0))
                            for item in resources
                        ),
                    }
                )
    if not records:
        return {"events": [], "duration_ms": 0.0}
    origin = min(item["start_ms"] for item in records)
    events = [
        [
            item["instance"],
            round(item["start_ms"] - origin, 4),
            round(item["duration_ms"], 4),
            item["batch_size"],
            item["prefill"],
            item["decode"],
            item["scheduled_tokens"],
        ]
        for item in records
    ]
    return {
        "origin_unix_ms": origin,
        "duration_ms": round(
            max(item["start_ms"] + item["duration_ms"] for item in records) - origin,
            4,
        ),
        "events": sorted(events, key=lambda item: (item[1], item[0])),
    }


def _bin_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    origin: float,
    bins: int,
    width: float,
) -> list[float]:
    values = [0.0] * bins
    for start, end in _merge(intervals):
        first = max(0, int((start - origin) // width))
        last = min(bins - 1, int((max(start, end - 1e-9) - origin) // width))
        for index in range(first, last + 1):
            left = origin + index * width
            right = left + width
            values[index] += max(0.0, min(end, right) - max(start, left))
    return [min(1.0, value / width) for value in values]


def _kernel_timeline(root: Path, *, bin_ms: float) -> dict[str, Any]:
    ranks = []
    for rank, path in enumerate(sorted(root.rglob("kernel_details.csv"))):
        compute: list[tuple[float, float]] = []
        communication: list[tuple[float, float]] = []
        names: dict[str, float] = defaultdict(float)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                start = _float(row, "Start Time(us)")
                duration = _float(row, "Duration(us)")
                if duration <= 0:
                    continue
                name = str(row.get("Name") or row.get("Type") or "unknown")
                names[name] += duration
                interval = (start, start + duration)
                lowered = f"{name} {row.get('Type', '')}".lower()
                if "hcom" in lowered or "hccl" in lowered:
                    communication.append(interval)
                else:
                    compute.append(interval)
        all_intervals = compute + communication
        if not all_intervals:
            continue
        origin = min(start for start, _ in all_intervals)
        end = max(end for _, end in all_intervals)
        width = bin_ms * 1_000.0
        bins = max(1, int((end - origin + width - 1) // width))
        ranks.append(
            {
                "rank": rank,
                "source": str(path),
                "bin_ms": bin_ms,
                "compute": _bin_intervals(
                    compute, origin=origin, bins=bins, width=width
                ),
                "communication": _bin_intervals(
                    communication, origin=origin, bins=bins, width=width
                ),
                "top_kernels_us": dict(
                    sorted(names.items(), key=lambda item: item[1], reverse=True)[:12]
                ),
            }
        )
    return {"ranks": ranks, "alignment": "first kernel per rank"}


def _algorithm_window(root: Path, *, bin_ms: float) -> dict[str, Any]:
    benchmark = json.loads((root / "benchmark.json").read_text())
    window = benchmark.get("profile", {}).get("window", {})
    started_us = float(window.get("started_at", 0.0)) * 1e6
    finished_us = float(window.get("finished_at", 0.0)) * 1e6
    width = bin_ms * 1_000.0
    bins = max(1, int((finished_us - started_us + width - 1) // width))
    by_instance: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0] * bins)
    )
    for path in sorted((root / "algorithm-traces").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for event in record.get("stage_events", ()):
                stage = str(event.get("name", "unknown"))
                if stage not in {"candidate", "rollout", "reward", "weight"}:
                    continue
                start = float(event.get("start_unix_us", 0.0))
                end = start + float(event.get("duration_us", 0.0))
                if end <= started_us or start >= finished_us:
                    continue
                instance = str(event.get("instance_id", "unknown"))
                first = max(0, int((max(start, started_us) - started_us) // width))
                last = min(
                    bins - 1,
                    int((max(start, min(end, finished_us) - 1e-9) - started_us) // width),
                )
                for index in range(first, last + 1):
                    by_instance[instance][stage][index] += 1
    return {
        "bin_ms": bin_ms,
        "instances": {
            instance: dict(stages) for instance, stages in by_instance.items()
        },
        "window_seconds": max(0.0, (finished_us - started_us) / 1e6),
    }


def _service_batches(root: Path, *, bin_seconds: float) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for instance, path in enumerate(sorted(root.rglob("batch.csv"))):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row.get("name") != "batchFrameworkProcessing":
                    continue
                try:
                    resources = ast.literal_eval(row.get("res_list", "[]"))
                except (SyntaxError, ValueError):
                    resources = []
                types = [int(item.get("type", -1)) for item in resources]
                records.append(
                    {
                        "instance": instance,
                        "start_ms": _float(row, "start_time(ms)"),
                        "duration_ms": _float(row, "during_time(ms)"),
                        "batch_size": _float(row, "batch_size"),
                        "prefill": types.count(0),
                        "decode": types.count(1),
                        "scheduled_tokens": sum(
                            int(item.get("num_scheduled_tokens", 0))
                            for item in resources
                        ),
                    }
                )
    if not records:
        return {"bins": [], "summary": {}}
    origin = min(item["start_ms"] for item in records)
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        bucket = int((record["start_ms"] - origin) / (bin_seconds * 1_000))
        buckets[(record["instance"], bucket)].append(record)
    binned = []
    for (instance, bucket), values in sorted(buckets.items()):
        batch_sizes = [item["batch_size"] for item in values]
        binned.append(
            {
                "instance": instance,
                "second": round(bucket * bin_seconds, 3),
                "batch_mean": statistics.fmean(batch_sizes),
                "batch_p95": _percentile(batch_sizes, 0.95),
                "batch_max": max(batch_sizes),
                "prefill_mean": statistics.fmean(item["prefill"] for item in values),
                "decode_mean": statistics.fmean(item["decode"] for item in values),
                "mixed_ratio": sum(
                    item["prefill"] > 0 and item["decode"] > 0 for item in values
                )
                / len(values),
                "scheduled_tokens_mean": statistics.fmean(
                    item["scheduled_tokens"] for item in values
                ),
                "forward_ms_mean": statistics.fmean(
                    item["duration_ms"] for item in values
                ),
            }
        )
    return {
        "bin_seconds": bin_seconds,
        "bins": binned,
        "summary": {
            "rows": len(records),
            "batch_p50": _percentile(
                [item["batch_size"] for item in records], 0.50
            ),
            "batch_p95": _percentile(
                [item["batch_size"] for item in records], 0.95
            ),
            "batch_p99": _percentile(
                [item["batch_size"] for item in records], 0.99
            ),
            "mixed_ratio": sum(
                item["prefill"] > 0 and item["decode"] > 0 for item in records
            )
            / len(records),
        },
    }


def export(
    torch_root: Path,
    service_root: Path,
    baseline_algorithm_root: Path | None = None,
) -> dict[str, Any]:
    benchmark = json.loads((torch_root / "benchmark.json").read_text())
    window = benchmark.get("profile", {}).get("window", {})
    started_us = float(window.get("started_at", 0.0)) * 1e6
    finished_us = float(window.get("finished_at", 0.0)) * 1e6
    active_stages = _active_stage_segments(
        torch_root,
        origin_us=started_us,
        finish_us=finished_us,
    )
    kernel_detail = _kernel_detail(
        torch_root,
        origin_us=started_us,
        finish_us=finished_us,
        active_stages=active_stages,
    )
    payload = {
        "schema_version": 1,
        "torch_source": str(torch_root),
        "service_source": str(service_root),
        "benchmark": analyze_benchmark(torch_root / "benchmark.json"),
        "algorithm": analyze_algorithm_trace(torch_root / "algorithm-traces"),
        "ascend": analyze_ascend_profile(torch_root),
        "kernel_timeline": _kernel_timeline(torch_root, bin_ms=250.0),
        "algorithm_window": _algorithm_window(torch_root, bin_ms=250.0),
        "service_batches": _service_batches(service_root, bin_seconds=1.0),
        "detailed_timeline": {
            "kernel": kernel_detail,
            "active_stages": active_stages,
            "phase_summary": _phase_summary(kernel_detail, active_stages),
            "service": _service_detail(service_root),
        },
    }
    if baseline_algorithm_root is not None:
        payload["detailed_timeline"]["algorithm_gantt"] = _algorithm_gantt(
            baseline_algorithm_root
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-profile", type=Path, required=True)
    parser.add_argument("--service-profile", type=Path, required=True)
    parser.add_argument("--baseline-algorithm", type=Path)
    parser.add_argument("--torch-source-label")
    parser.add_argument("--service-source-label")
    parser.add_argument("--baseline-source-label")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = export(
        args.torch_profile,
        args.service_profile,
        args.baseline_algorithm,
    )
    if args.torch_source_label:
        payload["torch_source"] = args.torch_source_label
    if args.service_source_label:
        payload["service_source"] = args.service_source_label
    if args.baseline_source_label:
        payload["baseline_algorithm_source"] = args.baseline_source_label
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
