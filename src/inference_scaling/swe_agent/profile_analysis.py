"""Summarize Conditional IS stages and exported Ascend profiler artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


_LEAF_STAGES = ("candidate", "rollout", "scoring", "reward", "weight", "resample")


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _merged(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in _merged(intervals))


def _intersection_duration(
    left: Iterable[tuple[float, float]], right: Iterable[tuple[float, float]]
) -> float:
    first = _merged(left)
    second = _merged(right)
    i = j = 0
    total = 0.0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        total += max(0.0, end - start)
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze_algorithm_trace(path: Path) -> dict[str, Any]:
    paths = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    records = [
        json.loads(line)
        for source in paths
        for line in source.read_text().splitlines()
        if line.strip()
    ]
    stage_values: dict[str, list[float]] = defaultdict(list)
    algorithm_seconds = 0.0
    block_gap_seconds = 0.0
    for record in records:
        diagnostics = record.get("diagnostics") or {}
        algorithm_seconds += float(diagnostics.get("algorithm_seconds", 0.0))
        by_step: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for event in record.get("stage_events", ()):
            name = str(event.get("name", "unknown"))
            seconds = float(event.get("duration_us", 0.0)) / 1e6
            stage_values[name].append(seconds)
            by_step[int(event.get("step", -1))][name] += seconds
        for stages in by_step.values():
            leaf = sum(stages.get(name, 0.0) for name in _LEAF_STAGES)
            block_gap_seconds += max(0.0, stages.get("block", leaf) - leaf)
    stage_seconds = {name: sum(values) for name, values in stage_values.items()}
    stage_share = {
        name: value / algorithm_seconds if algorithm_seconds else 0.0
        for name, value in stage_seconds.items()
        if name != "block"
    }
    return {
        "jobs": len(records),
        "trace_files": [str(source) for source in paths],
        "algorithm_seconds": algorithm_seconds,
        "stage_seconds": stage_seconds,
        "stage_share": stage_share,
        "stage_p95_seconds": {
            name: _percentile(values, 0.95) for name, values in stage_values.items()
        },
        "block_gap_seconds": block_gap_seconds,
        "block_gap_share": (
            block_gap_seconds / algorithm_seconds if algorithm_seconds else 0.0
        ),
    }


def analyze_benchmark(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    counters = payload.get("backend_delta") or {}
    prefill = float(counters.get("prefill_tokens", 0.0))
    saved = float(counters.get("shared_prefill_tokens_saved", 0.0))
    by_endpoint: dict[str, list[float]] = defaultdict(list)
    for measurement in payload.get("measurements", ()):
        if measurement.get("success"):
            by_endpoint[str(measurement.get("endpoint", "unknown"))].append(
                float(measurement.get("seconds", 0.0))
            )
    endpoint_mean = {
        endpoint: statistics.fmean(values)
        for endpoint, values in by_endpoint.items()
        if values
    }
    means = list(endpoint_mean.values())
    endpoint_skew = (
        (max(means) - min(means)) / min(means)
        if len(means) > 1 and min(means) > 0
        else 0.0
    )
    return {
        "requests": int(payload.get("requests", 0)),
        "success_rate": float(payload.get("success_rate", 0.0)),
        "jobs_per_second": float(payload.get("jobs_per_second", 0.0)),
        "latency_seconds": payload.get("latency_seconds", {}),
        "backend_delta": counters,
        "apc_token_hit_ratio": saved / (saved + prefill) if saved + prefill else 0.0,
        "endpoint_mean_seconds": endpoint_mean,
        "endpoint_mean_skew": endpoint_skew,
    }


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0").strip())
    except (AttributeError, ValueError):
        return 0.0


def _kernel_summary(path: Path) -> dict[str, Any]:
    intervals: list[tuple[float, float]] = []
    compute: list[tuple[float, float]] = []
    communication: list[tuple[float, float]] = []
    by_type: dict[str, float] = defaultdict(float)
    by_core: dict[str, float] = defaultdict(float)
    seen: set[tuple[float, float, str]] = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            start = _float(row, "Start Time(us)")
            duration = _float(row, "Duration(us)")
            kind = str(row.get("Type") or row.get("Name") or "unknown")
            key = (start, duration, kind)
            if duration <= 0 or key in seen:
                continue
            seen.add(key)
            interval = (start, start + duration)
            intervals.append(interval)
            by_type[kind] += duration
            by_core[str(row.get("Accelerator Core") or "unknown")] += duration
            lowered = f"{kind} {row.get('Name', '')}".lower()
            if "hcom" in lowered or "hccl" in lowered:
                communication.append(interval)
            else:
                compute.append(interval)
    busy = _duration(intervals)
    communication_time = _duration(communication)
    communication_overlap = _intersection_duration(communication, compute)
    span = (
        max(end for _, end in intervals) - min(start for start, _ in intervals)
        if intervals
        else 0.0
    )
    return {
        "path": str(path),
        "profile_span_us": span,
        "device_busy_us": busy,
        "device_busy_ratio": busy / span if span else 0.0,
        "communication_us": communication_time,
        "communication_compute_overlap_us": communication_overlap,
        "exposed_communication_us": max(
            0.0, communication_time - communication_overlap
        ),
        "exposed_communication_ratio": (
            max(0.0, communication_time - communication_overlap) / communication_time
            if communication_time
            else 0.0
        ),
        "top_kernel_types_us": dict(
            sorted(by_type.items(), key=lambda item: item[1], reverse=True)[:20]
        ),
        "accelerator_core_us": dict(by_core),
    }


def _operator_summary(path: Path) -> dict[str, Any]:
    host: dict[str, float] = defaultdict(float)
    device: dict[str, float] = defaultdict(float)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            name = str(row.get("Name") or "unknown")
            host[name] += _float(row, "Host Self Duration(us)")
            device[name] += _float(row, "Device Self Duration(us)")
    return {
        "path": str(path),
        "top_host_self_us": dict(
            sorted(host.items(), key=lambda item: item[1], reverse=True)[:20]
        ),
        "top_device_self_us": dict(
            sorted(device.items(), key=lambda item: item[1], reverse=True)[:20]
        ),
    }


def _communication_entries(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        timing = value.get("Communication Time Info")
        if isinstance(timing, dict):
            yield timing
        for nested in value.values():
            yield from _communication_entries(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _communication_entries(nested)


def _communication_summary(path: Path) -> dict[str, Any]:
    entries = list(_communication_entries(json.loads(path.read_text())))
    totals = {
        key: sum(float(entry.get(key, 0.0)) for entry in entries)
        for key in (
            "Elapse Time(ms)",
            "Transit Time(ms)",
            "Wait Time(ms)",
            "Synchronization Time(ms)",
            "Idle Time(ms)",
        )
    }
    return {"path": str(path), "operations": len(entries), "totals": totals}


def analyze_ascend_profile(path: Path) -> dict[str, Any]:
    kernels = [_kernel_summary(item) for item in path.rglob("kernel_details.csv")]
    operators = [_operator_summary(item) for item in path.rglob("operator_details.csv")]
    communication = [
        _communication_summary(item) for item in path.rglob("communication.json")
    ]
    return {
        "rank_count": len(kernels),
        "ranks": kernels,
        "operator_ranks": operators,
        "communication": communication,
        "device_busy_ratio": {
            "minimum": min(
                (item["device_busy_ratio"] for item in kernels), default=0.0
            ),
            "median": (
                statistics.median(item["device_busy_ratio"] for item in kernels)
                if kernels
                else 0.0
            ),
            "maximum": max(
                (item["device_busy_ratio"] for item in kernels), default=0.0
            ),
        },
        "exposed_communication_ratio": (
            sum(item["exposed_communication_us"] for item in kernels)
            / sum(item["communication_us"] for item in kernels)
            if sum(item["communication_us"] for item in kernels)
            else 0.0
        ),
    }


def _numeric_csv_summary(path: Path) -> dict[str, Any]:
    columns: dict[str, list[float]] = defaultdict(list)
    rows = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            for name, raw in row.items():
                try:
                    columns[str(name)].append(float(str(raw).strip()))
                except (TypeError, ValueError):
                    continue
    return {
        "path": str(path),
        "rows": rows,
        "numeric_columns": {
            name: {
                "count": len(values),
                "minimum": min(values),
                "mean": statistics.fmean(values),
                "p50": _percentile(values, 0.50),
                "p75": _percentile(values, 0.75),
                "p90": _percentile(values, 0.90),
                "p95": _percentile(values, 0.95),
                "p99": _percentile(values, 0.99),
                "maximum": max(values),
            }
            for name, values in columns.items()
            if values
        },
    }


def analyze_service_profile(path: Path) -> dict[str, Any]:
    names = {
        "request.csv",
        "request_summary.csv",
        "kvcache.csv",
        "batch.csv",
        "batch_summary.csv",
        "service_summary.csv",
    }
    files = [
        _numeric_csv_summary(item)
        for item in sorted(path.rglob("*.csv"))
        if item.name in names
    ]
    batch_quantiles: dict[str, Any] = {}
    for item in files:
        if Path(item["path"]).name != "batch.csv":
            continue
        for column, summary in item["numeric_columns"].items():
            lowered = column.lower().replace(" ", "_")
            if any(
                marker in lowered
                for marker in (
                    "batch_size",
                    "num_request",
                    "num_seq",
                    "scheduled_token",
                    "batch_token",
                    "running_request",
                    "waiting_request",
                )
            ):
                batch_quantiles[column] = summary
    return {
        "files": files,
        "batch_shape_quantiles": batch_quantiles,
        "has_batch_scheduler_evidence": bool(batch_quantiles),
    }


def recommendations(
    benchmark: dict[str, Any], algorithm: dict[str, Any], ascend: dict[str, Any]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    reward_share = sum(
        float(algorithm.get("stage_share", {}).get(name, 0.0))
        for name in ("reward", "scoring")
    )
    if reward_share > 0.15:
        results.append(
            {
                "trigger": "reward_or_scoring_above_15_percent",
                "value": reward_share,
                "action": "optimize fused generation statistics and prefix reuse",
            }
        )
    gap = float(algorithm.get("block_gap_share", 0.0))
    device_busy = float(ascend.get("device_busy_ratio", {}).get("median", 0.0))
    if gap > 0.10 or (ascend.get("rank_count", 0) and device_busy < 0.90):
        results.append(
            {
                "trigger": "stage_gap_above_10_percent_or_device_underfilled",
                "value": max(gap, 1.0 - device_busy),
                "action": "implement cross-job candidate/rollout pipeline",
            }
        )
    if float(benchmark.get("apc_token_hit_ratio", 0.0)) < 0.50:
        results.append(
            {
                "trigger": "apc_token_hit_ratio_below_50_percent",
                "value": benchmark.get("apc_token_hit_ratio", 0.0),
                "action": "add session affinity and inspect prefix identity",
            }
        )
    if float(benchmark.get("endpoint_mean_skew", 0.0)) > 0.10:
        results.append(
            {
                "trigger": "instance_latency_skew_above_10_percent",
                "value": benchmark["endpoint_mean_skew"],
                "action": "route by estimated remaining CIS work",
            }
        )
    if float(ascend.get("exposed_communication_ratio", 0.0)) > 0.15:
        results.append(
            {
                "trigger": "exposed_hccl_above_15_percent",
                "value": ascend["exposed_communication_ratio"],
                "action": "overlap communication with independent CIS stage work",
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--algorithm-trace", required=True)
    parser.add_argument("--npu-profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    benchmark = analyze_benchmark(Path(args.benchmark))
    algorithm = analyze_algorithm_trace(Path(args.algorithm_trace))
    ascend = analyze_ascend_profile(Path(args.npu_profile))
    service = analyze_service_profile(Path(args.npu_profile))
    result = {
        "schema_version": 1,
        "benchmark": benchmark,
        "algorithm": algorithm,
        "ascend": ascend,
        "service": service,
        "recommendations": recommendations(benchmark, algorithm, ascend),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
