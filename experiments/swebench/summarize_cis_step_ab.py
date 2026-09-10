"""Summarize exact Conditional IS step-scheduling A/B artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUEST_ID = re.compile(
    r"^(?P<job>.*):step:(?P<step>\d+):candidate:(?P<candidate>\d+)"
    r"(?::rollout:(?P<rollout>\d+))?"
)


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _canonical_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message
    tool_calls = []
    for call in message.get("tool_calls") or ():
        function = call.get("function") or {}
        tool_calls.append(
            {"name": function.get("name"), "arguments": function.get("arguments")}
        )
    return {
        "role": message.get("role"),
        "content": message.get("content"),
        "tool_calls": tool_calls,
    }


def _step_concurrency(intervals: list[tuple[float, float]]) -> dict[str, float]:
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = peak = 0
    area = 0.0
    previous = events[0][0] if events else 0.0
    for at, delta in events:
        area += active * max(0.0, at - previous)
        active += delta
        peak = max(peak, active)
        previous = at
    span = events[-1][0] - events[0][0] if len(events) > 1 else 0.0
    return {"mean": area / span if span else 0.0, "peak": float(peak)}


def summarize(root: Path) -> dict[str, Any]:
    benchmark = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
    algorithms = [
        record
        for record in _jsonl(sorted((root / "algorithm-traces").glob("*.jsonl")))
        if ":public:" in str(record.get("request_id", ""))
    ]
    lifecycle = [
        record
        for record in _jsonl(sorted((root / "request-traces").glob("*.jsonl")))
        if ":public:" in str(record.get("request_id", ""))
    ]

    block_ms = []
    end_to_end_step_ms = []
    admission_wait_ms = []
    intervals = []
    output_hash = hashlib.sha256()
    for record in sorted(
        algorithms,
        key=lambda item: json.dumps(
            item.get("prompt_token_ids"), separators=(",", ":")
        ),
    ):
        output_hash.update(
            json.dumps(record.get("prompt_token_ids"), separators=(",", ":")).encode()
        )
        output_hash.update(
            json.dumps(
                _canonical_message(record.get("message")),
                sort_keys=True,
                separators=(",", ":"),
            )
            .encode()
        )
        stages_by_step: dict[int, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for event in record.get("stage_events", ()):
            name = event.get("name")
            duration_ms = float(event.get("duration_us", 0.0)) / 1_000.0
            step = int(event.get("step", -1))
            stages_by_step[step][str(name)] += duration_ms
            if name == "block":
                block_ms.append(duration_ms)
                start = float(event.get("start_unix_us", 0.0)) / 1_000.0
                intervals.append((start, start + duration_ms))
            elif name == "step_admission_wait":
                admission_wait_ms.append(duration_ms)
        end_to_end_step_ms.extend(
            stages["block"] + stages.get("step_admission_wait", 0.0)
            for stages in stages_by_step.values()
            if "block" in stages
        )

    finished_rollouts: dict[tuple[str, int], list[float]] = defaultdict(list)
    queue_ms = []
    priorities = []
    for event in lifecycle:
        if event.get("event") != "finished":
            continue
        match = REQUEST_ID.match(str(event.get("request_id", "")))
        if match is None:
            continue
        queue = event.get("queue_us")
        if queue is not None:
            queue_ms.append(float(queue) / 1_000.0)
        if event.get("priority") is not None:
            priorities.append(int(event["priority"]))
        if match.group("rollout") is not None:
            finished_rollouts[(match.group("job"), int(match.group("step")))].append(
                float(event.get("event_unix_us", 0.0)) / 1_000.0
            )
    barrier_tail_ms = [
        max(values) - min(values)
        for values in finished_rollouts.values()
        if len(values) > 1
    ]
    backend = benchmark.get("backend_delta", {})
    wall_seconds = float(benchmark.get("wall_seconds", 0.0) or 0.0)
    prefill_tokens = float(backend.get("prefill_tokens", 0.0) or 0.0)
    cached_tokens = float(
        backend.get("shared_prefill_tokens_saved", 0.0) or 0.0
    )
    generated_tokens = float(backend.get("generated_tokens", 0.0) or 0.0)
    engine_requests = float(backend.get("engine_requests", 0.0) or 0.0)
    return {
        "root": str(root),
        "wall_seconds": wall_seconds,
        "success_rate": benchmark.get("success_rate"),
        "jobs_per_second": benchmark.get("jobs_per_second"),
        "job_latency_seconds": benchmark.get("latency_seconds"),
        "forward_token_slots_per_second": benchmark.get("throughput", {}).get(
            "generation_forward_token_slots_per_second"
        ),
        "preemptions": backend.get("num_preemptions"),
        "engine_requests": engine_requests,
        "engine_requests_per_second": (
            engine_requests / wall_seconds if wall_seconds else None
        ),
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": (
            generated_tokens / wall_seconds if wall_seconds else None
        ),
        "prefill_tokens": prefill_tokens,
        "shared_prefill_tokens_saved": cached_tokens,
        "apc_token_hit_ratio": (
            cached_tokens / (cached_tokens + prefill_tokens)
            if cached_tokens + prefill_tokens
            else None
        ),
        "maximum_in_flight_requests": backend.get("maximum_in_flight_requests"),
        "steps": len(block_ms),
        "step_completion_ms": {
            "mean": statistics.fmean(block_ms) if block_ms else None,
            "p50": _percentile(block_ms, 0.50),
            "p95": _percentile(block_ms, 0.95),
        },
        "step_end_to_end_ms": {
            "mean": statistics.fmean(end_to_end_step_ms)
            if end_to_end_step_ms
            else None,
            "p50": _percentile(end_to_end_step_ms, 0.50),
            "p95": _percentile(end_to_end_step_ms, 0.95),
        },
        "step_admission_wait_ms": {
            "mean": statistics.fmean(admission_wait_ms)
            if admission_wait_ms
            else 0.0,
            "p95": _percentile(admission_wait_ms, 0.95) or 0.0,
        },
        "rollout_barrier_tail_ms": {
            "mean": statistics.fmean(barrier_tail_ms) if barrier_tail_ms else None,
            "p95": _percentile(barrier_tail_ms, 0.95),
        },
        "engine_queue_ms": {
            "mean": statistics.fmean(queue_ms) if queue_ms else None,
            "p95": _percentile(queue_ms, 0.95),
        },
        "active_steps": _step_concurrency(intervals),
        "priority_range": [min(priorities), max(priorities)] if priorities else None,
        "output_hash": output_hash.hexdigest(),
        "overrides": benchmark.get("overrides"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {root.name: summarize(root) for root in args.roots}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
