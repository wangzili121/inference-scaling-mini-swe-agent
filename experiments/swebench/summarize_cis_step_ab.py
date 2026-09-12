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
FORK_CAPTURE = re.compile(
    r"CIS_KV_FORK capture parent=(?P<parent>\S+) "
    r"blocks=(?P<blocks>\d+) children=(?P<children>\d+)"
    r"(?: leased=(?P<leased>\d+))?(?: scope=(?P<scope>\S+))?"
    r"(?: held_unique=(?P<held_unique>\d+) budget=(?P<budget>\d+))?"
)
FORK_HIT = re.compile(
    r"CIS_KV_FORK hit child=(?P<child>\S+) parent=(?P<parent>\S+) "
    r"tokens=(?P<tokens>\d+)"
)
FORK_MISS = re.compile(
    r"CIS_KV_FORK miss child=(?P<child>\S+) parent=(?P<parent>\S+) "
    r"reason=(?P<reason>\S+)"
)
FORK_LEASE_RELEASE = re.compile(
    r"CIS_KV_FORK lease_release parent=(?P<parent>\S+) blocks=(?P<blocks>\d+)"
)
FORK_LEASE_EXPIRE = re.compile(r"CIS_KV_FORK lease_expire parent=(?P<parent>\S+)")
FORK_TERMINAL_PARENT = re.compile(
    r"CIS_KV_FORK terminal_parent parent=(?P<parent>\S+)"
)
BRANCH_DEMOTE = re.compile(
    r"CIS_KV_BRANCH demote request=(?P<request>\S+) blocks=(?P<blocks>\d+)"
)
BRANCH_RECORD = re.compile(
    r"CIS_KV_BRANCH record request=(?P<request>\S+) "
    r"kind=(?P<kind>\S+) blocks=(?P<blocks>\d+)"
)
RESAMPLE_GC = re.compile(
    r"CIS_KV_RESAMPLE_GC selected=(?P<selected>\S+) "
    r"candidates=(?P<candidates>\d+) rollouts=(?P<rollouts>\d+) "
    r"recorded=(?P<recorded>\d+) evicted=(?P<evicted>\d+) "
    r"stale=(?P<stale>\d+) active=(?P<active>\d+) "
    r"protected=(?P<protected>\d+)"
)
ENGINE_FORK_PARK = re.compile(
    r"CIS_ENGINE_FORK parked child=(?P<child>\S+) parent=(?P<parent>\S+)"
    r"(?: compact=(?P<compact>[01]) prompt_tokens_elided=(?P<elided>\d+))?"
)
ENGINE_FORK_ACTIVATE = re.compile(
    r"CIS_ENGINE_FORK activated children=(?P<children>\d+) "
    r"parent=(?P<parent>\S+) candidate_tokens=(?P<tokens>\d+)"
)
ENGINE_FORK_CANCEL = re.compile(
    r"CIS_ENGINE_FORK cancelled children=(?P<children>\d+) "
    r"parent=(?P<parent>\S+) status=(?P<status>\S+)"
)
ENGINE_FORK_GROUP_RELEASE = re.compile(
    r"CIS_ENGINE_FORK group_release group=(?P<group>\S+) "
    r"parents=(?P<parents>\d+) remaining=(?P<remaining>\d+) "
    r"threshold=(?P<threshold>\d+)"
)
ENGINE_FORK_ADAPTIVE_RELEASE = re.compile(
    r"CIS_ENGINE_FORK adaptive_release group=(?P<group>\S+) "
    r"parents=(?P<parents>\d+) children=(?P<children>\d+) "
    r"runnable=(?P<runnable>\d+) target=(?P<target>\d+)"
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
    measurements = [
        item
        for item in benchmark.get("measurements", ())
        if item.get("started_at") is not None and item.get("finished_at") is not None
    ]
    # Older artifacts do not persist the common release timestamp. The earliest
    # worker start is within milliseconds of release and is a much fairer origin
    # than per-worker start time when comparing different client worker counts.
    released_at = benchmark.get("released_at")
    burst_origin = (
        float(released_at)
        if released_at is not None
        else min((float(item["started_at"]) for item in measurements), default=None)
    )
    burst_latency_seconds = (
        [float(item["finished_at"]) - burst_origin for item in measurements]
        if burst_origin is not None
        else []
    )
    run_namespace = str(benchmark.get("run_namespace", ""))
    algorithms = [
        record
        for record in _jsonl(sorted((root / "algorithm-traces").glob("*.jsonl")))
        if str(record.get("request_id", "")).startswith(run_namespace)
    ]
    lifecycle = [
        record
        for record in _jsonl(sorted((root / "request-traces").glob("*.jsonl")))
        if str(record.get("request_id", "")).startswith(run_namespace)
    ]

    block_ms = []
    end_to_end_step_ms = []
    admission_wait_ms = []
    candidate_rollout_overlap_ms = []
    streamed_rollout_submission_batches = []
    subtree_rollout_submission_batches = []
    subtree_peak_active_batches = []
    fused_path_ms = []
    fused_repeated_candidate_tokens = []
    engine_fork_ms = []
    kv_gc_ms = []
    kv_gc_evicted_blocks = []
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
            elif name == "rollout" and event.get("candidate_overlap"):
                candidate_rollout_overlap_ms.append(
                    float(event.get("overlap_seconds", 0.0)) * 1_000.0
                )
                streamed_rollout_submission_batches.append(
                    float(event.get("submission_batches", 0.0))
                )
            elif name == "rollout" and event.get("subtree_max_active_batches"):
                subtree_rollout_submission_batches.append(
                    float(event.get("submission_batches", 0.0))
                )
                subtree_peak_active_batches.append(
                    float(event.get("subtree_peak_active_batches", 0.0))
                )
            elif name == "candidate_rollout_fused":
                fused_path_ms.append(duration_ms)
                fused_repeated_candidate_tokens.append(
                    float(event.get("repeated_candidate_tokens", 0.0))
                )
            elif name == "candidate_rollout_engine_fork":
                engine_fork_ms.append(duration_ms)
            elif name == "resample" and event.get("kv_gc_seconds") is not None:
                kv_gc_ms.append(float(event.get("kv_gc_seconds", 0.0)) * 1_000.0)
                kv_gc_evicted_blocks.append(
                    float(event.get("kv_gc_evicted_blocks", 0.0))
                )
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
    runtime_log = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((root / "logs").glob("*.log"))
    )
    measured_runtime_log = "\n".join(
        line for line in runtime_log.splitlines() if run_namespace in line
    )
    engine_fork_parks = list(ENGINE_FORK_PARK.finditer(measured_runtime_log))
    engine_fork_activations = list(
        ENGINE_FORK_ACTIVATE.finditer(measured_runtime_log)
    )
    engine_fork_cancellations = list(
        ENGINE_FORK_CANCEL.finditer(measured_runtime_log)
    )
    activated_engine_fork_children = sum(
        int(match.group("children")) for match in engine_fork_activations
    )
    cancelled_engine_fork_children = sum(
        int(match.group("children")) for match in engine_fork_cancellations
    )
    engine_fork_group_releases = list(
        ENGINE_FORK_GROUP_RELEASE.finditer(measured_runtime_log)
    )
    first_group_release: dict[str, re.Match[str]] = {}
    for match in engine_fork_group_releases:
        first_group_release.setdefault(match.group("group"), match)
    engine_fork_adaptive_releases = list(
        ENGINE_FORK_ADAPTIVE_RELEASE.finditer(measured_runtime_log)
    )
    fork_capture_by_parent = {
        match.group("parent"): match
        for match in FORK_CAPTURE.finditer(measured_runtime_log)
    }
    fork_hit_by_child = {
        match.group("child"): match
        for match in FORK_HIT.finditer(measured_runtime_log)
    }
    fork_miss_by_child: dict[str, set[str]] = defaultdict(set)
    for match in FORK_MISS.finditer(measured_runtime_log):
        fork_miss_by_child[match.group("child")].add(match.group("reason"))
    for child in fork_hit_by_child:
        fork_miss_by_child.pop(child, None)
    fork_captures = len(fork_capture_by_parent)
    fork_hits = len(fork_hit_by_child)
    fork_misses = len(fork_miss_by_child)
    fork_hit_tokens = [
        int(match.group("tokens")) for match in fork_hit_by_child.values()
    ]
    fork_capture_blocks = [
        int(match.group("blocks")) for match in fork_capture_by_parent.values()
    ]
    fork_leased_blocks = [
        int(match.group("leased") or 0) for match in fork_capture_by_parent.values()
    ]
    fork_lease_release_records = list(
        FORK_LEASE_RELEASE.finditer(measured_runtime_log)
    )
    fork_lease_expire_records = list(
        FORK_LEASE_EXPIRE.finditer(measured_runtime_log)
    )
    fork_terminal_parent_records = list(
        FORK_TERMINAL_PARENT.finditer(measured_runtime_log)
    )
    fork_lease_scopes = sorted(
        {
            scope
            for match in fork_capture_by_parent.values()
            if (scope := match.group("scope")) is not None
        }
    )
    fork_lease_scope_counts: dict[str, int] = defaultdict(int)
    fork_held_unique_blocks = []
    fork_lease_budgets = set()
    for match in fork_capture_by_parent.values():
        fork_lease_scope_counts[match.group("scope") or "unreported"] += 1
        if match.group("held_unique") is not None:
            fork_held_unique_blocks.append(int(match.group("held_unique")))
        if match.group("budget") is not None:
            fork_lease_budgets.add(int(match.group("budget")))
    fork_miss_reasons: dict[str, int] = defaultdict(int)
    for reasons in fork_miss_by_child.values():
        reason = (
            "stale_or_mismatch"
            if "stale_or_mismatch" in reasons
            else sorted(reasons)[0]
        )
        fork_miss_reasons[reason] += 1
    branch_demote_by_request = {
        match.group("request"): match
        for match in BRANCH_DEMOTE.finditer(measured_runtime_log)
    }
    branch_demoted_blocks = [
        int(match.group("blocks")) for match in branch_demote_by_request.values()
    ]
    branch_record_by_request = {
        match.group("request"): match
        for match in BRANCH_RECORD.finditer(measured_runtime_log)
    }
    resample_gc_records = list(RESAMPLE_GC.finditer(measured_runtime_log))
    wall_seconds = float(benchmark.get("wall_seconds", 0.0) or 0.0)
    prefill_tokens = float(backend.get("prefill_tokens", 0.0) or 0.0)
    cached_tokens = float(
        backend.get("shared_prefill_tokens_saved", 0.0) or 0.0
    )
    generated_tokens = float(backend.get("generated_tokens", 0.0) or 0.0)
    engine_requests = float(backend.get("engine_requests", 0.0) or 0.0)
    submitted_prefix_tokens = {
        str(event.get("request_id")): int(event.get("prefix_tokens", 0) or 0)
        for event in lifecycle
        if event.get("event") == "submitted"
    }
    cancelled_parent_handles = {
        re.sub(r"-[0-9a-f]{8}$", "", match.group("parent"))
        for match in engine_fork_cancellations
    }
    zero_compute_children = {
        request_id
        for request_id in submitted_prefix_tokens
        if any(
            request_id.startswith(f"{parent}:rollout:")
            for parent in cancelled_parent_handles
        )
    }
    zero_compute_prompt_tokens = sum(
        submitted_prefix_tokens[request_id] for request_id in zero_compute_children
    )
    measured_forward_slots = float(
        backend.get("generation_forward_token_slots", 0.0) or 0.0
    )
    # Older fork-waiter artifacts charged every cancelled placeholder's logical
    # prompt as prefill. Newer runtimes exclude it at collection time. In an
    # already-corrected artifact the cancelled logical prompts exceed the whole
    # measured prefill count, so do not subtract them twice.
    recorded_zero_compute_prompt_tokens = (
        zero_compute_prompt_tokens
        if zero_compute_prompt_tokens <= prefill_tokens
        else 0
    )
    corrected_prefill_tokens = max(
        0.0, prefill_tokens - recorded_zero_compute_prompt_tokens
    )
    corrected_forward_slots = max(
        0.0, measured_forward_slots - recorded_zero_compute_prompt_tokens
    )
    return {
        "root": str(root),
        "wall_seconds": wall_seconds,
        "success_rate": benchmark.get("success_rate"),
        "jobs_per_second": benchmark.get("jobs_per_second"),
        "job_latency_seconds": benchmark.get("latency_seconds"),
        "burst_latency_seconds": {
            "origin": (
                "benchmark_release"
                if released_at is not None
                else "earliest_worker_start"
            ),
            "mean": (
                statistics.fmean(burst_latency_seconds)
                if burst_latency_seconds
                else None
            ),
            "p50": _percentile(burst_latency_seconds, 0.50),
            "p95": _percentile(burst_latency_seconds, 0.95),
            "p99": _percentile(burst_latency_seconds, 0.99),
            "max": max(burst_latency_seconds, default=None),
        },
        "forward_token_slots_per_second": (
            corrected_forward_slots / wall_seconds if wall_seconds else None
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
        "prefill_tokens": corrected_prefill_tokens,
        "zero_compute_cancelled_children": len(zero_compute_children),
        "zero_compute_prompt_tokens_excluded": recorded_zero_compute_prompt_tokens,
        "shared_prefill_tokens_saved": cached_tokens,
        "apc_token_hit_ratio": (
            cached_tokens / (cached_tokens + corrected_prefill_tokens)
            if cached_tokens + corrected_prefill_tokens
            else None
        ),
        "maximum_in_flight_requests": backend.get("maximum_in_flight_requests"),
        "native_parallel_groups": backend.get("native_parallel_groups"),
        "native_parallel_children": backend.get("native_parallel_children"),
        "kv_fork": {
            "captures": fork_captures,
            "lease_scopes": fork_lease_scopes,
            "lease_scope_counts": dict(sorted(fork_lease_scope_counts.items())),
            "terminal_parents": len(fork_terminal_parent_records),
            "hits": fork_hits,
            "misses": fork_misses,
            "hit_rate": (
                fork_hits / (fork_hits + fork_misses)
                if fork_hits + fork_misses
                else None
            ),
            "hit_tokens": {
                "total": sum(fork_hit_tokens),
                "mean": (
                    statistics.fmean(fork_hit_tokens) if fork_hit_tokens else None
                ),
                "p50": _percentile(fork_hit_tokens, 0.50),
                "p95": _percentile(fork_hit_tokens, 0.95),
            },
            "captured_blocks": {
                "total": sum(fork_capture_blocks),
                "mean": (
                    statistics.fmean(fork_capture_blocks)
                    if fork_capture_blocks
                    else None
                ),
                "p95": _percentile(fork_capture_blocks, 0.95),
            },
            "leased_blocks": {
                "total": sum(fork_leased_blocks),
                "mean": (
                    statistics.fmean(fork_leased_blocks)
                    if fork_leased_blocks
                    else None
                ),
                "p95": _percentile(fork_leased_blocks, 0.95),
                "release_events": len(fork_lease_release_records),
                "expire_events": len(fork_lease_expire_records),
                "peak_unique": max(fork_held_unique_blocks, default=None),
                "budgets": sorted(fork_lease_budgets),
            },
            "miss_reasons": dict(sorted(fork_miss_reasons.items())),
        },
        "kv_branch_eviction": {
            "demoted_requests": len(branch_demote_by_request),
            "demoted_blocks": sum(branch_demoted_blocks),
            "blocks_per_request_mean": (
                statistics.fmean(branch_demoted_blocks)
                if branch_demoted_blocks
                else None
            ),
            "blocks_per_request_p95": _percentile(branch_demoted_blocks, 0.95),
        },
        "kv_resample_gc": {
            "recorded_requests": len(branch_record_by_request),
            "candidate_records": sum(
                match.group("kind") == "candidate"
                for match in branch_record_by_request.values()
            ),
            "rollout_records": sum(
                match.group("kind") == "rollout"
                for match in branch_record_by_request.values()
            ),
            "transitions": len(resample_gc_records),
            "matched_records": sum(
                int(match.group("recorded")) for match in resample_gc_records
            ),
            "evicted_blocks": sum(
                int(match.group("evicted")) for match in resample_gc_records
            ),
            "stale_blocks": sum(
                int(match.group("stale")) for match in resample_gc_records
            ),
            "active_blocks": sum(
                int(match.group("active")) for match in resample_gc_records
            ),
            "protected_blocks": sum(
                int(match.group("protected")) for match in resample_gc_records
            ),
            "latency_ms": {
                "mean": statistics.fmean(kv_gc_ms) if kv_gc_ms else None,
                "p95": _percentile(kv_gc_ms, 0.95),
            },
            "algorithm_trace_evicted_blocks": sum(kv_gc_evicted_blocks),
        },
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
        "candidate_rollout_overlap_ms": {
            "mean": (
                statistics.fmean(candidate_rollout_overlap_ms)
                if candidate_rollout_overlap_ms
                else 0.0
            ),
            "p50": _percentile(candidate_rollout_overlap_ms, 0.50) or 0.0,
            "p95": _percentile(candidate_rollout_overlap_ms, 0.95) or 0.0,
            "max": max(candidate_rollout_overlap_ms, default=0.0),
        },
        "streamed_rollout_submission_batches": {
            "mean": (
                statistics.fmean(streamed_rollout_submission_batches)
                if streamed_rollout_submission_batches
                else 0.0
            ),
            "p95": (
                _percentile(streamed_rollout_submission_batches, 0.95) or 0.0
            ),
            "max": max(streamed_rollout_submission_batches, default=0.0),
        },
        "subtree_rollout_batches": {
            "submission_mean": (
                statistics.fmean(subtree_rollout_submission_batches)
                if subtree_rollout_submission_batches
                else 0.0
            ),
            "peak_active_mean": (
                statistics.fmean(subtree_peak_active_batches)
                if subtree_peak_active_batches
                else 0.0
            ),
            "peak_active_max": max(subtree_peak_active_batches, default=0.0),
        },
        "fused_candidate_rollout_paths": {
            "stages": len(fused_path_ms),
            "duration_ms_mean": (
                statistics.fmean(fused_path_ms) if fused_path_ms else None
            ),
            "duration_ms_p95": _percentile(fused_path_ms, 0.95),
            "repeated_candidate_tokens": sum(fused_repeated_candidate_tokens),
        },
        "engine_fork": {
            "stages": len(engine_fork_ms),
            "duration_ms_mean": (
                statistics.fmean(engine_fork_ms) if engine_fork_ms else None
            ),
            "duration_ms_p95": _percentile(engine_fork_ms, 0.95),
            "parked_children": len(engine_fork_parks),
            "compact_parked_children": sum(
                match.group("compact") == "1" for match in engine_fork_parks
            ),
            "compact_prompt_tokens_elided": sum(
                int(match.group("elided") or 0) for match in engine_fork_parks
            ),
            "activated_parents": len(engine_fork_activations),
            "activated_children": activated_engine_fork_children,
            "cancelled_parents": len(engine_fork_cancellations),
            "cancelled_children": cancelled_engine_fork_children,
            "unresolved_children": max(
                0,
                len(engine_fork_parks)
                - activated_engine_fork_children
                - cancelled_engine_fork_children,
            ),
            "group_release_events": len(engine_fork_group_releases),
            "groups_released": len(first_group_release),
            "parents_released": sum(
                int(match.group("parents"))
                for match in engine_fork_group_releases
            ),
            "first_release_remaining": sorted(
                {
                    int(match.group("remaining"))
                    for match in first_group_release.values()
                }
            ),
            "release_thresholds": sorted(
                {
                    int(match.group("threshold"))
                    for match in engine_fork_group_releases
                }
            ),
            "adaptive_release_events": len(engine_fork_adaptive_releases),
            "adaptive_parents_released": sum(
                int(match.group("parents"))
                for match in engine_fork_adaptive_releases
            ),
            "adaptive_children_released": sum(
                int(match.group("children"))
                for match in engine_fork_adaptive_releases
            ),
            "adaptive_targets": sorted(
                {
                    int(match.group("target"))
                    for match in engine_fork_adaptive_releases
                }
            ),
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
