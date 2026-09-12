"""Join CIS trees, request lifecycles, and vLLM Service Profiler batches."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUEST_SUFFIX = re.compile(
    r"^(?P<job>.*):step:(?P<block>\d+):candidate:(?P<candidate>\d+)"
    r"(?::rollout:(?P<rollout>\d+))?(?:-(?P<engine_suffix>[0-9a-f]{8}))?$"
)
BURST_INDEX = re.compile(r"^burst:\d+:(?P<index>\d+):")


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _parse_request_id(request_id: str) -> dict[str, Any] | None:
    match = REQUEST_SUFFIX.search(request_id)
    if match is None:
        return None
    rollout = match.group("rollout")
    external_request_id = (
        f"{match.group('job')}:step:{match.group('block')}:"
        f"candidate:{match.group('candidate')}"
        + ("" if rollout is None else f":rollout:{rollout}")
    )
    return {
        "request_id": external_request_id,
        "engine_request_id": request_id,
        "job_id": match.group("job"),
        "block_id": int(match.group("block")),
        "candidate_index": int(match.group("candidate")),
        "rollout_index": None if rollout is None else int(rollout),
        "kind": "candidate" if rollout is None else "rollout",
    }


def _load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _request_lifecycles(root: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for event in _load_jsonl(sorted((root / "request-traces").glob("*.jsonl"))):
        request_id = str(event.get("request_id") or "")
        identity = _parse_request_id(request_id)
        if identity is None:
            continue
        entry = grouped[request_id]
        entry.update(identity)
        entry["request_id"] = request_id
        entry["instance_id"] = str(event.get("instance_id", "unknown"))
        entry[event["event"]] = event
    return dict(grouped)


def _service_batches(root: Path) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    batches: list[dict[str, Any]] = []
    request_batches: dict[str, list[int]] = defaultdict(list)
    for path in sorted(root.rglob("batch.csv"), key=str):
        instance = next(
            (part for part in path.parts if part.startswith("profile-")), "unknown"
        )
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row.get("name") != "batchFrameworkProcessing":
                    continue
                try:
                    resources = ast.literal_eval(row.get("res_list", "[]"))
                except (SyntaxError, ValueError):
                    continue
                parsed_resources = []
                for resource in resources:
                    engine_request_id = str(resource.get("rid", ""))
                    identity = _parse_request_id(engine_request_id)
                    parsed_resources.append(
                        {
                            "request_id": (
                                engine_request_id
                                if identity is None
                                else identity["request_id"]
                            ),
                            "engine_request_id": engine_request_id,
                            "identity": identity,
                            "iteration": int(resource.get("iter", 0)),
                            "type": int(resource.get("type", -1)),
                            "scheduled_tokens": int(
                                resource.get("num_scheduled_tokens", 0)
                            ),
                            "prompt_tokens": int(resource.get("num_prompt_tokens", 0)),
                            "computed_tokens": int(
                                resource.get("num_computed_tokens", 0)
                            ),
                        }
                    )
                index = len(batches)
                batches.append(
                    {
                        "instance_id": instance,
                        "start_us": float(row.get("start_time(ms)", 0.0)) * 1_000.0,
                        "duration_us": float(row.get("during_time(ms)", 0.0))
                        * 1_000.0,
                        "batch_size": int(float(row.get("batch_size", 0.0))),
                        "batch_type": str(row.get("batch_type", "unknown")),
                        "resources": parsed_resources,
                        "source": str(path),
                    }
                )
                for resource in parsed_resources:
                    if resource["identity"] is not None:
                        request_batches[resource["request_id"]].append(index)
    return batches, dict(request_batches)


def _algorithm_jobs(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    records = _load_jsonl(sorted((root / "algorithm-traces").glob("*.jsonl")))
    for record in records:
        if not record.get("conditional_steps"):
            continue
        result[str(record["request_id"])] = record
    return result


def _finished_us(lifecycle: dict[str, Any]) -> float | None:
    event = lifecycle.get("finished")
    return None if event is None else float(event.get("event_unix_us", 0.0))


def _submitted_us(lifecycle: dict[str, Any]) -> float | None:
    event = lifecycle.get("submitted")
    return None if event is None else float(event.get("event_unix_us", 0.0))


def _scheduled_us(lifecycle: dict[str, Any]) -> float | None:
    event = lifecycle.get("finished")
    if event is None:
        return None
    value = float(event.get("scheduled_unix_us", 0.0))
    return value if value > 0 else None


def _short_job_label(job_id: str) -> str:
    label = job_id.split(":public:", 1)[-1]
    return label if len(label) <= 58 else f"{label[:55]}..."


def _profile_overview(
    *,
    jobs: dict[str, dict[str, Any]],
    lifecycles: dict[str, dict[str, Any]],
    batches: list[dict[str, Any]],
    profile_started_us: float,
    profile_finished_us: float,
    workers: int,
) -> dict[str, Any]:
    duration_ms = max(0.0, (profile_finished_us - profile_started_us) / 1_000.0)
    active_job_ids = {
        resource["identity"]["job_id"]
        for batch in batches
        for resource in batch["resources"]
        if resource["identity"] is not None
    }
    active_jobs = []
    for job_id in active_job_ids:
        record = jobs.get(job_id)
        if record is None:
            continue
        segments = []
        for event in record.get("stage_events", ()):
            name = str(event.get("name", ""))
            if name not in {"candidate", "rollout", "reward", "weight", "resample"}:
                continue
            start_us = float(event.get("start_unix_us", 0.0))
            end_us = start_us + float(event.get("duration_us", 0.0))
            if end_us < profile_started_us or start_us > profile_finished_us:
                continue
            clipped_start = max(start_us, profile_started_us)
            clipped_end = min(end_us, profile_finished_us)
            segments.append(
                [
                    round((clipped_start - profile_started_us) / 1_000.0, 3),
                    round(max(0.0, clipped_end - clipped_start) / 1_000.0, 3),
                    name,
                    int(event.get("block_id", event.get("step", 0))),
                    int(event.get("sequence_count", 0)),
                ]
            )
        if not segments:
            continue
        match = BURST_INDEX.search(job_id)
        index = int(match.group("index")) if match else len(active_jobs)
        diagnostics = record.get("diagnostics", {})
        active_jobs.append(
            {
                "id": job_id,
                "index": index,
                "label": _short_job_label(job_id),
                "prompt_tokens": int(diagnostics.get("prompt_tokens", 0)),
                "completion_tokens": int(diagnostics.get("completion_tokens", 0)),
                "segments": segments,
            }
        )
    active_jobs.sort(key=lambda item: (item["index"], item["id"]))
    job_rows = {item["id"]: index for index, item in enumerate(active_jobs)}

    global_batches = []
    for batch_index, batch in enumerate(batches):
        start_us = batch["start_us"]
        if start_us > profile_finished_us or start_us + batch["duration_us"] < profile_started_us:
            continue
        candidate = rollout = prefill = decode = 0
        per_job: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for resource in batch["resources"]:
            identity = resource["identity"]
            if identity is None:
                continue
            is_candidate = identity["kind"] == "candidate"
            candidate += is_candidate
            rollout += not is_candidate
            prefill += resource["type"] == 0
            decode += resource["type"] == 1
            row = job_rows.get(identity["job_id"])
            if row is not None:
                per_job[row][0 if is_candidate else 1] += 1
        global_batches.append(
            [
                batch_index,
                round((start_us - profile_started_us) / 1_000.0, 3),
                round(batch["duration_us"] / 1_000.0, 3),
                batch["batch_size"],
                batch["batch_type"],
                candidate,
                rollout,
                prefill,
                decode,
                len(per_job),
                [[row, counts[0], counts[1]] for row, counts in sorted(per_job.items())],
            ]
        )

    lifecycle_rows = []
    for lifecycle in lifecycles.values():
        submitted = _submitted_us(lifecycle)
        scheduled = _scheduled_us(lifecycle)
        finished = _finished_us(lifecycle)
        if submitted is None or scheduled is None or finished is None:
            continue
        if finished < profile_started_us or submitted > profile_finished_us:
            continue
        lifecycle_rows.append(
            (
                lifecycle.get("kind"),
                submitted,
                scheduled,
                finished,
            )
        )
    sample_step_ms = 250.0
    queue_samples = []
    sample_count = int(duration_ms / sample_step_ms) + 1
    for sample_index in range(sample_count + 1):
        relative_ms = min(duration_ms, sample_index * sample_step_ms)
        at_us = profile_started_us + relative_ms * 1_000.0
        cq = rq = ca = ra = 0
        for kind, submitted, scheduled, finished in lifecycle_rows:
            if submitted <= at_us < scheduled:
                if kind == "candidate":
                    cq += 1
                else:
                    rq += 1
            elif scheduled <= at_us < finished:
                if kind == "candidate":
                    ca += 1
                else:
                    ra += 1
        queue_samples.append([round(relative_ms, 3), cq, rq, ca, ra])

    queue_peak = max(queue_samples, key=lambda item: item[1] + item[2], default=[0, 0, 0, 0, 0])
    active_peak = max(queue_samples, key=lambda item: item[3] + item[4], default=[0, 0, 0, 0, 0])
    batch_candidate_and_rollout = sum(
        batch[5] > 0 and batch[6] > 0 for batch in global_batches
    )
    batch_prefill_and_decode = sum(batch[7] > 0 and batch[8] > 0 for batch in global_batches)
    return {
        "duration_ms": round(duration_ms, 3),
        "workers": workers,
        "engine": "one AsyncLLM/EngineCore scheduler",
        "model_workers": ["TP rank 0", "TP rank 1"],
        "jobs": active_jobs,
        "batches": global_batches,
        "queue_samples": queue_samples,
        "summary": {
            "active_jobs": len(active_jobs),
            "max_queued_candidates": max((item[1] for item in queue_samples), default=0),
            "max_queued_rollouts": max((item[2] for item in queue_samples), default=0),
            "max_active_candidates": max((item[3] for item in queue_samples), default=0),
            "max_active_rollouts": max((item[4] for item in queue_samples), default=0),
            "queue_peak": queue_peak,
            "active_lifecycle_peak": active_peak,
            "candidate_rollout_batch_ratio": (
                batch_candidate_and_rollout / len(global_batches)
                if global_batches
                else None
            ),
            "prefill_decode_batch_ratio": (
                batch_prefill_and_decode / len(global_batches)
                if global_batches
                else None
            ),
            "batch_size_p50": _percentile((item[3] for item in global_batches), 0.5),
            "batch_size_p95": _percentile((item[3] for item in global_batches), 0.95),
            "jobs_per_batch_p50": _percentile((item[9] for item in global_batches), 0.5),
            "jobs_per_batch_p95": _percentile((item[9] for item in global_batches), 0.95),
        },
    }


def _block_score(
    job_id: str,
    block: dict[str, Any],
    lifecycles: dict[str, dict[str, Any]],
    request_batches: dict[str, list[int]],
) -> tuple[int, int, int]:
    request_ids = []
    for candidate in block["candidates"]:
        request_ids.append(candidate["request_id"])
        request_ids.extend(rollout["request_id"] for rollout in candidate["rollouts"])
    complete = sum(_finished_us(lifecycles.get(item, {})) is not None for item in request_ids)
    profiled = sum(bool(request_batches.get(item)) for item in request_ids)
    full_rollouts = sum(
        len(candidate["rollouts"]) for candidate in block["candidates"]
    )
    return profiled, complete, full_rollouts


def _request_payload(
    request_id: str,
    *,
    origin_us: float,
    selected_candidate: int,
    lifecycles: dict[str, dict[str, Any]],
    request_batches: dict[str, list[int]],
    batches: list[dict[str, Any]],
) -> dict[str, Any]:
    lifecycle = lifecycles.get(request_id, {})
    identity = _parse_request_id(request_id) or {}
    submitted = lifecycle.get("submitted", {})
    finished = lifecycle.get("finished", {})
    indices = request_batches.get(request_id, [])
    participation = []
    for batch_index in indices:
        batch = batches[batch_index]
        resource = next(
            item for item in batch["resources"] if item["request_id"] == request_id
        )
        participation.append(
            [
                round((batch["start_us"] - origin_us) / 1_000.0, 3),
                round(batch["duration_us"] / 1_000.0, 3),
                batch_index,
                resource["iteration"],
                resource["type"],
                resource["scheduled_tokens"],
                resource["computed_tokens"],
                batch["batch_size"],
            ]
        )
    def relative(value: Any) -> float | None:
        number = float(value or 0.0)
        return None if number <= 0 else round((number - origin_us) / 1_000.0, 3)

    def duration(value: Any) -> float | None:
        return None if value is None else round(float(value) / 1_000.0, 3)

    return {
        "id": request_id,
        "kind": identity.get("kind"),
        "candidate": identity.get("candidate_index"),
        "rollout": identity.get("rollout_index"),
        "selected": identity.get("candidate_index") == selected_candidate,
        "instance": lifecycle.get("instance_id", "unknown"),
        "submit_ms": relative(submitted.get("event_unix_us")),
        "scheduled_ms": relative(finished.get("scheduled_unix_us")),
        "first_token_ms": relative(finished.get("first_token_unix_us")),
        "finish_ms": relative(finished.get("event_unix_us")),
        "queue_ms": duration(finished.get("queue_us")),
        "scheduled_to_first_token_ms": duration(
            finished.get("scheduled_to_first_token_us")
        ),
        "decode_ms": duration(finished.get("decode_us")),
        "prefix_tokens": submitted.get("prefix_tokens"),
        "cached_tokens": finished.get("cached_tokens"),
        "output_tokens": finished.get("output_tokens"),
        "finish_reason": finished.get("finish_reason"),
        "participation": participation,
    }


def export(root: Path) -> dict[str, Any]:
    lifecycles = _request_lifecycles(root)
    batches, request_batches = _service_batches(root)
    jobs = _algorithm_jobs(root)
    benchmark = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
    profile_window = benchmark.get("profile", {}).get("window", {})
    profile_started_us = float(profile_window.get("started_at", 0.0)) * 1e6
    profile_finished_us = float(profile_window.get("finished_at", 0.0)) * 1e6
    overview = _profile_overview(
        jobs=jobs,
        lifecycles=lifecycles,
        batches=batches,
        profile_started_us=profile_started_us,
        profile_finished_us=profile_finished_us,
        workers=int(benchmark.get("workers", 0)),
    )
    candidates = []
    for job_id, record in jobs.items():
        for block in record.get("conditional_steps", ()):
            score = _block_score(job_id, block, lifecycles, request_batches)
            candidates.append((score, job_id, block, record))
    if not candidates:
        raise RuntimeError("no schema-v2 CIS block has request lifecycle data")
    _, job_id, block, record = max(candidates, key=lambda item: item[0])
    request_ids = []
    for candidate in block["candidates"]:
        request_ids.append(candidate["request_id"])
        request_ids.extend(rollout["request_id"] for rollout in candidate["rollouts"])
    submitted_values = [
        value
        for request_id in request_ids
        if (value := _submitted_us(lifecycles.get(request_id, {}))) is not None
    ]
    if not submitted_values:
        raise RuntimeError("selected block has no submitted requests")
    origin_us = min(submitted_values)
    requests = [
        _request_payload(
            request_id,
            origin_us=origin_us,
            selected_candidate=int(block["selected_candidate"]),
            lifecycles=lifecycles,
            request_batches=request_batches,
            batches=batches,
        )
        for request_id in request_ids
    ]
    by_id = {request["id"]: request for request in requests}
    candidate_finish = [
        request["finish_ms"]
        for request in requests
        if request["kind"] == "candidate" and request["finish_ms"] is not None
    ]
    rollout_finish = [
        request["finish_ms"]
        for request in requests
        if request["kind"] == "rollout" and request["finish_ms"] is not None
    ]
    last_candidate = max(candidate_finish, default=None)
    last_rollout = max(rollout_finish, default=None)
    child_delays = []
    barrier_waits = []
    candidate_rows = []
    for candidate in block["candidates"]:
        parent = by_id[candidate["request_id"]]
        children = [by_id[item["request_id"]] for item in candidate["rollouts"]]
        child_submits = [
            child["submit_ms"] for child in children if child["submit_ms"] is not None
        ]
        child_delay = (
            None
            if parent["finish_ms"] is None or not child_submits
            else min(child_submits) - parent["finish_ms"]
        )
        if child_delay is not None:
            child_delays.append(child_delay)
        child_barrier = [
            last_rollout - child["finish_ms"]
            for child in children
            if last_rollout is not None and child["finish_ms"] is not None
        ]
        barrier_waits.extend(child_barrier)
        child_finishes = [
            child["finish_ms"]
            for child in children
            if child["finish_ms"] is not None
        ]
        candidate_rows.append(
            {
                "candidate": candidate["candidate_index"],
                "selected": candidate["selected"],
                "output_tokens": candidate["output_tokens"],
                "log_weight": candidate["log_weight"],
                "child_admission_delay_ms": (
                    None if child_delay is None else round(child_delay, 3)
                ),
                "rollout_finish_spread_ms": (
                    None
                    if len(child_finishes) < 2
                    else round(
                        max(child_finishes) - min(child_finishes),
                        3,
                    )
                ),
            }
        )

    selected_set = set(request_ids)
    relevant_batch_indices = sorted(
        {
            batch_index
            for request_id in request_ids
            for batch_index in request_batches.get(request_id, ())
        }
    )
    compact_batches = []
    for batch_index in relevant_batch_indices:
        batch = batches[batch_index]
        related = [
            item for item in batch["resources"] if item["request_id"] in selected_set
        ]
        related_candidate = sum(
            item["identity"]["kind"] == "candidate" for item in related
        )
        related_rollout = len(related) - related_candidate
        compact_batches.append(
            [
                batch_index,
                round((batch["start_us"] - origin_us) / 1_000.0, 3),
                round(batch["duration_us"] / 1_000.0, 3),
                batch["batch_size"],
                batch["batch_type"],
                len(related),
                related_candidate,
                related_rollout,
                sum(item["scheduled_tokens"] for item in batch["resources"]),
            ]
        )

    request_queue_candidate = [
        request["queue_ms"]
        for request in requests
        if request["kind"] == "candidate" and request["queue_ms"] is not None
    ]
    request_queue_rollout = [
        request["queue_ms"]
        for request in requests
        if request["kind"] == "rollout" and request["queue_ms"] is not None
    ]
    selected_profiled_requests = sum(
        bool(request["participation"]) for request in requests
    )
    mixed_batches = sum(
        "Prefill" in batch[4] and "Decode" in batch[4] for batch in compact_batches
    )
    full_batches = sum(batch[3] >= 256 for batch in compact_batches)
    rollout_output_tokens = [
        int(request["output_tokens"])
        for request in requests
        if request["kind"] == "rollout" and request["output_tokens"] is not None
    ]
    sibling_hits = 0
    rollout_participations = 0
    for request in requests:
        if request["kind"] != "rollout":
            continue
        for item in request["participation"]:
            batch = batches[item[2]]
            siblings = sum(
                resource["identity"] is not None
                and resource["identity"]["job_id"] == job_id
                and resource["identity"]["block_id"] == block["block_id"]
                and resource["identity"]["candidate_index"] == request["candidate"]
                and resource["identity"]["kind"] == "rollout"
                for resource in batch["resources"]
            )
            rollout_participations += 1
            sibling_hits += siblings > 1

    return {
        "schema_version": 1,
        "source": str(root),
        "topology": benchmark.get("topology", {}),
        "profile": benchmark.get("profile", {}),
        "overview": overview,
        "job": {
            "id": job_id,
            "instance": str(record.get("diagnostics", {}).get("instance_id", "unknown")),
            "prompt_tokens": int(record.get("diagnostics", {}).get("prompt_tokens", 0)),
            "completion_tokens": int(
                record.get("diagnostics", {}).get("completion_tokens", 0)
            ),
            "block_id": int(block["block_id"]),
            "selected_candidate": int(block["selected_candidate"]),
            "origin_unix_us": origin_us,
            "last_candidate_ms": last_candidate,
            "last_rollout_ms": last_rollout,
            "profile_window_start_ms": (
                None
                if profile_started_us <= 0
                else round((profile_started_us - origin_us) / 1_000.0, 3)
            ),
            "profile_window_end_ms": (
                None
                if profile_finished_us <= 0
                else round((profile_finished_us - origin_us) / 1_000.0, 3)
            ),
        },
        "requests": requests,
        "candidates": candidate_rows,
        "batches": compact_batches,
        "summary": {
            "algorithm_jobs": len(jobs),
            "lifecycle_requests": len(lifecycles),
            "service_batches": len(batches),
            "selected_block_requests": len(requests),
            "selected_candidates": sum(
                request["kind"] == "candidate" for request in requests
            ),
            "selected_rollouts": sum(
                request["kind"] == "rollout" for request in requests
            ),
            "terminal_candidates": sum(
                not candidate["rollouts"] for candidate in block["candidates"]
            ),
            "profiled_block_requests": selected_profiled_requests,
            "candidate_child_delay_p50_ms": _percentile(child_delays, 0.5),
            "candidate_child_delay_p95_ms": _percentile(child_delays, 0.95),
            "rollout_barrier_wait_p50_ms": _percentile(barrier_waits, 0.5),
            "rollout_barrier_wait_p95_ms": _percentile(barrier_waits, 0.95),
            "candidate_queue_p50_ms": _percentile(request_queue_candidate, 0.5),
            "candidate_queue_p95_ms": _percentile(request_queue_candidate, 0.95),
            "rollout_queue_p50_ms": _percentile(request_queue_rollout, 0.5),
            "rollout_queue_p95_ms": _percentile(request_queue_rollout, 0.95),
            "rollout_sibling_cobatch_ratio": (
                sibling_hits / rollout_participations
                if rollout_participations
                else None
            ),
            "candidate_finish_spread_ms": (
                max(candidate_finish) - min(candidate_finish)
                if len(candidate_finish) > 1
                else None
            ),
            "rollout_finish_spread_ms": (
                max(rollout_finish) - min(rollout_finish)
                if len(rollout_finish) > 1
                else None
            ),
            "selected_mixed_batch_ratio": (
                mixed_batches / len(compact_batches) if compact_batches else None
            ),
            "selected_full_batch_ratio": (
                full_batches / len(compact_batches) if compact_batches else None
            ),
            "rollout_output_p50": _percentile(rollout_output_tokens, 0.5),
            "rollout_output_p95": _percentile(rollout_output_tokens, 0.95),
            "rollout_output_max": max(rollout_output_tokens, default=None),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source-label")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = export(args.profile)
    if args.source_label:
        payload["source"] = args.source_label
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
