from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.swe_agent.service import ConditionalISRunner
from inference_scaling.swe_agent import benchmark
from inference_scaling.swe_agent.algorithm_grid import _pareto
from inference_scaling.swe_agent.calibration import (
    block_ess_ratios,
    calibrate_logprob_alpha,
    calibrate_reward_temperature,
)
from inference_scaling.swe_agent.workload import freeze_workload
from inference_scaling.swe_agent.reward_screen import _screen_temperature
from inference_scaling.swe_agent.runtime_tune import select_arms
from tests.test_swe_agent import _AgentBackend, _runner_config


def test_reward_calibration_targets_median_candidate_ess() -> None:
    blocks = (
        ((-1.0, -1.2), (-2.0, -2.2), (-3.0, -3.2), (-4.0, -4.2)),
        ((-1.0, -1.1), (-1.7, -1.8), (-2.7, -2.8), (-4.5, -4.6)),
    )

    alpha = calibrate_logprob_alpha(blocks, target_ess_ratio=0.6)
    temperature = calibrate_reward_temperature(blocks, target_ess_ratio=0.6)

    assert 1.0 < alpha.parameter < 4.0
    assert alpha.median_ess_ratio == pytest.approx(0.6, abs=1e-6)
    assert 1e-3 < temperature.parameter < 100.0
    assert temperature.median_ess_ratio == pytest.approx(0.6, abs=1e-6)
    assert block_ess_ratios(blocks, scale=0.0) == pytest.approx((1.0, 1.0))


def _trace_record(index: int, *, successful: bool = True) -> dict:
    return {
        "request_id": f"request-{index}",
        "messages": [{"role": "user", "content": str(index)}],
        "message": {
            "extra": {
                "actions": ([{"command": "pwd"}] if successful else []),
            }
        },
        "diagnostics": {"prompt_tokens": 100 + index},
    }


def test_freeze_workload_is_deterministic_disjoint_and_filters_failures(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.jsonl"
    records = [_trace_record(index) for index in range(8)]
    records.extend([_trace_record(0), _trace_record(99, successful=False)])
    trace.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    first = freeze_workload(trace, tmp_path / "first", seed=7, total=8)
    second = freeze_workload(trace, tmp_path / "second", seed=7, total=8)

    assert first["tune"]["sha256"] == second["tune"]["sha256"]
    assert first["holdout"]["sha256"] == second["holdout"]["sha256"]
    tune = {
        json.loads(line)["request_id"]
        for line in (tmp_path / "first" / "tune-4.jsonl").read_text().splitlines()
    }
    holdout = {
        json.loads(line)["request_id"]
        for line in (tmp_path / "first" / "holdout-4.jsonl").read_text().splitlines()
    }
    assert len(tune) == len(holdout) == 4
    assert tune.isdisjoint(holdout)


def test_burst_routes_whole_jobs_across_endpoints(monkeypatch) -> None:
    calls = []

    def fake_post(endpoint, payload, timeout):
        calls.append((endpoint, payload, timeout))
        return {
            "message": {"extra": {"actions": [{"command": "pwd"}]}},
            "diagnostics": {"prompt_tokens": 10},
        }

    monkeypatch.setattr(benchmark, "_post", fake_post)
    records = [_trace_record(index) for index in range(4)]

    result = benchmark.run_burst(
        records,
        ("http://one", "http://two"),
        workers=4,
        seed=19,
        conditional_overrides={"candidate_count": 8, "rollout_count": 2},
    )

    assert result["success_rate"] == 1.0
    assert result["jobs_per_second"] > 0
    assert result["conditional_is"]["candidate_count"] == 8
    assert all(call[1]["conditional_is"]["rollout_count"] == 2 for call in calls)
    assert sorted(call[0] for call in calls) == [
        "http://one",
        "http://one",
        "http://two",
        "http://two",
    ]


def test_reward_screen_uses_one_pool_for_both_reward_families(
    tmp_path: Path,
) -> None:
    backend = _AgentBackend()
    TabularAutoregressiveBackend.__init__(
        backend,
        {},
        fallback=(0.30, 0.20, 0.15, 0.13, 0.12, 0.10),
    )
    backend.tokenizer.eos_token_id = 6
    config = _runner_config(tmp_path / "unused.jsonl")
    config["generation"]["max_new_tokens"] = 128
    config["vllm"]["max_model_len"] = 256
    record = _trace_record(1)

    result = _screen_temperature(
        backend,
        config,
        [record],
        temperature=0.7,
        root_seed=3,
    )

    assert result["temperature"] == 0.7
    assert result["blocks"] == 1
    assert result["rollout_sequences"] == 8
    assert len(result["reward_blocks"]["sequence_logprob"]) == 1
    assert len(result["reward_blocks"]["consilience"]) == 1


def test_runner_namespaces_internal_requests_by_job(tmp_path: Path) -> None:
    backend = _AgentBackend()
    requests = []
    original_sample_batch = backend.sample_batch

    def record_sample_batch(batch):
        requests.extend(batch)
        return original_sample_batch(batch)

    backend.sample_batch = record_sample_batch
    runner = ConditionalISRunner(backend, _runner_config(tmp_path / "unused.jsonl"))
    messages = [{"role": "user", "content": "fix it"}]

    first_execution = runner.execute(messages, seed=1, request_namespace="job-one")
    first_count = len(requests)
    runner.execute(messages, seed=2, request_namespace="job-two")

    first = {request.request_id for request in requests[:first_count]}
    second = {request.request_id for request in requests[first_count:]}
    assert first
    assert second
    assert all(request_id.startswith("job-one:") for request_id in first)
    assert all(request_id.startswith("job-two:") for request_id in second)
    assert first.isdisjoint(second)
    names = {event["name"] for event in first_execution.stage_events}
    assert {"candidate", "rollout", "reward", "weight", "resample", "block"} <= names
    assert all(event["duration_us"] >= 0 for event in first_execution.stage_events)


def _arm(arm_id: str, throughput: float, p95: float, *, success: float = 1.0):
    return {
        "arm_id": arm_id,
        "jobs_per_second": throughput,
        "success_rate": success,
        "latency_seconds": {"p95": p95},
        "forward_tokens": 100,
        "measurements": [],
    }


def test_runtime_gate_and_algorithm_pareto_are_deterministic() -> None:
    fast = _arm("fast", 2.0, 10.0)
    balanced = _arm("balanced", 1.8, 8.5)
    tail_heavy = _arm("tail-heavy", 3.0, 12.0)
    failed = _arm("failed", 9.0, 1.0, success=0.5)

    selected = select_arms(
        [tail_heavy, fast, balanced, failed], keep=2, p95_factor=1.25
    )

    assert [result["arm_id"] for result in selected] == ["fast", "balanced"]
    assert set(_pareto([fast, balanced])) == {"fast", "balanced"}
