from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.swe_agent.service import ConditionalISRunner
from inference_scaling.swe_agent import benchmark
from inference_scaling.swe_agent.algorithm_grid import _pareto
from inference_scaling.swe_agent.archive_artifacts import archive_artifacts
from inference_scaling.swe_agent.calibration import (
    block_ess_ratios,
    calibrate_logprob_alpha,
    calibrate_reward_temperature,
)
from inference_scaling.swe_agent.workload import extract_call_snapshots, freeze_workload
from inference_scaling.swe_agent.reward_screen import _screen_temperature
from inference_scaling.swe_agent.runtime_tune import (
    adaptive_search_plan,
    engine_grid,
    select_max_model_len,
    select_arms,
)
from inference_scaling.swe_agent.topology import capability_matrix, native_topologies
from inference_scaling.swe_agent.swebench import select_instances
from inference_scaling.swe_agent.profile_analysis import (
    analyze_algorithm_trace,
    analyze_ascend_profile,
    analyze_benchmark,
    recommendations,
)
from inference_scaling.swe_agent.profile_report import render_profile_report
from inference_scaling.swe_agent.graph_capture import graph_capture_candidates
from inference_scaling.swe_agent.evaluate import build_evaluation_command
from inference_scaling.swe_agent.deploy import (
    build_docker_command,
    parse_npu_processes,
)
from inference_scaling.swe_agent.deployment_manifest import build_deployment_manifest
from inference_scaling.swe_agent.profile_matrix import load_matrix, profile_commands
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
    snapshots = iter(
        (
            {"generation_forward_token_slots": 10},
            {"generation_forward_token_slots": 20},
            {"generation_forward_token_slots": 16},
            {"generation_forward_token_slots": 27},
        )
    )
    monkeypatch.setattr(
        benchmark, "_backend_snapshot", lambda endpoint: next(snapshots)
    )
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
    assert result["backend_delta"]["generation_forward_token_slots"] == 13
    assert all(call[1]["conditional_is"]["rollout_count"] == 2 for call in calls)
    assert sorted(call[0] for call in calls) == [
        "http://one",
        "http://one",
        "http://two",
        "http://two",
    ]
    assert all(
        call[1]["request_id"].startswith(result["run_namespace"]) for call in calls
    )


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
    config["generation"]["max_new_tokens"] = 256
    config["vllm"]["max_model_len"] = 512
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
    assert result["rollout_sequences"] == 16
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
    assert all(event["job_id"] == "job-one" for event in first_execution.stage_events)
    assert all(
        event["block_id"] == event["step"] for event in first_execution.stage_events
    )
    assert all(event["start_unix_us"] > 0 for event in first_execution.stage_events)


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


def test_four_card_topology_matrix_is_explicit_about_capability_gates() -> None:
    topologies = {item.topology_id: item for item in native_topologies(19000)}

    assert len(topologies["2xtp2-round-robin"].services) == 2
    assert topologies["2xtp2-least-outstanding"].routing == "least_outstanding"
    assert len(topologies["2xpp2-round-robin"].services) == 2
    assert topologies["2xpp2-round-robin"].services[0].pipeline_parallel_size == 2
    matrix = capability_matrix()
    assert all(item["verification_status"] for item in matrix["specified_two_instance"])
    assert "tp4" in matrix["excluded_from_primary_comparison"]
    gated = matrix["gated"]
    assert "pd-2-plus-2" in gated
    assert "candidate-rollout-stage-pipeline" in gated


def test_adaptive_runtime_plan_expands_real_boundaries() -> None:
    plan = adaptive_search_plan()
    configs = engine_grid()

    assert len(configs) == 24
    assert {item.topology for item in configs} == {"tp2", "pp2"}
    assert max(item.max_num_seqs for item in configs) == 512
    assert plan["boundary_expansion"]["max_num_seqs"][-1] == 2048
    assert plan["boundary_expansion"]["max_num_batched_tokens"][-1] == 524288
    assert plan["partial_prefill"][-1] == [8, 8]
    assert select_max_model_len([_trace_record(1)]) == 16384
    long_record = _trace_record(2)
    long_record["diagnostics"]["prompt_tokens"] = 32001
    assert select_max_model_len([long_record]) == 65536


def test_public_trajectory_extracts_each_model_call_without_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "task.traj.json"
    trajectory = {
        "instance_id": "task-1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": "inspect"},
            {"role": "tool", "content": "result", "tool_call_id": "call-1"},
            {"role": "assistant", "content": "finish"},
        ],
    }

    snapshots = extract_call_snapshots(
        trajectory,
        source=source,
        token_count=lambda messages: len(messages) * 10,
    )

    assert [item["request_id"] for item in snapshots] == [
        "public:task-1:call-0",
        "public:task-1:call-1",
    ]
    assert [item["diagnostics"]["prompt_tokens"] for item in snapshots] == [20, 40]
    assert snapshots[0]["messages"] == trajectory["messages"][:2]
    assert snapshots[1]["messages"] == trajectory["messages"][:4]


def test_swebench_launcher_selects_canonical_or_explicit_order() -> None:
    instances = [{"instance_id": f"task-{index}"} for index in range(5)]

    canonical = select_instances(
        instances,
        count=3,
    )
    explicit = select_instances(
        instances,
        count=1,
        instance_ids=("task-4", "task-1"),
    )

    assert [item["instance_id"] for item in canonical] == [
        "task-0",
        "task-1",
        "task-2",
    ]
    assert [item["instance_id"] for item in explicit] == ["task-4", "task-1"]


def test_evaluator_uses_pinned_local_dataset_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "verified.json"
    command = build_evaluation_command(
        dataset_snapshot=snapshot,
        predictions=tmp_path / "preds.json",
        instance_ids=("task-1", "task-2"),
        run_id="quality-10",
        report_directory=tmp_path / "reports",
        workers=2,
        timeout=1800,
        open_file_limit=4096,
    )

    assert command[command.index("--dataset_name") + 1] == str(snapshot)
    assert command[-2:] == ["task-1", "task-2"]


def test_deployer_rejects_busy_cards_and_mounts_native_categorical(
    tmp_path: Path,
) -> None:
    npu_output = """
| NPU     Chip              | Process id    | Process name             | Process memory(MB) |
| 0       0                 | 1234          | python                    | 100                |
| 2       0                 | 5678          | VLLMWorker_TP             | 40000              |
"""
    assert parse_npu_processes(npu_output) == {0: [1234], 2: [5678]}

    repository = tmp_path / "repo"
    config = repository / "configs" / "service.toml"
    model = tmp_path / "model"
    categorical = tmp_path / "categorical"
    cache = tmp_path / "cache"
    config.parent.mkdir(parents=True)
    config.write_text("config")
    model.mkdir()
    for relative in (
        "runtime/sampler.py",
        "vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so",
        "vllm_ascend/libvllm_ascend_kernels.so",
    ):
        path = categorical / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)

    command, assets = build_docker_command(
        image="vllm:0.18",
        name="cis",
        repository=repository,
        model=model,
        config=config,
        categorical_root=categorical,
        cache_root=cache,
        devices=(0, 1),
        port=8123,
    )

    assert command.count("--device") == 5
    assert "VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE=1" in command
    assert any("runtime/sampler.py" in value for value in command)
    assert all(asset["sha256"] for asset in assets)


def test_profile_analysis_combines_algorithm_runtime_and_npu_evidence(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "diagnostics": {"algorithm_seconds": 10.0},
                "stage_events": [
                    {"name": "candidate", "step": 0, "duration_us": 4_000_000},
                    {"name": "rollout", "step": 0, "duration_us": 4_000_000},
                    {"name": "reward", "step": 0, "duration_us": 1_000_000},
                    {"name": "weight", "step": 0, "duration_us": 200_000},
                    {"name": "resample", "step": 0, "duration_us": 100_000},
                    {"name": "block", "step": 0, "duration_us": 10_000_000},
                ],
            }
        )
        + "\n"
    )
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "requests": 1,
                "success_rate": 1.0,
                "jobs_per_second": 0.1,
                "latency_seconds": {"p95": 10.0},
                "backend_delta": {
                    "prefill_tokens": 20,
                    "shared_prefill_tokens_saved": 80,
                },
                "measurements": [
                    {"success": True, "endpoint": "http://one", "seconds": 10.0}
                ],
            }
        )
    )
    rank = tmp_path / "npu" / "rank0" / "ASCEND_PROFILER_OUTPUT"
    rank.mkdir(parents=True)
    (rank / "kernel_details.csv").write_text(
        "Start Time(us),Duration(us),Type,Name\n"
        "0,10,MatMul,matmul\n"
        "5,10,hcom_allReduce_,AivKernel\n"
    )
    (rank / "communication.json").write_text(
        json.dumps(
            {
                "step": {
                    "collective": {
                        "one": {
                            "Communication Time Info": {
                                "Elapse Time(ms)": 0.01,
                                "Transit Time(ms)": 0.005,
                            }
                        }
                    }
                }
            }
        )
    )

    benchmark = analyze_benchmark(benchmark_path)
    algorithm = analyze_algorithm_trace(trace)
    ascend = analyze_ascend_profile(tmp_path / "npu")
    actions = recommendations(benchmark, algorithm, ascend)

    assert benchmark["apc_token_hit_ratio"] == pytest.approx(0.8)
    assert algorithm["block_gap_share"] == pytest.approx(0.07)
    assert ascend["rank_count"] == 1
    assert ascend["ranks"][0]["exposed_communication_ratio"] == pytest.approx(0.5)
    assert any(item["trigger"] == "exposed_hccl_above_15_percent" for item in actions)


def test_profile_report_keeps_raw_data_and_renders_recomputable_outputs(
    tmp_path: Path,
) -> None:
    (tmp_path / "algorithm-traces").mkdir()
    (tmp_path / "algorithm-traces" / "rank0.jsonl").write_text(
        json.dumps(
            {
                "request_id": "job-1",
                "diagnostics": {"algorithm_seconds": 0.01},
                "stage_events": [
                    {
                        "name": "candidate",
                        "duration_us": 10,
                        "start_unix_us": 1_000_000,
                        "instance_id": "one",
                        "step": 0,
                    }
                ],
            }
        )
        + "\n"
    )
    (tmp_path / "benchmark.json").write_text(
        json.dumps(
            {
                "requests": 1,
                "success_rate": 1.0,
                "jobs_per_second": 1.0,
                "latency_seconds": {"p95": 1.0},
                "backend_delta": {},
                "measurements": [],
                "profile": {"window": {"started_at": 1.0}},
            }
        )
    )
    rank = tmp_path / "torch" / "rank0" / "ASCEND_PROFILER_OUTPUT"
    rank.mkdir(parents=True)
    (rank / "kernel_details.csv").write_text(
        "Start Time(us),Duration(us),Type,Name\n0,10,MatMul,matmul\n"
    )
    service = tmp_path / "service" / "output"
    service.mkdir(parents=True)
    (service / "batch.csv").write_text(
        "timestamp,num_scheduled_tokens,waiting_requests\n1,31,3\n2,63,2\n3,95,1\n"
    )

    result = render_profile_report(tmp_path)

    assert Path(result["analysis"]).is_file()
    assert Path(result["report"]).is_file()
    timeline = json.loads(Path(result["timeline"]["path"]).read_text())
    assert {item["cat"] for item in timeline["traceEvents"]} == {
        "conditional-is",
        "npu-kernel",
    }
    assert (tmp_path / "algorithm-traces" / "rank0.jsonl").is_file()
    analysis = json.loads(Path(result["analysis"]).read_text())
    capture = graph_capture_candidates(analysis["service"])
    assert capture["capture_sizes"][-1] == 96
    assert "cudagraph_capture_sizes" in capture["override"]


def test_selected_tuner_result_becomes_two_or_four_instance_profile_manifest(
    tmp_path: Path,
) -> None:
    tuning = tmp_path / "result.json"
    tuning.write_text(
        json.dumps(
            {
                "winner": {
                    "engine": {
                        "max_num_seqs": 384,
                        "max_num_batched_tokens": 65536,
                        "gpu_memory_utilization": 0.94,
                        "topology": "pp2",
                        "max_num_partial_prefills": 4,
                        "max_long_partial_prefills": 2,
                    },
                    "feature": {
                        "feature_id": "optimized-baseline",
                        "overrides": ["vllm.async_scheduling=false"],
                        "environment": [["VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE", "1"]],
                    },
                    "workers": 32,
                }
            }
        )
    )

    deployment = build_deployment_manifest(
        tuning,
        deployment_id="four-card",
        devices=("0,1", "2,3"),
        routing="least_outstanding",
        workers=64,
    )

    assert deployment["pipeline_parallel_size"] == 2
    assert deployment["tensor_parallel_size"] == 1
    assert deployment["workers"] == 64
    assert deployment["devices"] == ["0,1", "2,3"]
    assert "vllm.max_num_seqs=384" in deployment["overrides"]


def test_profile_matrix_plans_separate_unprofiled_service_and_torch_passes(
    tmp_path: Path,
) -> None:
    algorithms = load_matrix(
        Path(__file__).parents[1] / "configs" / "swebench" / "profile_matrix.toml"
    )
    commands = profile_commands(
        config=tmp_path / "service.toml",
        matrix=algorithms,
        deployments=[
            {
                "id": "two-card",
                "devices": ["0,1"],
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "routing": "round_robin",
                "workers": 64,
                "overrides": [],
            }
        ],
        workload=tmp_path / "workload.jsonl",
        warmup_workload=None,
        output=tmp_path / "profiles",
        seed=1,
    )

    assert len(commands) == 14
    assert sum(item["profiler"] == "none" for item in commands) == 6
    assert {item["algorithm"] for item in commands} == {"P0", "P1", "P2", "P3"}
    assert all("--profiler" in item["command"] for item in commands)


def test_profile_archive_has_external_sha256(tmp_path: Path) -> None:
    source = tmp_path / "profile"
    source.mkdir()
    (source / "raw.json").write_text("raw")

    result = archive_artifacts(source, tmp_path / "archives" / "profile.tar.gz")

    assert Path(result["archive"]).is_file()
    assert Path(result["checksum"]).read_text().startswith(str(result["sha256"]))
