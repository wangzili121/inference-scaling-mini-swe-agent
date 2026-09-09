"""Adaptive general-runtime tuner for saturated Conditional IS workloads."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from inference_scaling.swe_agent.artifacts import (
    sha256_file,
    source_revision,
    write_artifact_manifest,
)
from inference_scaling.swe_agent.benchmark import _load_records, run_burst


@dataclass(frozen=True, slots=True)
class EngineConfig:
    max_num_seqs: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    topology: str = "tp2"
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    max_model_len: int = 32768

    def __post_init__(self) -> None:
        if self.topology not in {"tp2", "pp2"}:
            raise ValueError("two-card topology must be tp2 or pp2")
        if not 1 <= self.max_long_partial_prefills <= self.max_num_partial_prefills:
            raise ValueError(
                "long partial prefills must be within total partial prefills"
            )

    @property
    def config_id(self) -> str:
        memory = str(self.gpu_memory_utilization).replace(".", "p")
        return (
            f"{self.topology}-mns{self.max_num_seqs}"
            f"-mbt{self.max_num_batched_tokens}-mem{memory}"
            f"-ppf{self.max_num_partial_prefills}x{self.max_long_partial_prefills}"
            f"-ctx{self.max_model_len}"
        )

    def overrides(self) -> tuple[str, ...]:
        tp, pp = (2, 1) if self.topology == "tp2" else (1, 2)
        return (
            f"vllm.tensor_parallel_size={tp}",
            f"vllm.pipeline_parallel_size={pp}",
            f"vllm.max_num_seqs={self.max_num_seqs}",
            f"vllm.max_num_batched_tokens={self.max_num_batched_tokens}",
            f"vllm.gpu_memory_utilization={self.gpu_memory_utilization}",
            f"vllm.max_model_len={self.max_model_len}",
            f"vllm.engine_kwargs.max_num_partial_prefills={self.max_num_partial_prefills}",
            f"vllm.engine_kwargs.max_long_partial_prefills={self.max_long_partial_prefills}",
        )


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    feature_id: str
    overrides: tuple[str, ...] = ()
    environment: tuple[tuple[str, str | None], ...] = ()


BASE_FEATURES = FeatureConfig(
    "optimized-baseline",
    (
        "vllm.async_scheduling=false",
        "vllm.enforce_eager=false",
        "vllm.engine_kwargs.additional_config.enable_cpu_binding=true",
    ),
    (
        ("VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE", "1"),
        ("HCCL_OP_EXPANSION_MODE", None),
    ),
)


def feature_ablation_matrix() -> tuple[FeatureConfig, ...]:
    return (
        BASE_FEATURES,
        FeatureConfig(
            "stock-sampler",
            BASE_FEATURES.overrides,
            (("VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE", "0"),),
        ),
        FeatureConfig(
            "eager",
            ("vllm.enforce_eager=true",),
            BASE_FEATURES.environment,
        ),
        FeatureConfig(
            "hccl-aiv",
            BASE_FEATURES.overrides,
            (
                ("VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE", "1"),
                ("HCCL_OP_EXPANSION_MODE", "AIV"),
            ),
        ),
        FeatureConfig(
            "async-scheduling",
            (
                "vllm.async_scheduling=true",
                "vllm.enforce_eager=false",
                "vllm.engine_kwargs.additional_config.enable_cpu_binding=true",
            ),
            BASE_FEATURES.environment,
        ),
        FeatureConfig(
            "cpu-binding-off",
            (
                "vllm.async_scheduling=false",
                "vllm.enforce_eager=false",
                "vllm.engine_kwargs.additional_config.enable_cpu_binding=false",
            ),
            BASE_FEATURES.environment,
        ),
    )


def engine_grid(
    max_num_seqs: Sequence[int] = (64, 128, 256, 512),
    max_num_batched_tokens: Sequence[int] = (8192, 32768, 131072),
    gpu_memory_utilization: Sequence[float] = (0.92,),
    topologies: Sequence[str] = ("tp2", "pp2"),
) -> tuple[EngineConfig, ...]:
    return tuple(
        EngineConfig(int(mns), int(mbt), float(memory), topology=str(topology))
        for topology, mns, mbt, memory in itertools.product(
            topologies,
            max_num_seqs,
            max_num_batched_tokens,
            gpu_memory_utilization,
        )
    )


def adaptive_search_plan() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "feature_ab": [item.feature_id for item in feature_ablation_matrix()],
        "topologies": ["tp2", "pp2"],
        "coarse": {
            "max_num_seqs": [64, 128, 256, 512],
            "max_num_batched_tokens": [8192, 32768, 131072],
            "gpu_memory_utilization": [0.92],
        },
        "boundary_expansion": {
            "minimum_gain": 0.03,
            "max_num_seqs": [768, 1024, 1536, 2048],
            "max_num_batched_tokens": [262144, 524288],
        },
        "refinement": {
            "max_num_seqs": [192, 384],
            "max_num_batched_tokens": [16384, 65536],
        },
        "memory": [0.88, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98],
        "partial_prefill": [[1, 1], [2, 1], [2, 2], [4, 2], [4, 4], [8, 4], [8, 8]],
        "workers": [4, 8, 16, 32, 64, 96, 128, 256],
        "successive_halving_requests": [16, 32, 64],
        "hard_gates": {
            "success_rate": 1.0,
            "oom": False,
            "kv_preemption": 0,
            "p95_factor": 1.25,
        },
    }


def select_max_model_len(
    records: Sequence[dict[str, Any]],
    *,
    max_new_tokens: int = 512,
    margin: int = 256,
) -> int:
    lengths = [
        int(record.get("diagnostics", {}).get("prompt_tokens", 0)) for record in records
    ]
    if not lengths or min(lengths) <= 0:
        raise ValueError("every tuning record requires exact diagnostics.prompt_tokens")
    required = max(lengths) + max_new_tokens + margin
    selected = next(
        (value for value in (16384, 32768, 65536) if value >= required), None
    )
    if selected is None:
        raise ValueError(
            f"workload requires context {required}, above the 64K stage limit"
        )
    return selected


def _preemptions(result: dict[str, Any]) -> int:
    aggregate = result.get("backend_delta") or {}
    aggregate_total = sum(
        int(value)
        for key, value in aggregate.items()
        if "preempt" in str(key).lower() and isinstance(value, (int, float))
    )
    if aggregate_total:
        return aggregate_total
    return sum(
        int(value)
        for measurement in result.get("measurements", ())
        for key, value in (
            (measurement.get("diagnostics") or {}).get("backend_delta") or {}
        ).items()
        if "preempt" in str(key).lower() and isinstance(value, (int, float))
    )


def _valid(result: dict[str, Any]) -> bool:
    return (
        result.get("success_rate") == 1.0
        and _preemptions(result) == 0
        and not any(
            "oom" in str(item.get("error", "")).lower()
            for item in result.get("measurements", ())
        )
    )


def select_arms(
    results: Sequence[dict[str, Any]],
    *,
    keep: int,
    p95_factor: float = 1.25,
) -> list[dict[str, Any]]:
    if keep <= 0 or p95_factor < 1:
        raise ValueError("invalid successive-halving gate")
    valid = [result for result in results if _valid(result)]
    if not valid:
        return []
    minimum_p95 = min(result["latency_seconds"]["p95"] for result in valid)
    gated = [
        result
        for result in valid
        if result["latency_seconds"]["p95"] <= minimum_p95 * p95_factor
    ]
    gated.sort(
        key=lambda result: (
            -float(result["jobs_per_second"]),
            float(result["latency_seconds"]["p95"]),
            str(result["arm_id"]),
        )
    )
    return gated[:keep]


def _wait_for_health(endpoint: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"service exited during startup with {process.returncode}"
            )
        try:
            with urllib.request.urlopen(endpoint + "/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as current:
            error = current
        time.sleep(1)
    raise TimeoutError(f"service did not become healthy: {error}")


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def arm_trace_path(log_path: Path) -> Path:
    """Keep each arm's algorithm events beside its immutable service log."""

    return log_path.with_suffix(".trace.jsonl")


def _run_arm(
    *,
    config: EngineConfig,
    feature: FeatureConfig = BASE_FEATURES,
    workers: int,
    records: Sequence[dict[str, Any]],
    warmup_records: Sequence[dict[str, Any]],
    service_config: Path,
    endpoint: str,
    port: int,
    devices: str,
    startup_timeout: float,
    request_timeout: float,
    seed: int,
    log_path: Path,
    routing: str = "round_robin",
    conditional_overrides: dict[str, int] | None = None,
) -> dict[str, Any]:
    trace_path = arm_trace_path(log_path)
    command = [
        sys.executable,
        "-m",
        "inference_scaling.swe_agent.server",
        "--config",
        str(service_config),
        "--port",
        str(port),
        "--set",
        f"service.trace_path={json.dumps(str(trace_path))}",
    ]
    for override in (*config.overrides(), *feature.overrides):
        command.extend(("--set", override))
    environment = dict(os.environ)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = devices
    environment["CIS_INSTANCE_ID"] = f"{config.config_id}-{feature.feature_id}"
    for key, value in feature.environment:
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment
        )
        try:
            _wait_for_health(endpoint, process, startup_timeout)
            if warmup_records:
                warmup = run_burst(
                    warmup_records,
                    (endpoint,),
                    workers=min(workers, len(warmup_records)),
                    timeout=request_timeout,
                    seed=seed,
                    routing=routing,
                    conditional_overrides=conditional_overrides,
                )
                if warmup["success_rate"] != 1.0:
                    raise RuntimeError("warmup workload failed")
            result = run_burst(
                records,
                (endpoint,),
                workers=workers,
                timeout=request_timeout,
                seed=seed,
                routing=routing,
                conditional_overrides=conditional_overrides,
            )
        except Exception as error:
            result = {
                "schema_version": 1,
                "requests": len(records),
                "workers": workers,
                "success_rate": 0.0,
                "jobs_per_second": 0.0,
                "latency_seconds": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0},
                "measurements": [{"error": f"{type(error).__name__}: {error}"}],
            }
        finally:
            _stop(process)
    result.update(
        {
            "arm_id": f"{config.config_id}-{feature.feature_id}-w{workers}",
            "engine": asdict(config),
            "feature": asdict(feature),
            "workers": workers,
            "service_log": str(log_path),
            "algorithm_trace": str(trace_path),
            "started_at": started,
        }
    )
    return result


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cached_arm(
    output: Path,
    phase: str,
    *,
    resume: bool,
    **kwargs: Any,
) -> dict[str, Any]:
    config: EngineConfig = kwargs["config"]
    feature: FeatureConfig = kwargs.get("feature", BASE_FEATURES)
    workers = int(kwargs["workers"])
    requests = len(kwargs["records"])
    conditional = kwargs.get("conditional_overrides") or {}
    algorithm_id = (
        f"c{conditional.get('candidate_count', 'cfg')}"
        f"r{conditional.get('rollout_count', 'cfg')}"
        f"b{conditional.get('block_size', 'cfg')}"
    )
    stem = (
        f"{config.config_id}-{feature.feature_id}-{algorithm_id}-w{workers}-n{requests}"
    )
    result_path = output / "arms" / phase / f"{stem}.json"
    if resume and result_path.exists():
        return json.loads(result_path.read_text())
    kwargs["log_path"] = output / "logs" / phase / f"{stem}.log"
    result = _run_arm(**kwargs)
    _write_checkpoint(result_path, result)
    return result


def _best(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected = select_arms(results, keep=1)
    if not selected:
        raise RuntimeError("no configuration passed the hard gates")
    return selected[0]


def _gain(left: dict[str, Any], right: dict[str, Any]) -> float:
    baseline = float(left["jobs_per_second"])
    if baseline <= 0:
        return float("inf")
    return float(right["jobs_per_second"]) / baseline - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--holdout-workload")
    parser.add_argument("--warmup-workload")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--initial-workers", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=15)
    parser.add_argument("--rollout-count", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[3]
    config_path = Path(args.config).resolve()
    workload_path = Path(args.workload).resolve()
    holdout_path = (
        Path(args.holdout_workload).resolve() if args.holdout_workload else None
    )
    warmup_path = (
        Path(args.warmup_workload).resolve() if args.warmup_workload else None
    )
    records = _load_records(workload_path)
    if len(records) < 64:
        raise ValueError("runtime tuning requires at least 64 unique workload records")
    holdout = (
        _load_records(holdout_path) if holdout_path else []
    )
    warmup = _load_records(warmup_path) if warmup_path else []
    output = Path(args.output_directory)
    max_model_len = select_max_model_len(records)
    plan = adaptive_search_plan()
    plan.update(
        {
            "devices": args.devices,
            "seed": args.seed,
            "source_revision": source_revision(repository),
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "workload": {
                "path": str(workload_path),
                "sha256": sha256_file(workload_path),
                "records": len(records),
            },
            "holdout_workload": (
                {
                    "path": str(holdout_path),
                    "sha256": sha256_file(holdout_path),
                    "records": len(holdout),
                }
                if holdout_path
                else None
            ),
            "warmup_workload": (
                {
                    "path": str(warmup_path),
                    "sha256": sha256_file(warmup_path),
                    "records": len(warmup),
                }
                if warmup_path
                else None
            ),
            "algorithm": {
                "candidate_count": args.candidate_count,
                "rollout_count": args.rollout_count,
                "block_size": args.block_size,
            },
            "selected_max_model_len": max_model_len,
        }
    )
    _write_checkpoint(output / "plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    common = {
        "warmup_records": warmup,
        "service_config": config_path,
        "endpoint": f"http://{args.host}:{args.port}",
        "port": args.port,
        "devices": args.devices,
        "startup_timeout": args.startup_timeout,
        "request_timeout": args.request_timeout,
        "seed": args.seed,
        "conditional_overrides": {
            "candidate_count": args.candidate_count,
            "rollout_count": args.rollout_count,
            "block_size": args.block_size,
        },
    }
    history: list[dict[str, Any]] = []
    baseline_engine = EngineConfig(128, 32768, 0.92, max_model_len=max_model_len)

    feature_results = [
        _cached_arm(
            output,
            "feature-ab",
            resume=args.resume,
            config=baseline_engine,
            feature=feature,
            workers=args.initial_workers,
            records=records[:16],
            **common,
        )
        for feature in feature_ablation_matrix()
    ]
    history.extend(feature_results)
    chosen_feature = FeatureConfig(**_best(feature_results)["feature"])

    survivors = [
        replace(config, max_model_len=max_model_len) for config in engine_grid()
    ]
    for round_index, (request_count, keep) in enumerate(((16, 8), (32, 4), (64, 2)), 1):
        round_results = [
            _cached_arm(
                output,
                f"coarse-{round_index}",
                resume=args.resume,
                config=config,
                feature=chosen_feature,
                workers=args.initial_workers,
                records=records[:request_count],
                **common,
            )
            for config in survivors
        ]
        history.extend(round_results)
        selected = select_arms(round_results, keep=min(keep, len(round_results)))
        survivors = [EngineConfig(**item["engine"]) for item in selected]
        _write_checkpoint(
            output / f"coarse-{round_index}.json",
            {
                "request_count": request_count,
                "results": round_results,
                "selected": selected,
            },
        )
        if not survivors:
            raise RuntimeError(f"no configuration survived coarse round {round_index}")

    current_result = _best(selected)
    current = EngineConfig(**current_result["engine"])
    expansion_results: list[dict[str, Any]] = []
    if current.max_num_seqs == 512:
        for value in (768, 1024, 1536, 2048):
            candidate = replace(current, max_num_seqs=value)
            result = _cached_arm(
                output,
                "expand-mns",
                resume=args.resume,
                config=candidate,
                feature=chosen_feature,
                workers=args.initial_workers,
                records=records[:64],
                **common,
            )
            expansion_results.append(result)
            if not _valid(result) or _gain(current_result, result) < 0.03:
                break
            current, current_result = candidate, result
    if current.max_num_batched_tokens == 131072:
        for value in (262144, 524288):
            candidate = replace(current, max_num_batched_tokens=value)
            result = _cached_arm(
                output,
                "expand-mbt",
                resume=args.resume,
                config=candidate,
                feature=chosen_feature,
                workers=args.initial_workers,
                records=records[:64],
                **common,
            )
            expansion_results.append(result)
            if not _valid(result) or _gain(current_result, result) < 0.03:
                break
            current, current_result = candidate, result
    history.extend(expansion_results)

    refinement_configs = {
        replace(current, max_num_seqs=value) for value in (192, 384)
    } | {replace(current, max_num_batched_tokens=value) for value in (16384, 65536)}
    refinement_results = [
        _cached_arm(
            output,
            "refine",
            resume=args.resume,
            config=config,
            feature=chosen_feature,
            workers=args.initial_workers,
            records=records[:64],
            **common,
        )
        for config in sorted(refinement_configs, key=lambda item: item.config_id)
    ]
    history.extend(refinement_results)
    current_result = _best([current_result, *refinement_results])
    current = EngineConfig(**current_result["engine"])

    memory_results = [
        _cached_arm(
            output,
            "memory",
            resume=args.resume,
            config=replace(current, gpu_memory_utilization=memory),
            feature=chosen_feature,
            workers=args.initial_workers,
            records=records[:64],
            **common,
        )
        for memory in (0.88, 0.90, 0.92, 0.94, 0.96)
    ]
    history.extend(memory_results)
    current_result = _best(memory_results)
    current = EngineConfig(**current_result["engine"])
    if current.gpu_memory_utilization == 0.96:
        upper_memory = []
        for memory in (0.97, 0.98):
            result = _cached_arm(
                output,
                "memory-upper",
                resume=args.resume,
                config=replace(current, gpu_memory_utilization=memory),
                feature=chosen_feature,
                workers=args.initial_workers,
                records=records[:64],
                **common,
            )
            upper_memory.append(result)
            if not _valid(result):
                break
        history.extend(upper_memory)
        current_result = _best([current_result, *upper_memory])
        current = EngineConfig(**current_result["engine"])

    partial_results = [
        _cached_arm(
            output,
            "partial-prefill",
            resume=args.resume,
            config=replace(
                current,
                max_num_partial_prefills=partial,
                max_long_partial_prefills=long_partial,
            ),
            feature=chosen_feature,
            workers=args.initial_workers,
            records=records[:64],
            **common,
        )
        for partial, long_partial in (
            (1, 1),
            (2, 1),
            (2, 2),
            (4, 2),
            (4, 4),
            (8, 4),
            (8, 8),
        )
    ]
    history.extend(partial_results)
    current_result = _best(partial_results)
    current = EngineConfig(**current_result["engine"])

    worker_values = [4, 8, 16, 32, 64]
    worker_results = [
        _cached_arm(
            output,
            "workers",
            resume=args.resume,
            config=current,
            feature=chosen_feature,
            workers=worker,
            records=records[:64],
            **common,
        )
        for worker in worker_values
    ]
    for next_worker in (96, 128, 256):
        previous = _best(worker_results)
        if int(previous["workers"]) != worker_values[-1] or len(records) < next_worker:
            break
        result = _cached_arm(
            output,
            "workers",
            resume=args.resume,
            config=current,
            feature=chosen_feature,
            workers=next_worker,
            records=records[:next_worker],
            **common,
        )
        worker_values.append(next_worker)
        worker_results.append(result)
        if not _valid(result) or _gain(previous, result) < 0.03:
            break
    history.extend(worker_results)
    finalists = select_arms(worker_results, keep=2)

    holdout_results = []
    if holdout:
        for finalist in finalists:
            holdout_results.append(
                _cached_arm(
                    output,
                    "holdout",
                    resume=args.resume,
                    config=EngineConfig(**finalist["engine"]),
                    feature=FeatureConfig(**finalist["feature"]),
                    workers=int(finalist["workers"]),
                    records=holdout[:64],
                    **common,
                )
            )
        history.extend(holdout_results)
    winner = _best(holdout_results or finalists)
    result = {
        **plan,
        "history": history,
        "chosen_feature": asdict(chosen_feature),
        "winner": winner,
    }
    _write_checkpoint(output / "result.json", result)
    write_artifact_manifest(
        output,
        repository=repository,
        command=sys.argv,
        metadata={
            "kind": "two-card-runtime-tuning",
            "devices": args.devices,
            "source_revision": plan["source_revision"],
            "config": plan["config"],
            "workload": plan["workload"],
            "holdout_workload": plan["holdout_workload"],
            "warmup_workload": plan["warmup_workload"],
            "algorithm": plan["algorithm"],
            "winner": winner["arm_id"],
        },
    )
    print(json.dumps({"winner": winner}, indent=2))


if __name__ == "__main__":
    main()
