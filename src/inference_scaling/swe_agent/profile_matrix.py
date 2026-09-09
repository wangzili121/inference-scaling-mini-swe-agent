"""Run the fixed P0-P3 matrix on selected two-card and four-card deployments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from inference_scaling.swe_agent.profile_report import render_profile_report


def load_matrix(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("rb") as stream:
        payload = tomllib.load(stream)
    algorithms = payload.get("algorithm")
    if not isinstance(algorithms, list) or not algorithms:
        raise ValueError("profile matrix requires at least one [[algorithm]]")
    required = {
        "id",
        "candidate_count",
        "rollout_count",
        "block_size",
        "max_new_tokens",
        "reward",
    }
    for algorithm in algorithms:
        missing = sorted(required - set(algorithm))
        if missing:
            raise ValueError(f"algorithm profile is missing: {', '.join(missing)}")
    return [dict(item) for item in algorithms]


def load_deployment(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    required = {
        "id",
        "devices",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "routing",
        "workers",
        "overrides",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"deployment is missing: {', '.join(missing)}")
    if len(payload["devices"]) not in {1, 2}:
        raise ValueError("deployment must describe one or two service instances")
    return payload


def profile_commands(
    *,
    config: Path,
    matrix: list[dict[str, Any]],
    deployments: list[dict[str, Any]],
    workload: Path,
    warmup_workload: Path | None,
    output: Path,
    seed: int,
    categorical_root: Path | None = None,
) -> list[dict[str, Any]]:
    planned = []
    for deployment in deployments:
        for algorithm in matrix:
            for profiler in ("none", "service", "torch"):
                repeats = 3 if profiler == "none" and algorithm["id"] == "P0" else 1
                for repeat in range(1, repeats + 1):
                    run_directory = (
                        output
                        / str(deployment["id"])
                        / str(algorithm["id"])
                        / f"{profiler}-{repeat}"
                    )
                    command = [
                        sys.executable,
                        "-m",
                        "inference_scaling.swe_agent.profile",
                        "--config",
                        str(config),
                        "--workload",
                        str(workload),
                        "--output-directory",
                        str(run_directory),
                        "--profiler",
                        profiler,
                        "--workers",
                        str(deployment["workers"]),
                        "--limit",
                        str(deployment.get("limit", 64)),
                        "--tensor-parallel-size",
                        str(deployment["tensor_parallel_size"]),
                        "--pipeline-parallel-size",
                        str(deployment["pipeline_parallel_size"]),
                        "--routing",
                        str(deployment["routing"]),
                        "--seed",
                        str(seed),
                        "--candidate-count",
                        str(algorithm["candidate_count"]),
                        "--rollout-count",
                        str(algorithm["rollout_count"]),
                        "--block-size",
                        str(algorithm["block_size"]),
                        "--set",
                        f"generation.max_new_tokens={algorithm['max_new_tokens']}",
                        "--set",
                        f"reward.kind={json.dumps(algorithm['reward'])}",
                    ]
                    for devices in deployment["devices"]:
                        command.extend(("--devices", str(devices)))
                    if categorical_root is not None:
                        command.extend(("--categorical-root", str(categorical_root)))
                    for override in deployment["overrides"]:
                        command.extend(("--set", str(override)))
                    for assignment in deployment.get("environment", ()):
                        command.extend(("--env", str(assignment)))
                    if warmup_workload is not None:
                        command.extend(("--warmup-workload", str(warmup_workload)))
                    if profiler == "service":
                        command.extend(("--profile-seconds", "60"))
                    elif profiler == "torch":
                        command.extend(("--profile-seconds", "10"))
                    planned.append(
                        {
                            "deployment": deployment["id"],
                            "algorithm": algorithm["id"],
                            "profiler": profiler,
                            "repeat": repeat,
                            "output": str(run_directory),
                            "command": command,
                        }
                    )
    return planned


def render_matrix_summary(output: Path, index: dict[str, Any]) -> dict[str, str]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in index["runs"]:
        grouped.setdefault((run["deployment"], run["algorithm"]), []).append(run)
    rows = []
    report = [
        "# Conditional IS General 部署与 Profiling 总报告",
        "",
        "> 性能数字只来自 `none` pass；service/torch pass 只用于归因 profiler 扰动。",
        "",
        "| 部署 | 算法 | jobs/s | P95(s) | 重复噪声 | block gap | NPU busy | 暴露 HCCL |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (deployment, algorithm), runs in sorted(grouped.items()):
        unprofiled = []
        torch_analysis = None
        for run in runs:
            benchmark_path = Path(run["output"]) / "benchmark.json"
            if not benchmark_path.exists():
                continue
            benchmark = json.loads(benchmark_path.read_text())
            if run["profiler"] == "none":
                unprofiled.append(benchmark)
            elif run["profiler"] == "torch":
                analysis_path = Path(run["output"]) / "derived" / "analysis.json"
                if analysis_path.exists():
                    torch_analysis = json.loads(analysis_path.read_text())
        throughputs = [float(item["jobs_per_second"]) for item in unprofiled]
        p95_values = [float(item["latency_seconds"]["p95"]) for item in unprofiled]
        mean_throughput = statistics.fmean(throughputs) if throughputs else 0.0
        noise = (
            statistics.pstdev(throughputs) / mean_throughput
            if len(throughputs) > 1 and mean_throughput
            else 0.0
        )
        algorithm_analysis = (torch_analysis or {}).get("algorithm", {})
        ascend = (torch_analysis or {}).get("ascend", {})
        row = {
            "deployment": deployment,
            "algorithm": algorithm,
            "unprofiled_repeats": len(unprofiled),
            "jobs_per_second_mean": mean_throughput,
            "p95_seconds_mean": statistics.fmean(p95_values) if p95_values else 0.0,
            "throughput_noise": noise,
            "block_gap_share": float(algorithm_analysis.get("block_gap_share", 0.0)),
            "npu_busy_ratio_median": float(
                ascend.get("device_busy_ratio", {}).get("median", 0.0)
            ),
            "exposed_communication_ratio": float(
                ascend.get("exposed_communication_ratio", 0.0)
            ),
        }
        rows.append(row)
        report.append(
            f"| {deployment} | {algorithm} | {row['jobs_per_second_mean']:.4f} | "
            f"{row['p95_seconds_mean']:.3f} | {row['throughput_noise']:.2%} | "
            f"{row['block_gap_share']:.2%} | {row['npu_busy_ratio_median']:.2%} | "
            f"{row['exposed_communication_ratio']:.2%} |"
        )
    report.extend(
        [
            "",
            "## 解读规则",
            "",
            "- 同一部署中比较 P0/P1/P2，判断瓶颈是否随 C/R 压力迁移；P2 不能用于单独归因 R。",
            "- P0/P3 只改变 reward 路径，用来验证 generation statistics 后 reward 是否仍是主要开销。",
            "- 双卡/四卡比较必须同时报告 jobs/s、实例偏斜、HCCL 暴露和 queue/batch 证据。",
            "- 重复噪声超过 5% 的单元必须追加无 profiler 复验，不能据此给出参数胜负。",
            "- 定向优化建议必须引用对应 run 的原始 trace 和 `derived/analysis.json`，本报告不把阈值触发等同于已验证收益。",
            "",
            "## 原始数据索引",
            "",
            "完整命令和目录见 `matrix-plan.json`；每个目录的 `artifact-manifest.json` 可校验原始数据。",
        ]
    )
    csv_path = output / "matrix-summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report_path = output / "FINAL_REPORT.zh-CN.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--matrix", default="configs/swebench/profile_matrix.toml")
    parser.add_argument("--deployment", action="append", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--warmup-workload")
    parser.add_argument("--categorical-root")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    planned = profile_commands(
        config=Path(args.config).resolve(),
        matrix=load_matrix(args.matrix),
        deployments=[load_deployment(path) for path in args.deployment],
        workload=Path(args.workload).resolve(),
        warmup_workload=(
            Path(args.warmup_workload).resolve() if args.warmup_workload else None
        ),
        output=output,
        seed=args.seed,
        categorical_root=(
            Path(args.categorical_root).resolve() if args.categorical_root else None
        ),
    )
    index = {
        "schema_version": 1,
        "deployments": [load_deployment(path) for path in args.deployment],
        "algorithms": load_matrix(args.matrix),
        "runs": planned,
    }
    (output / "matrix-plan.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(index, indent=2))
        return

    for run in planned:
        run_output = Path(run["output"])
        benchmark = run_output / "benchmark.json"
        if args.resume and benchmark.exists():
            continue
        run_output.mkdir(parents=True, exist_ok=True)
        with (run_output / "launcher.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                run["command"], stdout=log, stderr=subprocess.STDOUT
            )
        run["returncode"] = completed.returncode
        if completed.returncode != 0:
            (output / "matrix-plan.json").write_text(
                json.dumps(index, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError(
                f"profile failed for {run['deployment']}/{run['algorithm']}/{run['profiler']}"
            )
        run["derived"] = render_profile_report(run_output)
        (output / "matrix-plan.json").write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8"
        )
    index["summary"] = render_matrix_summary(output, index)
    (output / "matrix-plan.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
