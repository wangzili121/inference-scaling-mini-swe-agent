"""Evaluate a pinned Conditional IS mini-SWE-agent run with SWE-bench."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from inference_scaling.swe_agent.swebench import (
    VERIFIED_DATASET,
    VERIFIED_REVISION,
    _docker_preflight,
)


PINNED_SWEBENCH_VERSION = "5.0.2"


def _validate_run(run_directory: Path) -> tuple[Path, list[str]]:
    manifest_path = run_directory / "conditional_is_run_manifest.json"
    predictions_path = run_directory / "preds.json"
    manifest = json.loads(manifest_path.read_text())
    predictions = json.loads(predictions_path.read_text())
    if manifest.get("dataset") != VERIFIED_DATASET:
        raise ValueError("run manifest does not target SWE-bench Verified")
    if manifest.get("dataset_revision") != VERIFIED_REVISION:
        raise ValueError("run manifest uses a different Verified revision")
    instance_ids = [str(value) for value in manifest.get("instance_ids", ())]
    if not isinstance(predictions, dict):
        raise ValueError("mini-SWE-agent predictions must be an object keyed by task")
    missing = [value for value in instance_ids if value not in predictions]
    if missing:
        raise ValueError("predictions are missing tasks: " + ", ".join(missing))
    for instance_id in instance_ids:
        prediction = predictions[instance_id]
        if not isinstance(prediction, dict) or prediction.get("instance_id") != instance_id:
            raise ValueError(f"invalid prediction record for {instance_id}")
    return predictions_path, instance_ids


def build_evaluation_command(
    *,
    dataset_snapshot: Path,
    predictions: Path,
    instance_ids: Sequence[str],
    run_id: str,
    report_directory: Path,
    workers: int,
    timeout: int,
    open_file_limit: int,
) -> list[str]:
    if workers <= 0 or timeout <= 0 or open_file_limit <= 0:
        raise ValueError("evaluation workers, timeout, and file limit must be positive")
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(dataset_snapshot),
        "--split",
        "test",
        "--predictions_path",
        str(predictions),
        "--max_workers",
        str(workers),
        "--timeout",
        str(timeout),
        "--open_file_limit",
        str(open_file_limit),
        "--run_id",
        run_id,
        "--report_dir",
        str(report_directory),
        "--instance_ids",
        *instance_ids,
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--report-directory", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--workers", type=int, default=max(1, min(24, (os.cpu_count() or 2) * 3 // 4)))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--open-file-limit", type=int, default=4096)
    parser.add_argument("--skip-docker-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    installed = importlib.metadata.version("swebench")
    if installed != PINNED_SWEBENCH_VERSION:
        raise RuntimeError(
            f"SWE-bench version mismatch: expected {PINNED_SWEBENCH_VERSION}, "
            f"found {installed}"
        )
    run_directory = Path(args.run_directory).resolve()
    report_directory = Path(args.report_directory).resolve()
    predictions, instance_ids = _validate_run(run_directory)
    run_id = args.run_id or run_directory.name
    from datasets import load_dataset

    dataset = load_dataset(
        VERIFIED_DATASET, revision=VERIFIED_REVISION, split="test"
    )
    by_id = {str(item["instance_id"]): dict(item) for item in dataset}
    dataset_snapshot = report_directory / f"{run_id}.dataset.json"
    selected_dataset = [by_id[instance_id] for instance_id in instance_ids]
    command = build_evaluation_command(
        dataset_snapshot=dataset_snapshot,
        predictions=predictions,
        instance_ids=instance_ids,
        run_id=run_id,
        report_directory=report_directory,
        workers=args.workers,
        timeout=args.timeout,
        open_file_limit=args.open_file_limit,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "swebench_version": installed,
        "dataset": VERIFIED_DATASET,
        "dataset_revision": VERIFIED_REVISION,
        "dataset_snapshot": str(dataset_snapshot),
        "run_id": run_id,
        "instance_ids": instance_ids,
        "predictions": str(predictions),
        "command": command,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if not args.skip_docker_preflight:
        _docker_preflight()
    report_directory.mkdir(parents=True, exist_ok=True)
    dataset_snapshot.write_text(
        json.dumps(selected_dataset, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (report_directory / f"{run_id}.evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
