"""Launch pinned mini-SWE-agent on deterministic SWE-bench Verified tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Sequence


PINNED_MINI_SWE_AGENT_VERSION = "2.4.6"
VERIFIED_DATASET = "princeton-nlp/SWE-Bench_Verified"
VERIFIED_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _wait_for_service(endpoint: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                endpoint.rstrip("/") + "/healthz", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except Exception as current:
            error = current
        time.sleep(1)
    raise TimeoutError(f"Conditional IS service is not healthy: {error}")


def select_instances(
    instances: Sequence[dict],
    *,
    count: int,
    instance_ids: Sequence[str] = (),
) -> list[dict]:
    if not 1 <= count <= 500:
        raise ValueError("count must lie in [1, 500]")
    if not instance_ids:
        return list(instances[:count])
    by_id = {str(item["instance_id"]): item for item in instances}
    missing = [value for value in instance_ids if value not in by_id]
    if missing:
        raise ValueError("unknown Verified instance IDs: " + ", ".join(missing))
    return [by_id[value] for value in instance_ids]


def apply_image_template(
    instances: Sequence[dict], template: str | None
) -> list[dict]:
    selected = [dict(item) for item in instances]
    if not template:
        return selected
    for item in selected:
        instance_id = str(item["instance_id"])
        item["image_name"] = template.format(
            instance_id=instance_id,
            instance_id_dash=instance_id.replace("__", "-"),
        )
    return selected


def _run_batch(
    instances: Sequence[dict],
    *,
    overlay_config: Path,
    endpoint: str,
    output: Path,
    workers: int,
    redo_existing: bool,
) -> None:
    from minisweagent.config import get_config_from_spec
    from minisweagent.run.benchmarks.swebench import process_instance
    from minisweagent.run.benchmarks.utils.batch_progress import (
        RunBatchProgressManager,
    )
    from minisweagent.utils.log import add_file_handler, logger
    from minisweagent.utils.serialize import recursive_merge
    from rich.live import Live

    output.mkdir(parents=True, exist_ok=True)
    add_file_handler(output / "minisweagent.log")
    selected = list(instances)
    if not redo_existing and (output / "preds.json").exists():
        existing = set(json.loads((output / "preds.json").read_text()))
        selected = [item for item in selected if item["instance_id"] not in existing]
    config = recursive_merge(
        get_config_from_spec("swebench.yaml"),
        get_config_from_spec(str(overlay_config)),
        {
            "model": {
                "endpoint": endpoint.rstrip("/"),
                "model_class": "inference_scaling.swe_agent.model.ConditionalISModel",
            }
        },
    )
    progress = RunBatchProgressManager(
        len(selected), output / f"exit_statuses_{time.time()}.yaml"
    )

    def consume(futures: dict[concurrent.futures.Future, str]) -> None:
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as error:
                instance_id = futures[future]
                logger.error(
                    "Error in future for %s: %s",
                    instance_id,
                    error,
                    exc_info=True,
                )
                progress.on_uncaught_exception(instance_id, error)

    with Live(progress.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, item, output, config, progress): str(
                    item["instance_id"]
                )
                for item in selected
            }
            try:
                consume(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending SWE-bench tasks")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                consume(futures)


def _docker_preflight() -> None:
    try:
        result = subprocess.run(
            ("docker", "info", "--format", "{{json .ServerVersion}}"),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Docker preflight failed: {error}") from error
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Docker is unavailable: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            _repository_root() / "configs" / "mini_swe_agent" / "conditional_is.yaml"
        ),
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8123")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument(
        "--image-template",
        help=(
            "optional Docker image template with {instance_id} and "
            "{instance_id_dash} fields"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--service-timeout", type=float, default=30.0)
    parser.add_argument("--redo-existing", action="store_true")
    parser.add_argument("--skip-docker-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    installed = importlib.metadata.version("mini-swe-agent")
    if installed != PINNED_MINI_SWE_AGENT_VERSION:
        raise RuntimeError(
            "mini-SWE-agent version mismatch: "
            f"expected {PINNED_MINI_SWE_AGENT_VERSION}, found {installed}"
        )
    overlay = Path(args.config).resolve()
    if not overlay.is_file():
        raise FileNotFoundError(overlay)
    output = Path(args.output).resolve()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    from datasets import load_dataset

    dataset = load_dataset(
        VERIFIED_DATASET,
        revision=VERIFIED_REVISION,
        split="test",
    )
    instances = apply_image_template(
        select_instances(
            list(dataset), count=args.count, instance_ids=args.instance_id
        ),
        args.image_template,
    )
    selected_ids = [str(item["instance_id"]) for item in instances]
    manifest = {
        "schema_version": 1,
        "dataset": VERIFIED_DATASET,
        "dataset_revision": VERIFIED_REVISION,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "split": "test",
        "mini_swe_agent_version": installed,
        "conditional_is_endpoint": args.endpoint.rstrip("/"),
        "instance_ids": selected_ids,
        "images": {
            str(item["instance_id"]): item.get("image_name")
            for item in instances
        },
        "count": len(selected_ids),
        "workers": args.workers,
        "argv": sys.argv,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    _wait_for_service(args.endpoint, args.service_timeout)
    if not args.skip_docker_preflight:
        _docker_preflight()
    output.mkdir(parents=True, exist_ok=True)
    (output / "conditional_is_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    _run_batch(
        instances,
        overlay_config=overlay,
        endpoint=args.endpoint,
        output=output,
        workers=args.workers,
        redo_existing=args.redo_existing,
    )


if __name__ == "__main__":
    main()
