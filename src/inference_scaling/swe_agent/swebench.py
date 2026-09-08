"""Launch pinned mini-SWE-agent on deterministic SWE-bench Verified tasks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Sequence


PINNED_MINI_SWE_AGENT_VERSION = "2.4.6"
VERIFIED_DATASET = "princeton-nlp/SWE-Bench_Verified"


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


def _instance_filter(instance_ids: Sequence[str]) -> str:
    return "^(?:" + "|".join(re.escape(value) for value in instance_ids) + ")$"


def build_swebench_command(
    *,
    overlay_config: Path,
    output: Path,
    endpoint: str,
    count: int,
    workers: int,
    instance_ids: Sequence[str] = (),
    redo_existing: bool = False,
) -> list[str]:
    if not 1 <= count <= 500:
        raise ValueError("count must lie in [1, 500]")
    if workers <= 0:
        raise ValueError("workers must be positive")
    command = [
        sys.executable,
        "-m",
        "minisweagent.run.benchmarks.swebench",
        "--subset",
        "verified",
        "--split",
        "test",
        "--output",
        str(output),
        "--workers",
        str(workers),
        "--config",
        "swebench.yaml",
        "--config",
        str(overlay_config),
        "--config",
        f"model.endpoint={endpoint.rstrip('/')}",
        "--model-class",
        "inference_scaling.swe_agent.model.ConditionalISModel",
    ]
    if instance_ids:
        command.extend(("--filter", _instance_filter(instance_ids)))
    else:
        command.extend(("--slice", f"0:{count}"))
    if redo_existing:
        command.append("--redo-existing")
    return command


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
    command = build_swebench_command(
        overlay_config=overlay,
        output=output,
        endpoint=args.endpoint,
        count=args.count,
        workers=args.workers,
        instance_ids=args.instance_id,
        redo_existing=args.redo_existing,
    )
    manifest = {
        "schema_version": 1,
        "dataset": VERIFIED_DATASET,
        "split": "test",
        "mini_swe_agent_version": installed,
        "conditional_is_endpoint": args.endpoint.rstrip("/"),
        "instance_ids": list(args.instance_id),
        "count": len(args.instance_id) if args.instance_id else args.count,
        "workers": args.workers,
        "command": command,
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
    environment = dict(os.environ)
    environment.setdefault("MSWEA_SILENT_STARTUP", "1")
    result = subprocess.run(command, check=False, env=environment)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
