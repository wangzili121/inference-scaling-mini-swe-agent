"""Reproducibility manifests and checksums for tuning/profile artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repository: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ("git", *args),
        cwd=repository,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def source_revision(repository: Path) -> str | None:
    revision = _git(repository, "rev-parse", "HEAD")
    if revision:
        return revision
    snapshot = repository / ".source-commit"
    if snapshot.is_file():
        return snapshot.read_text(encoding="utf-8").strip() or None
    return None


def _versions(names: Iterable[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def artifact_index(
    root: Path, *, excluded: Iterable[Path] = ()
) -> list[dict[str, Any]]:
    excluded_resolved = {path.resolve() for path in excluded}
    result = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() in excluded_resolved:
            continue
        result.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def write_artifact_manifest(
    output_directory: str | Path,
    *,
    repository: str | Path,
    command: Iterable[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output = Path(output_directory).resolve()
    repository_path = Path(repository).resolve()
    manifest_path = output / "artifact-manifest.json"
    payload = {
        "schema_version": 1,
        "created_at": time.time(),
        "repository": str(repository_path),
        "git": {
            "commit": source_revision(repository_path),
            "branch": _git(repository_path, "branch", "--show-current"),
            "status": _git(repository_path, "status", "--short"),
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "packages": _versions(
            (
                "inference-scaling",
                "vllm",
                "vllm-ascend",
                "torch",
                "torch-npu",
                "transformers",
                "msserviceprofiler",
                "tzdata",
            )
        ),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "HCCL_OP_EXPANSION_MODE",
                "VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE",
                "PYTORCH_NPU_ALLOC_CONF",
                "TASK_QUEUE_ENABLE",
                "CPU_AFFINITY_CONF",
            )
        },
        "command": list(command),
        "metadata": metadata,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["artifacts"] = artifact_index(output, excluded=(manifest_path,))
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["manifest_sha256"] = sha256_file(manifest_path)
    return payload


def refresh_artifact_manifest(output_directory: str | Path) -> dict[str, Any]:
    """Refresh artifact hashes without replacing collection-time metadata."""

    output = Path(output_directory).resolve()
    manifest_path = output / "artifact-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"] = artifact_index(output, excluded=(manifest_path,))
    payload.pop("manifest_sha256", None)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["manifest_sha256"] = sha256_file(manifest_path)
    return payload
