"""Launch a reproducible v0.18 Conditional IS service container on free NPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence


DEFAULT_IMAGE = "quay.io/ascend/vllm-ascend:v0.18.0"
_CATEGORICAL_MOUNTS = (
    (
        "runtime/sampler.py",
        "/vllm-workspace/vllm-ascend/vllm_ascend/sample/sampler.py",
    ),
    (
        "vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so",
        "/vllm-workspace/vllm-ascend/vllm_ascend/"
        "vllm_ascend_C.cpython-311-aarch64-linux-gnu.so",
    ),
    (
        "vllm_ascend/libvllm_ascend_kernels.so",
        "/vllm-workspace/vllm-ascend/vllm_ascend/libvllm_ascend_kernels.so",
    ),
    (
        "vllm_ascend/_cann_ops_custom",
        "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom",
    ),
)


def parse_npu_processes(output: str) -> dict[int, list[int]]:
    processes: dict[int, list[int]] = {}
    in_process_table = False
    for line in output.splitlines():
        if "Process id" in line and "Process name" in line:
            in_process_table = True
            continue
        if not in_process_table:
            continue
        match = re.match(
            r"\|\s*(\d+)\s+\d+\s+\|\s*(\d+)\s+\|", line
        )
        if match:
            processes.setdefault(int(match.group(1)), []).append(int(match.group(2)))
    return processes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _categorical_assets(root: Path) -> list[dict[str, Any]]:
    assets = []
    for relative, target in _CATEGORICAL_MOUNTS:
        source = root / relative
        if not source.exists():
            raise FileNotFoundError(source)
        assets.append(
            {
                "source": str(source),
                "target": target,
                "sha256": _sha256(source) if source.is_file() else _tree_sha256(source),
            }
        )
    return assets


def build_docker_command(
    *,
    image: str,
    name: str,
    repository: Path,
    model: Path,
    config: Path,
    categorical_root: Path,
    cache_root: Path,
    devices: Sequence[int],
    port: int,
    artifact_root: Path | None = None,
    overrides: Sequence[str] = (),
) -> tuple[list[str], list[dict[str, Any]]]:
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be a nonempty unique list")
    assets = _categorical_assets(categorical_root)
    command = [
        "docker",
        "run",
        "--detach",
        "--network",
        "host",
        "--ipc",
        "host",
        "--name",
        name,
    ]
    for device in devices:
        command.extend(("--device", f"/dev/davinci{device}"))
    for path in (
        "/dev/davinci_manager",
        "/dev/devmm_svm",
        "/dev/hisi_hdc",
    ):
        command.extend(("--device", path))
    for source, target in (
        ("/usr/local/Ascend/driver/lib64", "/usr/local/Ascend/driver/lib64"),
        ("/usr/local/Ascend/driver/version.info", "/usr/local/Ascend/driver/version.info"),
        ("/etc/ascend_install.info", "/etc/ascend_install.info"),
        ("/usr/local/dcmi", "/usr/local/dcmi"),
        ("/usr/local/bin/npu-smi", "/usr/local/bin/npu-smi"),
    ):
        command.extend(("--volume", f"{source}:{target}:ro"))
    command.extend(("--volume", f"{repository}:/workspace"))
    command.extend(("--volume", f"{model}:/models/conditional-is:ro"))
    command.extend(("--volume", f"{cache_root}:/root/.cache/vllm"))
    if artifact_root is not None:
        command.extend(("--volume", f"{artifact_root}:/artifacts"))
    for asset in assets:
        command.extend(
            ("--volume", f"{asset['source']}:{asset['target']}:ro")
        )
    custom_opp = (
        "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/"
        "vendors/vllm-ascend"
    )
    command.extend(
        (
            "--env",
            "ASCEND_RT_VISIBLE_DEVICES=" + ",".join(map(str, devices)),
            "--env",
            "CIS_MODEL_PATH=/models/conditional-is",
            "--env",
            "VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE=1",
            "--env",
            f"ASCEND_CUSTOM_OPP_PATH={custom_opp}",
            "--env",
            f"LD_PRELOAD={custom_opp}/op_api/lib/libcust_opapi.so",
            "--workdir",
            "/workspace",
            "--entrypoint",
            "bash",
            image,
            "-lc",
            (
                "export PYTHONPATH=/workspace/src:/workspace"
                "${PYTHONPATH:+:${PYTHONPATH}}; "
                'exec python -m inference_scaling.swe_agent.server "$@"'
            ),
            "conditional-is-server",
            "--config",
            "/workspace/" + str(config.relative_to(repository)),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        )
    )
    for override in overrides:
        command.extend(("--set", override))
    return command, assets


def _git_revision(repository: Path) -> str | None:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    snapshot_revision = repository / ".source-commit"
    if snapshot_revision.is_file():
        return snapshot_revision.read_text(encoding="utf-8").strip() or None
    return None


def _image_id(image: str) -> str | None:
    result = subprocess.run(
        ("docker", "image", "inspect", "--format", "{{.Id}}", image),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _model_metadata(model: Path) -> dict[str, str]:
    result = {}
    for name in ("config.json", "model.safetensors.index.json"):
        path = model / name
        if path.is_file():
            result[name] = _sha256(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=str(Path.cwd()))
    parser.add_argument("--model", required=True)
    parser.add_argument("--categorical-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument(
        "--artifact-root",
        help=(
            "host directory mounted read-write at /artifacts; when supplied, "
            "the default algorithm trace is stored below this directory"
        ),
    )
    parser.add_argument(
        "--config", default="configs/swebench/conditional_is_smoke.toml"
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--name", default="conditional-is-swebench")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--skip-device-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repository = Path(args.repository).resolve()
    model = Path(args.model).resolve()
    config = (repository / args.config).resolve()
    categorical_root = Path(args.categorical_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    artifact_root = Path(args.artifact_root).resolve() if args.artifact_root else None
    devices = tuple(int(value) for value in args.devices.split(","))
    for required in (repository, model, config, categorical_root):
        if not required.exists():
            raise FileNotFoundError(required)
    if repository not in config.parents:
        raise ValueError("service config must be inside the mounted repository")
    if not args.skip_device_check:
        result = subprocess.run(
            ("/usr/local/bin/npu-smi", "info"),
            check=True,
            capture_output=True,
            text=True,
        )
        busy = parse_npu_processes(result.stdout)
        conflicts = {device: busy[device] for device in devices if busy.get(device)}
        if conflicts:
            raise RuntimeError(f"refusing to use NPUs with existing processes: {conflicts}")
    overrides = list(args.overrides)
    trace_path = None
    if artifact_root is not None:
        trace_path = "/artifacts/algorithm-traces/model_calls.jsonl"
        if not any(value.startswith("service.trace_path=") for value in overrides):
            overrides.append(f"service.trace_path={json.dumps(trace_path)}")
    command, assets = build_docker_command(
        image=args.image,
        name=args.name,
        repository=repository,
        model=model,
        config=config,
        categorical_root=categorical_root,
        cache_root=cache_root,
        artifact_root=artifact_root,
        devices=devices,
        port=args.port,
        overrides=overrides,
    )
    manifest = {
        "schema_version": 1,
        "git_revision": _git_revision(repository),
        "image": args.image,
        "image_id": _image_id(args.image),
        "repository": str(repository),
        "model": str(model),
        "model_metadata_sha256": _model_metadata(model),
        "config": str(config),
        "config_sha256": _sha256(config),
        "categorical_assets": assets,
        "artifact_root": str(artifact_root) if artifact_root is not None else None,
        "container_trace_path": trace_path,
        "devices": list(devices),
        "port": args.port,
        "overrides": overrides,
        "command": command,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    cache_root.mkdir(parents=True, exist_ok=True)
    if artifact_root is not None:
        (artifact_root / "algorithm-traces").mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
