"""Apply the guarded vLLM 0.18 tiled selected-token reference patch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import subprocess
from pathlib import Path


UPSTREAM_VERSION = "0.18.0"
UPSTREAM_SHA256 = "e164efbb988bc23ddff83842d27930a8cc904104d3c74d28c8c9369535b08f01"
TARGET_MODULE = "vllm.v1.worker.gpu_model_runner"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply(*, check: bool = False) -> Path:
    version = importlib.metadata.version("vllm")
    if version != UPSTREAM_VERSION:
        raise RuntimeError(f"expected vLLM {UPSTREAM_VERSION}, found {version}")
    spec = importlib.util.find_spec(TARGET_MODULE)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate {TARGET_MODULE}")
    target = Path(spec.origin).resolve()
    current = _sha256(target)
    source = target.read_text(encoding="utf-8")
    marker = "VLLM_TILED_SELECTED_TOKEN_LOGPROBS"
    if marker in source:
        return target
    if current != UPSTREAM_SHA256:
        raise RuntimeError(
            "refusing to patch an unknown gpu_model_runner.py: "
            f"sha256={current}"
        )
    if check:
        return target
    patch = Path(__file__).with_name("vllm-0.18-tiled-selected-token.patch")
    package_root = target.parents[3]
    subprocess.run(
        ["patch", "--forward", "--batch", "-p1", "-i", str(patch)],
        cwd=package_root,
        check=True,
    )
    if marker not in target.read_text(encoding="utf-8"):
        raise RuntimeError("patch command completed without installing the guard")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(apply(check=args.check))


if __name__ == "__main__":
    main()
