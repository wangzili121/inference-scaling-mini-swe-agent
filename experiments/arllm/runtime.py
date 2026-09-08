"""Shared artifact validation for real-model AR experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.shared.artifacts import (
    adapter_hashes,
    cached_file_sha256,
    checkpoint_metadata_hashes,
    implementation_hashes,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HASH_CACHE = REPOSITORY_ROOT / ".cache" / "artifact_hashes"


def source_hashes(entrypoints: Iterable[str | Path]) -> dict[str, str]:
    return implementation_hashes(REPOSITORY_ROOT, entrypoints=entrypoints)


def set_rl_adapter_override(config: dict[str, Any], adapter: Path | None) -> None:
    """Point an AR evaluation config at an adapter produced by the same suite."""

    if adapter is None:
        return
    models = config["models"]
    models["rl"] = str(adapter)
    models["rl_source"] = "local GRPO adapter from the current reproduction suite"
    models["rl_revision"] = "suite-output"
    models["rl_kind"] = "peft_adapter"


def validate_model_artifacts(
    config: Mapping[str, Any],
    roles: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Hash every file that determines the requested AR model roles."""

    models = config["models"]
    requested = set(roles)
    unknown = requested - {"base", "proposal", "rl"}
    if unknown:
        raise ValueError(f"unknown AR model roles: {sorted(unknown)}")

    weights: dict[str, str] = {}
    metadata: dict[str, dict[str, str]] = {}
    adapters: dict[str, dict[str, str]] = {}
    for role in sorted(requested & {"base", "proposal"}):
        directory = Path(str(models[role]))
        weight = directory / "model.safetensors"
        expected_key = f"{role}_weight_sha256"
        digest = cached_file_sha256(
            weight,
            cache_directory=HASH_CACHE,
            expected=(str(models[expected_key]) if expected_key in models else None),
        )
        weights[role] = digest
        metadata[role] = checkpoint_metadata_hashes(directory)

    if "rl" in requested:
        directory = Path(str(models["rl"]))
        if str(models.get("rl_kind", "full_model")) == "peft_adapter":
            adapter_manifest = adapter_hashes(directory)
            adapters["rl"] = adapter_manifest
            weights["rl_adapter"] = adapter_manifest["adapter_model.safetensors"]
        else:
            weight = directory / "model.safetensors"
            weights["rl"] = cached_file_sha256(
                weight,
                cache_directory=HASH_CACHE,
            )
            metadata["rl"] = checkpoint_metadata_hashes(directory)

    return {
        "weight_sha256": weights,
        "metadata_sha256": metadata,
        "adapter_sha256": adapters,
    }


__all__ = [
    "REPOSITORY_ROOT",
    "set_rl_adapter_override",
    "source_hashes",
    "validate_model_artifacts",
]
