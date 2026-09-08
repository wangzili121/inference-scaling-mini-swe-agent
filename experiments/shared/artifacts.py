"""Stable hashing, resumable JSONL, and experiment-artifact helpers."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping


def file_sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def cached_file_sha256(
    path: Path,
    *,
    cache_directory: Path,
    expected: str | None = None,
) -> str:
    """Hash a large immutable artifact once and validate its full file identity.

    The cache is accepted only when absolute path, size, modification time, and
    metadata-change time all match.  A changed identity triggers a complete
    SHA-256 pass before the digest is returned.
    """

    artifact = path.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    stat = artifact.stat()
    identity = {
        "path": str(artifact),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }
    cache_key = hashlib.sha256(str(artifact).encode("utf-8")).hexdigest()
    cache_path = cache_directory.resolve() / f"{cache_key}.json"
    digest: str | None = None
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = None
        if isinstance(cached, dict) and all(
            cached.get(name) == value for name, value in identity.items()
        ):
            candidate = cached.get("sha256")
            if isinstance(candidate, str) and len(candidate) == 64:
                digest = candidate
    if digest is None:
        digest = file_sha256(artifact)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({**identity, "sha256": digest}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
    if expected is not None and digest != expected:
        raise ValueError(
            f"artifact hash mismatch for {artifact}: expected {expected}, got {digest}"
        )
    return digest


def json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_file(root: Path, path: str | Path) -> tuple[str, Path]:
    repository = root.resolve()
    candidate = Path(path)
    absolute = candidate.resolve() if candidate.is_absolute() else (repository / candidate).resolve()
    try:
        relative = absolute.relative_to(repository).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact lies outside the repository: {absolute}") from exc
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return relative, absolute


def implementation_hashes(
    root: Path,
    *,
    entrypoints: Iterable[str | Path] = (),
) -> dict[str, str]:
    """Hash the complete library and experiment helpers used by an entrypoint.

    Experiment drivers may list their own files, while the model-independent
    package and shared experiment helpers are discovered automatically.  This
    deliberately invalidates cached results after any library change rather
    than relying on a hand-maintained import closure.
    """

    repository = root.resolve()
    discovered = {
        path.resolve()
        for base in (
            repository / "src" / "inference_scaling",
            repository / "experiments" / "shared",
        )
        if base.is_dir()
        for path in base.rglob("*.py")
        if path.is_file()
    }
    for entrypoint in entrypoints:
        _, absolute = _relative_file(repository, entrypoint)
        discovered.add(absolute)
    return {
        path.relative_to(repository).as_posix(): file_sha256(path)
        for path in sorted(discovered, key=lambda item: item.as_posix())
    }


_CHECKPOINT_METADATA_SUFFIXES = frozenset(
    {".json", ".jinja", ".model", ".py", ".tiktoken", ".txt"}
)


def directory_hashes(
    path: Path,
    *,
    suffixes: Collection[str] | None = None,
) -> dict[str, str]:
    """Hash regular files below a directory using stable relative names."""

    directory = path.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    normalized_suffixes = None if suffixes is None else {value.lower() for value in suffixes}
    files = [
        candidate
        for candidate in directory.rglob("*")
        if candidate.is_file()
        and (
            normalized_suffixes is None
            or candidate.suffix.lower() in normalized_suffixes
        )
    ]
    return {
        candidate.relative_to(directory).as_posix(): file_sha256(candidate)
        for candidate in sorted(files, key=lambda item: item.as_posix())
    }


def checkpoint_metadata_hashes(path: Path) -> dict[str, str]:
    """Hash tokenizer, configuration, and custom-code checkpoint files."""

    return directory_hashes(path, suffixes=_CHECKPOINT_METADATA_SUFFIXES)


def adapter_hashes(path: Path) -> dict[str, str]:
    """Hash the PEFT files that determine an adapter's inference behavior."""

    directory = path.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = [
        candidate
        for candidate in directory.rglob("*")
        if candidate.is_file()
        and (
            candidate.name == "adapter_config.json"
            or candidate.name.startswith("adapter_model.")
        )
    ]
    if not any(candidate.name == "adapter_config.json" for candidate in files):
        raise FileNotFoundError(directory / "adapter_config.json")
    if not any(candidate.name.startswith("adapter_model.") for candidate in files):
        raise FileNotFoundError(f"adapter weights are absent below {directory}")
    return {
        candidate.relative_to(directory).as_posix(): file_sha256(candidate)
        for candidate in sorted(files, key=lambda item: item.as_posix())
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL object stream, rejecting malformed rows explicitly."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            records.append(value)
    return records


def indexed_records(
    records: Iterable[Mapping[str, Any]],
    *,
    key: str = "problem_index",
) -> dict[int, Mapping[str, Any]]:
    """Index resumable records and reject duplicate or invalid identifiers."""

    indexed: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if key not in record:
            raise ValueError(f"record is missing {key!r}")
        identifier = record[key]
        if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier < 0:
            raise ValueError(f"{key} must be a non-negative integer")
        if identifier in indexed:
            raise ValueError(f"duplicate {key} {identifier}")
        indexed[identifier] = record
    return indexed


def dataclass_snapshot_delta(
    before: Any,
    after: Any,
    *,
    constant_fields: Collection[str] = (),
) -> dict[str, Any]:
    left = asdict(before)
    right = asdict(after)
    if left.keys() != right.keys():
        raise ValueError("snapshot schemas do not match")
    constants = set(constant_fields)
    unknown = constants - right.keys()
    if unknown:
        raise ValueError(f"constant fields are absent from the snapshot: {sorted(unknown)}")
    return {
        key: right[key] if key in constants else right[key] - left[key]
        for key in right
    }


__all__ = [
    "adapter_hashes",
    "checkpoint_metadata_hashes",
    "cached_file_sha256",
    "dataclass_snapshot_delta",
    "directory_hashes",
    "file_sha256",
    "implementation_hashes",
    "indexed_records",
    "json_fingerprint",
    "load_jsonl",
]
