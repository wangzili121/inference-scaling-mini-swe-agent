"""Create a checksummed compressed archive without modifying raw profile files."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from inference_scaling.swe_agent.artifacts import sha256_file


def archive_artifacts(source: str | Path, output: str | Path) -> dict[str, object]:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if not source_path.is_dir():
        raise ValueError(f"artifact source is not a directory: {source_path}")
    if output_path == source_path or source_path in output_path.parents:
        raise ValueError("archive output must be outside the source directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w:gz" if output_path.name.endswith((".tar.gz", ".tgz")) else "w"
    with tarfile.open(output_path, mode) as archive:
        archive.add(source_path, arcname=source_path.name, recursive=True)
    digest = sha256_file(output_path)
    checksum = output_path.with_suffix(output_path.suffix + ".sha256")
    checksum.write_text(f"{digest}  {output_path.name}\n", encoding="ascii")
    return {
        "source": str(source_path),
        "archive": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": digest,
        "checksum": str(checksum),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(archive_artifacts(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
