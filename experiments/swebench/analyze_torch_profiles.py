"""Analyze every raw Ascend Torch profile below an artifact directory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


_ANALYZE = """
import sys
from torch_npu.profiler.profiler import analyse

analyse(
    sys.argv[1],
    max_process_number=int(sys.argv[2]),
    export_type=["text", "db"],
)
"""


def _analyze_one(path: Path, processes: int) -> dict[str, object]:
    completed = subprocess.run(
        (sys.executable, "-c", _ANALYZE, str(path), str(processes)),
        text=True,
        capture_output=True,
    )
    outputs = list((path / "ASCEND_PROFILER_OUTPUT").glob("*"))
    return {
        "path": str(path),
        "returncode": completed.returncode,
        "outputs": len(outputs),
        "stdout": completed.stdout[-2_000:],
        "stderr": completed.stderr[-2_000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--processes-per-rank", type=int, default=4)
    args = parser.parse_args()

    roots = sorted(args.root.glob("profile-*/*_ascend_pt"))
    if not roots:
        raise SystemExit(f"no raw Ascend profiles below {args.root}")
    results = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(roots))) as pool:
        futures = {
            pool.submit(_analyze_one, path, args.processes_per_rank): path
            for path in roots
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    if any(result["returncode"] or not result["outputs"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
