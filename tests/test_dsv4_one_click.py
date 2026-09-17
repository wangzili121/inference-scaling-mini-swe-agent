from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load_script(name: str):
    path = ROOT / "deploy" / "dsv4_flash" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_npu_idle_parser_requires_health_process_and_hbm_evidence() -> None:
    module = load_script("check_npu_idle.py")
    text = """
| 0     910B3               | OK            | 101.0       41                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          3436 / 65536         |
| No running processes found in NPU 0                                                            |
"""

    idle, health, hbm = module.parse_npu_smi(text)

    assert idle == {0}
    assert health == {0: "OK"}
    assert hbm == {0: 3436}


def test_capacity_summarizer_ignores_diagnostics_json(tmp_path: Path) -> None:
    result = {
        "max_concurrency": 4,
        "completed": 16,
        "failed": 0,
        "duration": 12.0,
        "request_throughput": 1.25,
        "mean_e2el_ms": 2000.0,
        "p95_e2el_ms": 3000.0,
        "p99_e2el_ms": 3500.0,
    }
    (tmp_path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "diagnostics.json").write_text(
        json.dumps({"backend": {"sample_calls": 10}}), encoding="utf-8"
    )
    module = load_script("summarize_benchmark.py")

    import sys

    previous = sys.argv
    sys.argv = ["summarize_benchmark.py", str(tmp_path)]
    try:
        module.main()
    finally:
        sys.argv = previous

    summary = json.loads((tmp_path / "capacity-summary.json").read_text())
    assert len(summary) == 1
    assert summary[0]["request_throughput"] == 1.25
