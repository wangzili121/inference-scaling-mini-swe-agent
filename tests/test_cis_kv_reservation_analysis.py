import json

import pytest

from experiments.swebench.analyze_cis_kv_reservations import analyze_trace


def test_analyze_trace_models_shared_prefix_and_realized_branches(tmp_path) -> None:
    trace = tmp_path / "algorithm.jsonl"
    trace.write_text(
        json.dumps(
            {
                "request_id": "job-0",
                "stage_events": [
                    {
                        "name": "candidate",
                        "step": 0,
                        "prefix_tokens": 1000,
                        "block_length": 128,
                    }
                ],
                "conditional_steps": [
                    {
                        "block_id": 0,
                        "generated_tokens_before": 0,
                        "candidates": [
                            {
                                "output_tokens": 64,
                                "terminal": True,
                                "rollouts": [],
                            },
                            {
                                "output_tokens": 128,
                                "terminal": False,
                                "rollouts": [
                                    {"output_tokens": 100},
                                    {"output_tokens": 200},
                                ],
                            },
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = analyze_trace(
        trace,
        total_length=512,
        rollout_count=2,
        kv_block_size=128,
    )

    assert result["steps"] == 1
    assert result["conservative_tokens"]["mean"] == 2816
    assert result["realized_peak_tokens"]["mean"] == 1664
    assert result["overreservation_ratio"]["mean"] == pytest.approx(2816 / 1664)
    assert result["terminal_candidate_fraction"]["mean"] == 0.5
