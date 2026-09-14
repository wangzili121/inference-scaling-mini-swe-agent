import json

from experiments.swebench.simulate_cis_two_phase_admission import (
    StepClaim,
    admitted_frontier,
    extract_step_claims,
    safe_completion_order,
    simulate_trace,
)


def _claim(name: str, allocation: int, maximum: int) -> StepClaim:
    return StepClaim(
        claim_id=name,
        candidate_count=1,
        candidate_tokens=allocation,
        maximum_tokens=maximum,
        realized_candidate_tokens=allocation,
        post_candidate_max_tokens=maximum,
        realized_peak_tokens=maximum,
    )


def test_banker_admission_exposes_safe_headroom() -> None:
    claims = [_claim("a", 40, 70), _claim("b", 30, 80)]

    full = admitted_frontier(claims, capacity_tokens=100, mode="full_reservation")
    two_phase = admitted_frontier(claims, capacity_tokens=100, mode="two_phase_safe")

    assert [claim.claim_id for claim in full] == ["a"]
    assert [claim.claim_id for claim in two_phase] == ["a", "b"]
    assert safe_completion_order(two_phase, 100) == ["a", "b"]


def test_unsafe_candidate_only_state_is_rejected() -> None:
    claims = [
        _claim("a", 30, 80),
        _claim("b", 30, 80),
        _claim("c", 30, 80),
    ]

    assert (
        len(admitted_frontier(claims, capacity_tokens=100, mode="candidate_only")) == 3
    )
    assert (
        len(admitted_frontier(claims, capacity_tokens=100, mode="two_phase_safe")) == 1
    )


def test_max_num_seqs_limits_theoretical_kv_headroom() -> None:
    claims = [
        StepClaim(
            claim_id=name,
            candidate_count=8,
            candidate_tokens=20,
            maximum_tokens=40,
            realized_candidate_tokens=20,
            post_candidate_max_tokens=40,
            realized_peak_tokens=40,
        )
        for name in ("a", "b")
    ]

    assert (
        len(
            admitted_frontier(
                claims,
                capacity_tokens=100,
                mode="two_phase_safe",
                max_num_seqs=8,
            )
        )
        == 1
    )


def test_extracts_phase_claims_from_algorithm_trace(tmp_path) -> None:
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

    claims = extract_step_claims(
        trace, total_length=512, rollout_count=2, kv_block_size=128
    )

    assert len(claims) == 1
    assert claims[0].candidate_tokens == 1280
    assert claims[0].maximum_tokens == 2816
    assert claims[0].realized_candidate_tokens == 1280
    assert claims[0].post_candidate_max_tokens == 2048
    assert claims[0].realized_peak_tokens == 1664


def test_simulation_is_deterministic() -> None:
    claims = [_claim("a", 40, 70), _claim("b", 30, 80)]

    first = simulate_trace(
        claims, capacity_tokens=100, max_num_seqs=2, shuffle_trials=10, seed=7
    )
    second = simulate_trace(
        claims, capacity_tokens=100, max_num_seqs=2, shuffle_trials=10, seed=7
    )

    assert first == second
    assert first["fixed_order"]["two_phase_safe"]["mns_bound"]["steps"] == 2
