import pytest

from inference_scaling.arllm.config import (
    BaseReplayConfig,
    ConditionalISConfig,
    DynamicISConfig,
    MHConfig,
    SamplingConfig,
)
from inference_scaling.arllm.types import GenerationRequest, SequenceSample


def test_sampling_config_identifies_actual_policy() -> None:
    config = SamplingConfig(temperature=0.7, top_p=0.9, top_k=20, eos_token_id=2)
    assert config.policy_id == "temperature=0.7;top_p=0.9;top_k=20;eos=2"


def test_policy_id_preserves_distinct_float_values() -> None:
    assert (
        SamplingConfig(temperature=1.0000001).policy_id
        != SamplingConfig(temperature=1.0000002).policy_id
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SamplingConfig(temperature=0),
        lambda: SamplingConfig(top_p=1.1),
        lambda: MHConfig(total_length=4, block_size=8),
        lambda: MHConfig(suffix_schedule="unknown"),
        lambda: BaseReplayConfig(fresh_rollouts=0),
        lambda: DynamicISConfig(auxiliary_mixture=1.0),
        lambda: SamplingConfig(temperature=float("nan")),
        lambda: SamplingConfig(top_p=float("inf")),
        lambda: ConditionalISConfig(reward_temperature=float("inf")),
        lambda: ConditionalISConfig(active_step_limit=0),
        lambda: ConditionalISConfig(rollout_design="unknown"),
        lambda: ConditionalISConfig(exact_rollout_early_stop=True),
        lambda: ConditionalISConfig(
            rollout_log_weight_bounds=(0.0, 1.0),
        ),
        lambda: ConditionalISConfig(
            exact_rollout_early_stop=True,
            rollout_log_weight_bounds=(1.0, 0.0),
        ),
        lambda: ConditionalISConfig(
            rollout_design="scrambled_sobol",
            exact_rollout_early_stop=True,
            rollout_log_weight_bounds=(0.0, 1.0),
        ),
        lambda: DynamicISConfig(auxiliary_mixture=float("nan")),
    ],
)
def test_invalid_configs_fail_early(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_sampled_token_logprob_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        SequenceSample((), (1,), (float("nan"),), "policy", "model", "request")


@pytest.mark.parametrize(
    "uniforms",
    [(0.1,), (0.1, float("nan")), (0.1, 1.0), (0.1, -0.1)],
)
def test_generation_request_validates_explicit_uniforms(uniforms) -> None:
    with pytest.raises(ValueError, match="uniform"):
        GenerationRequest((), 2, SamplingConfig(), 1, "invalid", uniforms=uniforms)


def test_generation_request_validates_arithmetic_uniform() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        GenerationRequest(
            (),
            2,
            SamplingConfig(),
            1,
            "invalid",
            uniforms=(0.1, 0.2),
            arithmetic_uniform=0.3,
        )
    for value in (-0.1, 1.0, float("inf")):
        with pytest.raises(ValueError, match="arithmetic sampling uniform"):
            GenerationRequest(
                (),
                2,
                SamplingConfig(),
                1,
                "invalid",
                arithmetic_uniform=value,
            )


def test_generation_request_validates_segmented_rng() -> None:
    with pytest.raises(ValueError, match="both a boundary"):
        GenerationRequest(
            (), 4, SamplingConfig(), 1, "missing-seed", rng_switch_after_tokens=2
        )
    with pytest.raises(ValueError, match="inside"):
        GenerationRequest(
            (),
            4,
            SamplingConfig(),
            1,
            "invalid-boundary",
            rng_switch_after_tokens=4,
            rng_switch_seed=2,
        )
    with pytest.raises(ValueError, match="requires segmented RNG"):
        GenerationRequest(
            (),
            4,
            SamplingConfig(),
            1,
            "missing-boundary",
            rng_prefix_group="group",
            rng_prefix_group_size=2,
        )
    request = GenerationRequest(
        (),
        4,
        SamplingConfig(),
        1,
        "valid",
        rng_switch_after_tokens=2,
        rng_switch_seed=3,
        rng_prefix_group="group",
        rng_prefix_group_size=2,
    )
    assert request.rng_switch_after_tokens == 2
    assert request.rng_switch_seed == 3
    assert request.rng_prefix_group == "group"
    assert request.rng_prefix_group_size == 2


def test_generation_request_validates_fork_waiter_parent() -> None:
    with pytest.raises(ValueError, match="requires a parent"):
        GenerationRequest(
            (), 4, SamplingConfig(), 1, "orphan", fork_wait_for_parent=True
        )


def test_generation_request_validates_fork_group_metadata() -> None:
    with pytest.raises(ValueError, match="provided together"):
        GenerationRequest(
            (),
            4,
            SamplingConfig(),
            1,
            "partial-group",
            fork_expected_children=2,
            fork_group_id="step-0",
        )
    with pytest.raises(ValueError, match="fork_release_remaining"):
        GenerationRequest(
            (),
            4,
            SamplingConfig(),
            1,
            "bad-threshold",
            fork_expected_children=2,
            fork_group_id="step-0",
            fork_group_size=4,
            fork_release_remaining=4,
        )
    with pytest.raises(ValueError, match="requires fork group metadata"):
        GenerationRequest(
            (),
            4,
            SamplingConfig(),
            1,
            "adaptive-without-group",
            fork_adaptive_release=True,
        )
    with pytest.raises(ValueError, match="must be in"):
        GenerationRequest(
            (),
            4,
            SamplingConfig(),
            1,
            "bad-adaptive-fraction",
            fork_expected_children=2,
            fork_group_id="step-0",
            fork_group_size=4,
            fork_release_remaining=0,
            fork_adaptive_release=True,
            fork_adaptive_runnable_fraction=0.0,
        )
