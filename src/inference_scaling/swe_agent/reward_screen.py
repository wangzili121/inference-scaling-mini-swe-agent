"""Generate shared rollout pools and screen reward/temperature choices offline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

from inference_scaling.arllm.rewards import (
    ConsilienceReward,
    SequenceLogProbabilityReward,
)
from inference_scaling.arllm.types import GeneratedSequenceStatistics
from inference_scaling.shared.rng import SeedStream
from inference_scaling.swe_agent.calibration import (
    calibrate_logprob_alpha,
    calibrate_reward_temperature,
)
from inference_scaling.swe_agent.service import ConditionalISRunner, load_service_config
from inference_scaling.swe_agent.tool_calls import (
    ToolCallParseError,
    parse_assistant_text,
)


class _SharedPoolReward:
    def __init__(
        self,
        logprob: SequenceLogProbabilityReward,
        *,
        confidence_top_k: int,
    ) -> None:
        self.logprob = logprob
        self.generation_confidence_top_k = confidence_top_k

    def __call__(self, prompt, completion) -> float:
        return self.logprob(prompt, completion)

    def batch_statistics(
        self,
        prompt,
        values: Sequence[GeneratedSequenceStatistics],
    ) -> tuple[float, ...]:
        return self.logprob.batch_statistics(prompt, values)


def _load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    usable = [record for record in records if isinstance(record.get("messages"), list)]
    return usable if limit is None else usable[:limit]


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_variance * right_variance)
    return numerator / denominator if denominator else 0.0


def _valid_action(backend: Any, statistics: GeneratedSequenceStatistics) -> bool:
    text = backend.decode(statistics.token_ids, skip_special_tokens=False)
    try:
        return bool(parse_assistant_text(text, request_id="reward-screen").actions)
    except ToolCallParseError:
        return False


def _screen_temperature(
    backend: Any,
    base_config: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    temperature: float,
    root_seed: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["sampling"]["temperature"] = temperature
    config["conditional_is"].update(
        {"candidate_count": 8, "rollout_count": 2, "block_size": 128}
    )
    config["service"] = {}
    runner = ConditionalISRunner(backend, config)
    logprob = SequenceLogProbabilityReward(backend, runner.sampling, scale=1.0)
    consilience = ConsilienceReward(
        backend,
        runner.sampling,
        top_k=5,
        window_fraction=0.2,
        skip_fraction=0.05,
        initial_penalty=3.0,
    )
    runner.reward = _SharedPoolReward(logprob, confidence_top_k=consilience.top_k)

    logprob_blocks: list[list[list[float]]] = []
    consilience_blocks: list[list[list[float]]] = []
    output_lengths: list[float] = []
    logprob_values: list[float] = []
    consilience_values: list[float] = []
    token_sequences: list[tuple[int, ...]] = []
    valid_actions = 0
    jobs: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        execution = runner.execute(
            record["messages"],
            seed=SeedStream(root_seed).derive("reward-screen", temperature, index),
            request_namespace=(
                f"reward-screen:{temperature}:{record.get('request_id', index)}"
            ),
        )
        jobs.append(
            {
                "request_id": record.get("request_id", str(index)),
                "prompt_tokens": len(execution.prompt),
                "completion_tokens": len(execution.result.token_ids),
                "seconds": execution.total_seconds,
                "backend_delta": execution.backend_delta,
            }
        )
        for step in execution.result.steps:
            candidate_statistics: list[list[GeneratedSequenceStatistics]] = []
            for candidate in step.candidates:
                values = [
                    rollout.generation_statistics for rollout in candidate.rollouts
                ]
                if any(value is None for value in values):
                    raise RuntimeError(
                        "Conditional IS result omitted generation statistics"
                    )
                candidate_statistics.append(
                    [value for value in values if value is not None]
                )
            flat = [value for candidate in candidate_statistics for value in candidate]
            logprob_flat = logprob.batch_statistics(execution.prompt, flat)
            consilience_flat = consilience.batch_statistics(execution.prompt, flat)
            offset = 0
            logprob_block: list[list[float]] = []
            consilience_block: list[list[float]] = []
            for candidate in candidate_statistics:
                count = len(candidate)
                logprob_block.append(list(logprob_flat[offset : offset + count]))
                consilience_block.append(
                    list(consilience_flat[offset : offset + count])
                )
                offset += count
            logprob_blocks.append(logprob_block)
            consilience_blocks.append(consilience_block)
            for values, logprob_value, consilience_value in zip(
                flat, logprob_flat, consilience_flat, strict=True
            ):
                output_lengths.append(float(len(values.token_ids)))
                logprob_values.append(float(logprob_value))
                consilience_values.append(float(consilience_value))
                token_sequences.append(tuple(values.token_ids))
                valid_actions += int(_valid_action(backend, values))

    alpha = calibrate_logprob_alpha(logprob_blocks)
    tau = calibrate_reward_temperature(consilience_blocks)
    pool_hash = hashlib.sha256(
        json.dumps(token_sequences, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    total_sequences = len(token_sequences)
    return {
        "temperature": temperature,
        "pool_sha256": pool_hash,
        "jobs": jobs,
        "blocks": len(logprob_blocks),
        "rollout_sequences": total_sequences,
        "unique_sequence_ratio": (
            len(set(token_sequences)) / total_sequences if total_sequences else 0.0
        ),
        "valid_tool_call_ratio": (
            valid_actions / total_sequences if total_sequences else 0.0
        ),
        "length_correlation": {
            "sequence_logprob": _correlation(output_lengths, logprob_values),
            "consilience": _correlation(output_lengths, consilience_values),
        },
        "calibration": {
            "logprob_alpha": {
                "value": alpha.parameter,
                "median_ess_ratio": alpha.median_ess_ratio,
                "bounded": alpha.bounded,
            },
            "consilience_tau": {
                "value": tau.parameter,
                "median_ess_ratio": tau.median_ess_ratio,
                "bounded": tau.bounded,
            },
        },
        "reward_blocks": {
            "sequence_logprob": logprob_blocks,
            "consilience": consilience_blocks,
        },
    }


def run_reward_screen(
    config_path: str | Path,
    workload_path: str | Path,
    *,
    temperatures: Sequence[float] = (0.25, 0.7, 1.0),
    limit: int | None = None,
    seed: int = 20260908,
) -> dict[str, Any]:
    config = load_service_config(config_path)
    records = _load_records(Path(workload_path), limit)
    if not records:
        raise ValueError("reward screen requires at least one workload record")
    root = ConditionalISRunner.from_toml(config_path)
    try:
        results = [
            _screen_temperature(
                root.backend,
                config,
                records,
                temperature=float(temperature),
                root_seed=seed,
            )
            for temperature in temperatures
        ]
    finally:
        root.close()
    return {
        "schema_version": 1,
        "config": str(Path(config_path).resolve()),
        "workload": str(Path(workload_path).resolve()),
        "requests": len(records),
        "candidate_count": 8,
        "rollout_count": 2,
        "block_size": 128,
        "seed": seed,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--temperature", type=float, action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_reward_screen(
        args.config,
        args.workload,
        temperatures=args.temperature or (0.25, 0.7, 1.0),
        limit=args.limit,
        seed=args.seed,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "results"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
