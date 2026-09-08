"""Validate the v0.18 generator-to-stateless-position adapter on NPU."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def load_sampler():
    path = Path(__file__).parent / "runtime" / "sampler.py"
    spec = importlib.util.spec_from_file_location("categorical_v018_sampler", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_generators(seeds: list[int]) -> dict[int, torch.Generator]:
    result = {}
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device="npu")
        generator.manual_seed(seed)
        result[index] = generator
    return result


def run_once(module, logits: torch.Tensor, seeds: list[int]):
    generators = make_generators(seeds)
    before = [int(generator.get_offset()) for generator in generators.values()]
    sampled = module.native_categorical_sample(logits, generators)
    assert sampled is not None
    after = [int(generator.get_offset()) for generator in generators.values()]
    return sampled.cpu(), before, after


def main() -> None:
    os.environ["VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE"] = "1"
    torch.npu.set_device(0)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    module = load_sampler()
    logits = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]],
        dtype=torch.float16,
        device="npu",
    )
    first, before, after = run_once(module, logits, [1234, 5678])
    second, _, second_after = run_once(module, logits, [1234, 5678])
    assert torch.equal(first, second)
    assert before == [0, 0]
    assert after == [12, 12]
    assert second_after == [12, 12]

    partial = {0: make_generators([1234])[0]}
    assert module.native_categorical_sample(logits, partial) is None

    print(
        json.dumps(
            {
                "deterministic": True,
                "offsets_before": before,
                "offsets_after": after,
                "partial_batch_falls_back": True,
                "sampled_tokens": first.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
