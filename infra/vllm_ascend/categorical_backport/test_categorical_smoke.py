"""Focused NPU smoke tests for the backported categorical sampler."""

from __future__ import annotations

import json

print("import torch", flush=True)
import torch

print("import torch_npu", flush=True)
import torch_npu  # noqa: F401

torch.npu.set_device(0)
print("import vllm_ascend extension", flush=True)
import vllm_ascend.vllm_ascend_C  # noqa: F401
print("extension imported", flush=True)


def sample(logits: torch.Tensor, seed: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    rows = logits.shape[0]
    mapping = torch.arange(rows, dtype=torch.int32, device="npu")
    temperature = torch.ones(rows, dtype=torch.float32, device="npu")
    token_ids, _ = torch.ops._C_ascend.npu_categorical_sample(
        logits,
        mapping,
        temperature,
        seed,
        pos,
        False,
        False,
        None,
        None,
        False,
    )
    return token_ids.cpu()


def main() -> None:
    logits = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]],
        dtype=torch.float16,
        device="npu",
    )
    seed = torch.tensor([1234, 5678], dtype=torch.int64, device="npu")
    pos = torch.tensor([0, 0], dtype=torch.int64, device="npu")

    first = sample(logits, seed, pos)
    second = sample(logits, seed, pos)
    assert torch.equal(first, second), (first, second)

    # Each row is statelessly keyed by its own seed and position.
    swapped = sample(logits.flip(0), seed.flip(0), pos.flip(0)).flip(0)
    assert torch.equal(first, swapped), (first, swapped)

    positions = torch.arange(4096, dtype=torch.int64, device="npu")
    binary_logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device="npu").expand(4096, -1)
    repeated_seed = torch.full((4096,), 20260831, dtype=torch.int64, device="npu")
    samples = sample(binary_logits, repeated_seed, positions)
    observed = float((samples == 1).float().mean())
    expected = float(torch.softmax(torch.tensor([0.0, 1.0]), dim=0)[1])
    assert abs(observed - expected) < 0.035, (observed, expected)

    result = {
        "registered": hasattr(torch.ops._C_ascend, "npu_categorical_sample"),
        "deterministic": True,
        "row_order_invariant": True,
        "binary_expected_probability": expected,
        "binary_observed_probability": observed,
        "first_tokens": first.tolist(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
