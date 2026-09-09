"""Screen prefix-shared decode attention on Ascend with Qwen3-Coder shapes.

This is a kernel-level opportunity test, not an end-to-end benchmark.  The
shared path follows Hydragen's exact prefix/suffix decomposition: compute both
parts with softmax LSE, then merge their outputs with online-softmax weights.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from typing import Any

import torch
import torch_npu


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_heads: int,
    kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, lse = torch_npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        num_heads=query_heads,
        num_key_value_heads=kv_heads,
        input_layout="BNSD",
        scale=1.0 / math.sqrt(query.shape[-1]),
        pre_tokens=2**31 - 1,
        next_tokens=2**31 - 1,
        sparse_mode=0,
        softmax_lse_flag=True,
    )
    return output, lse


def _synchronize() -> None:
    torch.npu.synchronize()


def _measure(callback: Any, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        callback()
    _synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        callback()
        _synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


def _screen_shape(
    *,
    group_size: int,
    prefix_tokens: int,
    suffix_tokens: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    device = torch.device("npu:0")
    query = torch.randn(
        group_size, query_heads, 1, head_dim, device=device, dtype=dtype
    )
    prefix_key = torch.randn(
        1, kv_heads, prefix_tokens, head_dim, device=device, dtype=dtype
    )
    prefix_value = torch.randn_like(prefix_key)
    suffix_key = torch.randn(
        group_size, kv_heads, suffix_tokens, head_dim, device=device, dtype=dtype
    )
    suffix_value = torch.randn_like(suffix_key)
    full_key = torch.cat(
        (prefix_key.expand(group_size, -1, -1, -1), suffix_key), dim=2
    ).contiguous()
    full_value = torch.cat(
        (prefix_value.expand(group_size, -1, -1, -1), suffix_value), dim=2
    ).contiguous()
    packed_query = query.squeeze(2).permute(1, 0, 2).unsqueeze(0).contiguous()

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        return _attention(
            query,
            full_key,
            full_value,
            query_heads=query_heads,
            kv_heads=kv_heads,
        )

    def decomposed() -> tuple[torch.Tensor, torch.Tensor]:
        prefix_output, prefix_lse = _attention(
            packed_query,
            prefix_key,
            prefix_value,
            query_heads=query_heads,
            kv_heads=kv_heads,
        )
        suffix_output, suffix_lse = _attention(
            query,
            suffix_key,
            suffix_value,
            query_heads=query_heads,
            kv_heads=kv_heads,
        )
        prefix_output = prefix_output.squeeze(0).permute(1, 0, 2).unsqueeze(2)
        prefix_lse = prefix_lse.squeeze(0).permute(1, 0, 2).unsqueeze(2)
        total_lse = torch.logaddexp(prefix_lse, suffix_lse)
        output = (
            torch.exp(prefix_lse - total_lse).to(dtype) * prefix_output
            + torch.exp(suffix_lse - total_lse).to(dtype) * suffix_output
        )
        return output, total_lse

    baseline_output, baseline_lse = baseline()
    shared_output, shared_lse = decomposed()
    _synchronize()
    maximum_output_error = float(
        (baseline_output.float() - shared_output.float()).abs().max().cpu()
    )
    maximum_lse_error = float(
        (baseline_lse.float() - shared_lse.float()).abs().max().cpu()
    )
    baseline_latency = _measure(baseline, warmup=warmup, iterations=iterations)
    shared_latency = _measure(decomposed, warmup=warmup, iterations=iterations)
    return {
        "group_size": group_size,
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "baseline": baseline_latency,
        "prefix_shared": shared_latency,
        "p50_speedup": baseline_latency["p50_ms"] / shared_latency["p50_ms"],
        "maximum_output_error": maximum_output_error,
        "maximum_lse_error": maximum_lse_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-sizes", default="2,3,4,8,15")
    parser.add_argument("--prefix-lengths", default="2048,8192,16384")
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    results = []
    for prefix_tokens in map(int, args.prefix_lengths.split(",")):
        for group_size in map(int, args.group_sizes.split(",")):
            results.append(
                _screen_shape(
                    group_size=group_size,
                    prefix_tokens=prefix_tokens,
                    suffix_tokens=args.suffix_tokens,
                    query_heads=args.query_heads,
                    kv_heads=args.kv_heads,
                    head_dim=args.head_dim,
                    dtype=torch.bfloat16,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            )
    print(json.dumps({"schema_version": 1, "results": results}, indent=2))


if __name__ == "__main__":
    main()
