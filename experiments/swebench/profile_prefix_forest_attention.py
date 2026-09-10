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


def _screen_cis_forest_shape(
    *,
    candidate_count: int,
    rollout_count: int,
    trunk_tokens: int,
    candidate_tokens: int,
    unique_tokens: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Measure the exact trunk -> candidate -> rollout attention geometry."""

    device = torch.device("npu:0")
    branches = candidate_count * rollout_count
    query = torch.randn(
        branches, query_heads, 1, head_dim, device=device, dtype=dtype
    )
    trunk_key = torch.randn(
        1, kv_heads, trunk_tokens, head_dim, device=device, dtype=dtype
    )
    trunk_value = torch.randn_like(trunk_key)
    candidate_key = torch.randn(
        candidate_count,
        kv_heads,
        candidate_tokens,
        head_dim,
        device=device,
        dtype=dtype,
    )
    candidate_value = torch.randn_like(candidate_key)
    unique_key = torch.randn(
        branches, kv_heads, unique_tokens, head_dim, device=device, dtype=dtype
    )
    unique_value = torch.randn_like(unique_key)
    expanded_candidate_key = candidate_key.repeat_interleave(rollout_count, dim=0)
    expanded_candidate_value = candidate_value.repeat_interleave(
        rollout_count, dim=0
    )
    full_key = torch.cat(
        (
            trunk_key.expand(branches, -1, -1, -1),
            expanded_candidate_key,
            unique_key,
        ),
        dim=2,
    ).contiguous()
    full_value = torch.cat(
        (
            trunk_value.expand(branches, -1, -1, -1),
            expanded_candidate_value,
            unique_value,
        ),
        dim=2,
    ).contiguous()
    trunk_query = query.squeeze(2).permute(1, 0, 2).unsqueeze(0).contiguous()
    candidate_query = (
        query.squeeze(2)
        .reshape(candidate_count, rollout_count, query_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    trunk_stream = torch.npu.Stream()
    candidate_stream = torch.npu.Stream()
    unique_stream = torch.npu.Stream()

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        return _attention(
            query,
            full_key,
            full_value,
            query_heads=query_heads,
            kv_heads=kv_heads,
        )

    def decomposed() -> tuple[torch.Tensor, torch.Tensor]:
        trunk_output, trunk_lse = _attention(
            trunk_query,
            trunk_key,
            trunk_value,
            query_heads=query_heads,
            kv_heads=kv_heads,
        )
        middle_output, middle_lse = _attention(
            candidate_query,
            candidate_key,
            candidate_value,
            query_heads=query_heads,
            kv_heads=kv_heads,
        )
        unique_output, unique_lse = _attention(
            query,
            unique_key,
            unique_value,
            query_heads=query_heads,
            kv_heads=kv_heads,
        )
        trunk_output = trunk_output.squeeze(0).permute(1, 0, 2).unsqueeze(2)
        trunk_lse = trunk_lse.squeeze(0).permute(1, 0, 2).unsqueeze(2)
        middle_output = (
            middle_output.permute(0, 2, 1, 3)
            .reshape(branches, query_heads, 1, head_dim)
        )
        middle_lse = (
            middle_lse.permute(0, 2, 1, 3)
            .reshape(branches, query_heads, 1, -1)
        )
        total_lse = torch.logaddexp(
            torch.logaddexp(trunk_lse, middle_lse), unique_lse
        )
        output = (
            torch.exp(trunk_lse - total_lse).to(dtype) * trunk_output
            + torch.exp(middle_lse - total_lse).to(dtype) * middle_output
            + torch.exp(unique_lse - total_lse).to(dtype) * unique_output
        )
        return output, total_lse

    def parallel_decomposed() -> tuple[torch.Tensor, torch.Tensor]:
        with torch.npu.stream(trunk_stream):
            trunk_output, trunk_lse = _attention(
                trunk_query,
                trunk_key,
                trunk_value,
                query_heads=query_heads,
                kv_heads=kv_heads,
            )
        with torch.npu.stream(candidate_stream):
            middle_output, middle_lse = _attention(
                candidate_query,
                candidate_key,
                candidate_value,
                query_heads=query_heads,
                kv_heads=kv_heads,
            )
        with torch.npu.stream(unique_stream):
            unique_output, unique_lse = _attention(
                query,
                unique_key,
                unique_value,
                query_heads=query_heads,
                kv_heads=kv_heads,
            )
        current_stream = torch.npu.current_stream()
        current_stream.wait_stream(trunk_stream)
        current_stream.wait_stream(candidate_stream)
        current_stream.wait_stream(unique_stream)
        trunk_output = trunk_output.squeeze(0).permute(1, 0, 2).unsqueeze(2)
        trunk_lse = trunk_lse.squeeze(0).permute(1, 0, 2).unsqueeze(2)
        middle_output = (
            middle_output.permute(0, 2, 1, 3)
            .reshape(branches, query_heads, 1, head_dim)
        )
        middle_lse = (
            middle_lse.permute(0, 2, 1, 3)
            .reshape(branches, query_heads, 1, -1)
        )
        total_lse = torch.logaddexp(
            torch.logaddexp(trunk_lse, middle_lse), unique_lse
        )
        output = (
            torch.exp(trunk_lse - total_lse).to(dtype) * trunk_output
            + torch.exp(middle_lse - total_lse).to(dtype) * middle_output
            + torch.exp(unique_lse - total_lse).to(dtype) * unique_output
        )
        return output, total_lse

    baseline_output, baseline_lse = baseline()
    shared_output, shared_lse = decomposed()
    parallel_output, parallel_lse = parallel_decomposed()
    _synchronize()
    baseline_latency = _measure(baseline, warmup=warmup, iterations=iterations)
    shared_latency = _measure(decomposed, warmup=warmup, iterations=iterations)
    parallel_latency = _measure(
        parallel_decomposed, warmup=warmup, iterations=iterations
    )
    baseline_kv_tokens = branches * (
        trunk_tokens + candidate_tokens + unique_tokens
    )
    shared_kv_tokens = (
        trunk_tokens
        + candidate_count * candidate_tokens
        + branches * unique_tokens
    )
    return {
        "candidate_count": candidate_count,
        "rollout_count": rollout_count,
        "branches": branches,
        "trunk_tokens": trunk_tokens,
        "candidate_tokens": candidate_tokens,
        "unique_tokens": unique_tokens,
        "baseline": baseline_latency,
        "cis_forest": shared_latency,
        "cis_forest_parallel": parallel_latency,
        "p50_speedup": baseline_latency["p50_ms"] / shared_latency["p50_ms"],
        "parallel_p50_speedup": (
            baseline_latency["p50_ms"] / parallel_latency["p50_ms"]
        ),
        "ideal_kv_read_reduction": baseline_kv_tokens / shared_kv_tokens,
        "maximum_output_error": float(
            (baseline_output.float() - shared_output.float()).abs().max().cpu()
        ),
        "maximum_lse_error": float(
            (baseline_lse.float() - shared_lse.float()).abs().max().cpu()
        ),
        "parallel_maximum_output_error": float(
            (baseline_output.float() - parallel_output.float()).abs().max().cpu()
        ),
        "parallel_maximum_lse_error": float(
            (baseline_lse.float() - parallel_lse.float()).abs().max().cpu()
        ),
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
    parser.add_argument(
        "--cis-forest",
        action="store_true",
        help="measure the two-level Conditional IS branch geometry",
    )
    parser.add_argument("--candidate-counts", default="4,8,15")
    parser.add_argument("--rollout-count", type=int, default=3)
    parser.add_argument("--candidate-tokens", type=int, default=128)
    parser.add_argument("--unique-tokens", type=int, default=128)
    args = parser.parse_args()

    results = []
    if args.cis_forest:
        for prefix_tokens in map(int, args.prefix_lengths.split(",")):
            for candidate_count in map(int, args.candidate_counts.split(",")):
                results.append(
                    _screen_cis_forest_shape(
                        candidate_count=candidate_count,
                        rollout_count=args.rollout_count,
                        trunk_tokens=prefix_tokens,
                        candidate_tokens=args.candidate_tokens,
                        unique_tokens=args.unique_tokens,
                        query_heads=args.query_heads,
                        kv_heads=args.kv_heads,
                        head_dim=args.head_dim,
                        dtype=torch.bfloat16,
                        warmup=args.warmup,
                        iterations=args.iterations,
                    )
                )
    else:
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
