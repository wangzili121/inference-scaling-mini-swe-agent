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


def _paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    *,
    query_heads: int,
    kv_heads: int,
    block_size: int,
    atten_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if key_cache.ndim == 4:
        key_cache = key_cache.view(key_cache.shape[0], key_cache.shape[1], -1)
        value_cache = value_cache.view(
            value_cache.shape[0], value_cache.shape[1], -1
        )
    output, lse = torch_npu.npu_fused_infer_attention_score(
        query,
        key_cache,
        value_cache,
        num_heads=query_heads,
        num_key_value_heads=kv_heads,
        input_layout="BNSD",
        scale=1.0 / math.sqrt(query.shape[-1]),
        pre_tokens=2**31 - 1,
        next_tokens=2**31 - 1,
        sparse_mode=0,
        atten_mask=atten_mask,
        block_table=block_table,
        block_size=block_size,
        actual_seq_lengths_kv=seq_lens,
        softmax_lse_flag=True,
    )
    return output, lse


def _native_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    *,
    query_heads: int,
    kv_heads: int,
) -> torch.Tensor:
    """Call the private decode fast path used by vLLM-Ascend 0.18."""

    if key_cache.ndim != 4:
        raise ValueError("native paged attention requires a 4-D KV cache")
    native_query = query.squeeze(2) if query.ndim == 4 else query
    output = torch.empty_like(native_query)
    # vLLM-Ascend passes its pinned CPU seq_lens tensor to this private op.
    context_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cpu")
    torch_npu._npu_paged_attention(
        query=native_query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=kv_heads,
        num_heads=query_heads,
        scale_value=1.0 / math.sqrt(native_query.shape[-1]),
        block_table=block_table,
        context_lens=context_lens,
        out=output,
    )
    return output.unsqueeze(2)


def _paged_attention_bsh(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    *,
    query_heads: int,
    kv_heads: int,
    block_size: int,
    atten_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run paged FIA in BSH layout to avoid BNSD query transposes."""

    if key_cache.ndim == 4:
        key_cache = key_cache.view(key_cache.shape[0], key_cache.shape[1], -1)
        value_cache = value_cache.view(
            value_cache.shape[0], value_cache.shape[1], -1
        )
    return torch_npu.npu_fused_infer_attention_score(
        query,
        key_cache,
        value_cache,
        num_heads=query_heads,
        num_key_value_heads=kv_heads,
        input_layout="BSH",
        scale=1.0 / math.sqrt(query.shape[-1] // query_heads),
        pre_tokens=2**31 - 1,
        next_tokens=2**31 - 1,
        sparse_mode=0,
        atten_mask=atten_mask,
        block_table=block_table,
        block_size=block_size,
        actual_seq_lengths_kv=seq_lens,
        softmax_lse_flag=True,
    )


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
    candidate_prefix_key = torch.cat(
        (
            trunk_key.expand(candidate_count, -1, -1, -1),
            candidate_key,
        ),
        dim=2,
    ).contiguous()
    candidate_prefix_value = torch.cat(
        (
            trunk_value.expand(candidate_count, -1, -1, -1),
            candidate_value,
        ),
        dim=2,
    ).contiguous()
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

    def candidate_group_decomposed() -> tuple[torch.Tensor, torch.Tensor]:
        """Use one shared prefix per candidate and merge only the unique tail.

        This intentionally rereads the trunk C times instead of once, but it
        reduces the three-level forest path to two FIA launches.  It is a
        useful bridge implementation for 6-8K CIS workloads where launch and
        merge overhead can dominate the theoretical trunk I/O saving.
        """

        prefix_output, prefix_lse = _attention(
            candidate_query,
            candidate_prefix_key,
            candidate_prefix_value,
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
        prefix_output = (
            prefix_output.permute(0, 2, 1, 3)
            .reshape(branches, query_heads, 1, head_dim)
        )
        prefix_lse = (
            prefix_lse.permute(0, 2, 1, 3)
            .reshape(branches, query_heads, 1, -1)
        )
        total_lse = torch.logaddexp(prefix_lse, unique_lse)
        output = (
            torch.exp(prefix_lse - total_lse).to(dtype) * prefix_output
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
    candidate_group_output, candidate_group_lse = candidate_group_decomposed()
    parallel_output, parallel_lse = parallel_decomposed()
    _synchronize()
    baseline_latency = _measure(baseline, warmup=warmup, iterations=iterations)
    shared_latency = _measure(decomposed, warmup=warmup, iterations=iterations)
    candidate_group_latency = _measure(
        candidate_group_decomposed,
        warmup=warmup,
        iterations=iterations,
    )
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
    candidate_group_kv_tokens = (
        candidate_count * (trunk_tokens + candidate_tokens)
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
        "candidate_group_forest": candidate_group_latency,
        "cis_forest_parallel": parallel_latency,
        "p50_speedup": baseline_latency["p50_ms"] / shared_latency["p50_ms"],
        "parallel_p50_speedup": (
            baseline_latency["p50_ms"] / parallel_latency["p50_ms"]
        ),
        "candidate_group_p50_speedup": (
            baseline_latency["p50_ms"] / candidate_group_latency["p50_ms"]
        ),
        "ideal_kv_read_reduction": baseline_kv_tokens / shared_kv_tokens,
        "candidate_group_ideal_kv_read_reduction": (
            baseline_kv_tokens / candidate_group_kv_tokens
        ),
        "maximum_output_error": float(
            (baseline_output.float() - shared_output.float()).abs().max().cpu()
        ),
        "maximum_lse_error": float(
            (baseline_lse.float() - shared_lse.float()).abs().max().cpu()
        ),
        "candidate_group_maximum_output_error": float(
            (baseline_output.float() - candidate_group_output.float()).abs().max().cpu()
        ),
        "candidate_group_maximum_lse_error": float(
            (baseline_lse.float() - candidate_group_lse.float()).abs().max().cpu()
        ),
        "parallel_maximum_output_error": float(
            (baseline_output.float() - parallel_output.float()).abs().max().cpu()
        ),
        "parallel_maximum_lse_error": float(
            (baseline_lse.float() - parallel_lse.float()).abs().max().cpu()
        ),
    }


def _screen_paged_candidate_group_shape(
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
    block_size: int = 128,
) -> dict[str, Any]:
    """Measure the two-FIA bridge against physical paged KV sharing.

    Every rollout block table references the same trunk blocks and the same
    candidate blocks as its siblings. The bridge passes those shared blocks
    once per candidate to FIA, then evaluates each branch's unique tail in a
    second FIA call and merges the partial outputs exactly.
    """

    for name, tokens in (
        ("trunk", trunk_tokens),
        ("candidate", candidate_tokens),
        ("unique", unique_tokens),
    ):
        if tokens <= 0 or tokens % block_size:
            raise ValueError(f"{name} tokens must be a positive block multiple")

    device = torch.device("npu:0")
    branches = candidate_count * rollout_count
    trunk_blocks = trunk_tokens // block_size
    candidate_blocks = candidate_tokens // block_size
    unique_blocks = unique_tokens // block_size
    total_blocks = (
        trunk_blocks
        + candidate_count * candidate_blocks
        + branches * unique_blocks
    )
    key_cache = torch.randn(
        total_blocks,
        block_size,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    query = torch.randn(
        branches, query_heads, 1, head_dim, device=device, dtype=dtype
    )

    trunk_ids = list(range(trunk_blocks))
    next_block = trunk_blocks
    candidate_ids = []
    for _ in range(candidate_count):
        ids = list(range(next_block, next_block + candidate_blocks))
        candidate_ids.append(ids)
        next_block += candidate_blocks
    unique_ids = []
    for _ in range(branches):
        ids = list(range(next_block, next_block + unique_blocks))
        unique_ids.append(ids)
        next_block += unique_blocks
    assert next_block == total_blocks

    full_rows = []
    prefix_rows = []
    tail_rows = []
    packed_rows = []
    for candidate_index in range(candidate_count):
        shared = trunk_ids + candidate_ids[candidate_index]
        prefix_rows.append(shared)
        packed_row = list(shared)
        for rollout_index in range(rollout_count):
            branch_index = candidate_index * rollout_count + rollout_index
            tail = unique_ids[branch_index]
            full_rows.append(shared + tail)
            tail_rows.append(tail)
            packed_row.extend(tail)
        packed_rows.append(packed_row)
    full_table = torch.tensor(full_rows, dtype=torch.int32, device=device)
    prefix_table = torch.tensor(prefix_rows, dtype=torch.int32, device=device)
    tail_table = torch.tensor(tail_rows, dtype=torch.int32, device=device)
    packed_table = torch.tensor(packed_rows, dtype=torch.int32, device=device)
    prefix_tokens = trunk_tokens + candidate_tokens
    full_tokens = prefix_tokens + unique_tokens
    packed_tokens = prefix_tokens + rollout_count * unique_tokens
    candidate_query = (
        query.squeeze(2)
        .reshape(candidate_count, rollout_count, query_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    candidate_query_bsh = query.reshape(
        candidate_count, rollout_count, query_heads * head_dim
    )

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        return _paged_attention(
            query,
            key_cache,
            value_cache,
            full_table,
            [full_tokens] * branches,
            query_heads=query_heads,
            kv_heads=kv_heads,
            block_size=block_size,
        )

    def native_baseline() -> torch.Tensor:
        return _native_paged_attention(
            query,
            key_cache,
            value_cache,
            full_table,
            [full_tokens] * branches,
            query_heads=query_heads,
            kv_heads=kv_heads,
        )

    def candidate_group() -> tuple[torch.Tensor, torch.Tensor]:
        prefix_output, prefix_lse = _paged_attention(
            candidate_query,
            key_cache,
            value_cache,
            prefix_table,
            [prefix_tokens] * candidate_count,
            query_heads=query_heads,
            kv_heads=kv_heads,
            block_size=block_size,
        )
        tail_output, tail_lse = _paged_attention(
            query,
            key_cache,
            value_cache,
            tail_table,
            [unique_tokens] * branches,
            query_heads=query_heads,
            kv_heads=kv_heads,
            block_size=block_size,
        )
        prefix_output = (
            prefix_output.permute(0, 2, 1, 3)
            .reshape(branches, query_heads, 1, head_dim)
        )
        prefix_lse = (
            prefix_lse.permute(0, 2, 1, 3)
            .reshape(branches, query_heads, 1, -1)
        )
        total_lse = torch.logaddexp(prefix_lse, tail_lse)
        output = (
            torch.exp(prefix_lse - total_lse).to(dtype) * prefix_output
            + torch.exp(tail_lse - total_lse).to(dtype) * tail_output
        )
        return output, total_lse

    packed_mask = torch.ones(
        rollout_count,
        packed_tokens,
        dtype=torch.bool,
        device=device,
    )
    packed_mask[:, :prefix_tokens] = False
    for rollout_index in range(rollout_count):
        start = prefix_tokens + rollout_index * unique_tokens
        packed_mask[rollout_index, start : start + unique_tokens] = False
    packed_batched_mask = packed_mask.unsqueeze(0).unsqueeze(1).expand(
        candidate_count, 1, -1, -1
    ).contiguous()

    def packed_candidate_group() -> tuple[torch.Tensor, torch.Tensor]:
        output, lse = _paged_attention(
            candidate_query,
            key_cache,
            value_cache,
            packed_table,
            [packed_tokens] * candidate_count,
            query_heads=query_heads,
            kv_heads=kv_heads,
            block_size=block_size,
            atten_mask=packed_mask,
        )
        return (
            output.permute(0, 2, 1, 3).reshape(
                branches, query_heads, 1, head_dim
            ),
            lse.permute(0, 2, 1, 3).reshape(branches, query_heads, 1, -1),
        )

    def packed_candidate_group_batched_mask() -> tuple[torch.Tensor, torch.Tensor]:
        output, lse = _paged_attention(
            candidate_query,
            key_cache,
            value_cache,
            packed_table,
            [packed_tokens] * candidate_count,
            query_heads=query_heads,
            kv_heads=kv_heads,
            block_size=block_size,
            atten_mask=packed_batched_mask,
        )
        return (
            output.permute(0, 2, 1, 3).reshape(
                branches, query_heads, 1, head_dim
            ),
            lse.permute(0, 2, 1, 3).reshape(branches, query_heads, 1, -1),
        )

    def packed_candidate_group_bsh() -> tuple[torch.Tensor, torch.Tensor]:
        output, lse = _paged_attention_bsh(
            candidate_query_bsh,
            key_cache,
            value_cache,
            packed_table,
            [packed_tokens] * candidate_count,
            query_heads=query_heads,
            kv_heads=kv_heads,
            block_size=block_size,
            atten_mask=packed_mask,
        )
        return (
            output.reshape(branches, query_heads, 1, head_dim),
            lse.permute(0, 2, 1, 3).reshape(branches, query_heads, 1, -1),
        )

    baseline_output, baseline_lse = baseline()
    native_output = native_baseline()
    shared_output, shared_lse = candidate_group()
    packed_output, packed_lse = packed_candidate_group()
    batched_mask_output, batched_mask_lse = packed_candidate_group_batched_mask()
    bsh_output, bsh_lse = packed_candidate_group_bsh()
    _synchronize()
    baseline_latency = _measure(baseline, warmup=warmup, iterations=iterations)
    native_latency = _measure(
        native_baseline, warmup=warmup, iterations=iterations
    )
    shared_latency = _measure(
        candidate_group,
        warmup=warmup,
        iterations=iterations,
    )
    packed_latency = _measure(
        packed_candidate_group,
        warmup=warmup,
        iterations=iterations,
    )
    batched_mask_latency = _measure(
        packed_candidate_group_batched_mask,
        warmup=warmup,
        iterations=iterations,
    )
    bsh_latency = _measure(
        packed_candidate_group_bsh,
        warmup=warmup,
        iterations=iterations,
    )
    baseline_kv_tokens = branches * full_tokens
    shared_kv_tokens = candidate_count * prefix_tokens + branches * unique_tokens
    return {
        "candidate_count": candidate_count,
        "rollout_count": rollout_count,
        "branches": branches,
        "trunk_tokens": trunk_tokens,
        "candidate_tokens": candidate_tokens,
        "unique_tokens": unique_tokens,
        "block_size": block_size,
        "physical_kv_blocks": total_blocks,
        "baseline": baseline_latency,
        "native_paged_attention_baseline": native_latency,
        "candidate_group_forest": shared_latency,
        "packed_candidate_group": packed_latency,
        "packed_candidate_group_batched_mask": batched_mask_latency,
        "packed_candidate_group_bsh": bsh_latency,
        "candidate_group_p50_speedup": (
            baseline_latency["p50_ms"] / shared_latency["p50_ms"]
        ),
        "packed_candidate_group_p50_speedup": (
            baseline_latency["p50_ms"] / packed_latency["p50_ms"]
        ),
        "packed_vs_native_pa_p50_speedup": (
            native_latency["p50_ms"] / packed_latency["p50_ms"]
        ),
        "packed_bsh_vs_native_pa_p50_speedup": (
            native_latency["p50_ms"] / bsh_latency["p50_ms"]
        ),
        "packed_candidate_group_batched_mask_p50_speedup": (
            baseline_latency["p50_ms"] / batched_mask_latency["p50_ms"]
        ),
        "candidate_group_ideal_kv_read_reduction": (
            baseline_kv_tokens / shared_kv_tokens
        ),
        "maximum_output_error": float(
            (baseline_output.float() - shared_output.float()).abs().max().cpu()
        ),
        "maximum_lse_error": float(
            (baseline_lse.float() - shared_lse.float()).abs().max().cpu()
        ),
        "native_maximum_output_error": float(
            (baseline_output.float() - native_output.float()).abs().max().cpu()
        ),
        "packed_maximum_output_error": float(
            (baseline_output.float() - packed_output.float()).abs().max().cpu()
        ),
        "packed_maximum_lse_error": float(
            (baseline_lse.float() - packed_lse.float()).abs().max().cpu()
        ),
        "batched_mask_maximum_output_error": float(
            (baseline_output.float() - batched_mask_output.float()).abs().max().cpu()
        ),
        "batched_mask_maximum_lse_error": float(
            (baseline_lse.float() - batched_mask_lse.float()).abs().max().cpu()
        ),
        "bsh_maximum_output_error": float(
            (baseline_output.float() - bsh_output.float()).abs().max().cpu()
        ),
        "bsh_maximum_lse_error": float(
            (baseline_lse.float() - bsh_lse.float()).abs().max().cpu()
        ),
    }


def _screen_paged_mixed_decode_shape(
    *,
    cis_steps: int,
    total_batch: int,
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
    block_size: int = 128,
) -> dict[str, Any]:
    """Measure packed CIS groups while preserving a saturated decode batch."""

    branches = cis_steps * candidate_count * rollout_count
    if branches > total_batch:
        raise ValueError("CIS branches cannot exceed the total decode batch")
    for name, tokens in (
        ("trunk", trunk_tokens),
        ("candidate", candidate_tokens),
        ("unique", unique_tokens),
    ):
        if tokens <= 0 or tokens % block_size:
            raise ValueError(f"{name} tokens must be a positive block multiple")

    ordinary = total_batch - branches
    device = torch.device("npu:0")
    trunk_blocks = trunk_tokens // block_size
    candidate_blocks = candidate_tokens // block_size
    unique_blocks = unique_tokens // block_size
    full_tokens = trunk_tokens + candidate_tokens + unique_tokens
    full_blocks = full_tokens // block_size
    next_block = 0
    branch_rows: list[list[int]] = []
    packed_rows: list[list[int]] = []
    for _ in range(cis_steps):
        trunk_ids = list(range(next_block, next_block + trunk_blocks))
        next_block += trunk_blocks
        for _ in range(candidate_count):
            candidate_ids = list(
                range(next_block, next_block + candidate_blocks)
            )
            next_block += candidate_blocks
            shared = trunk_ids + candidate_ids
            packed_row = list(shared)
            for _ in range(rollout_count):
                tail = list(range(next_block, next_block + unique_blocks))
                next_block += unique_blocks
                branch_rows.append(shared + tail)
                packed_row.extend(tail)
            packed_rows.append(packed_row)
    ordinary_rows = []
    for _ in range(ordinary):
        row = list(range(next_block, next_block + full_blocks))
        next_block += full_blocks
        ordinary_rows.append(row)

    key_cache = torch.randn(
        next_block,
        block_size,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    query = torch.randn(
        total_batch, query_heads, 1, head_dim, device=device, dtype=dtype
    )
    branch_query = query[:branches]
    ordinary_query = query[branches:]
    baseline_table = torch.tensor(
        branch_rows + ordinary_rows, dtype=torch.int32, device=device
    )
    packed_table = torch.tensor(packed_rows, dtype=torch.int32, device=device)
    ordinary_table = (
        torch.tensor(ordinary_rows, dtype=torch.int32, device=device)
        if ordinary_rows
        else None
    )
    candidate_groups = cis_steps * candidate_count
    packed_query = (
        branch_query.squeeze(2)
        .reshape(candidate_groups, rollout_count, query_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    packed_query_bsh = branch_query.reshape(
        candidate_groups, rollout_count, query_heads * head_dim
    )
    prefix_tokens = trunk_tokens + candidate_tokens
    packed_tokens = prefix_tokens + rollout_count * unique_tokens
    packed_mask = torch.ones(
        rollout_count,
        packed_tokens,
        dtype=torch.bool,
        device=device,
    )
    packed_mask[:, :prefix_tokens] = False
    for rollout_index in range(rollout_count):
        start = prefix_tokens + rollout_index * unique_tokens
        packed_mask[rollout_index, start : start + unique_tokens] = False

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        return _paged_attention(
            query,
            key_cache,
            value_cache,
            baseline_table,
            [full_tokens] * total_batch,
            query_heads=query_heads,
            kv_heads=kv_heads,
            block_size=block_size,
        )

    def native_baseline() -> torch.Tensor:
        return _native_paged_attention(
            query,
            key_cache,
            value_cache,
            baseline_table,
            [full_tokens] * total_batch,
            query_heads=query_heads,
            kv_heads=kv_heads,
        )

    def split_packed() -> tuple[torch.Tensor, torch.Tensor]:
        packed_output, packed_lse = _paged_attention(
            packed_query,
            key_cache,
            value_cache,
            packed_table,
            [packed_tokens] * candidate_groups,
            query_heads=query_heads,
            kv_heads=kv_heads,
            block_size=block_size,
            atten_mask=packed_mask,
        )
        branch_output = packed_output.permute(0, 2, 1, 3).reshape(
            branches, query_heads, 1, head_dim
        )
        branch_lse = packed_lse.permute(0, 2, 1, 3).reshape(
            branches, query_heads, 1, -1
        )
        if ordinary_table is None:
            return branch_output, branch_lse
        ordinary_output, ordinary_lse = _paged_attention(
            ordinary_query,
            key_cache,
            value_cache,
            ordinary_table,
            [full_tokens] * ordinary,
            query_heads=query_heads,
            kv_heads=kv_heads,
            block_size=block_size,
        )
        return (
            torch.cat((branch_output, ordinary_output), dim=0),
            torch.cat((branch_lse, ordinary_lse), dim=0),
        )

    def split_packed_bsh() -> tuple[torch.Tensor, torch.Tensor]:
        packed_output, packed_lse = _paged_attention_bsh(
            packed_query_bsh,
            key_cache,
            value_cache,
            packed_table,
            [packed_tokens] * candidate_groups,
            query_heads=query_heads,
            kv_heads=kv_heads,
            block_size=block_size,
            atten_mask=packed_mask,
        )
        branch_output = packed_output.reshape(
            branches, query_heads, 1, head_dim
        )
        branch_lse = packed_lse.permute(0, 2, 1, 3).reshape(
            branches, query_heads, 1, -1
        )
        if ordinary_table is None:
            return branch_output, branch_lse
        ordinary_output, ordinary_lse = _paged_attention(
            ordinary_query,
            key_cache,
            value_cache,
            ordinary_table,
            [full_tokens] * ordinary,
            query_heads=query_heads,
            kv_heads=kv_heads,
            block_size=block_size,
        )
        return (
            torch.cat((branch_output, ordinary_output), dim=0),
            torch.cat((branch_lse, ordinary_lse), dim=0),
        )

    baseline_output, baseline_lse = baseline()
    native_output = native_baseline()
    packed_output, packed_lse = split_packed()
    packed_bsh_output, packed_bsh_lse = split_packed_bsh()
    _synchronize()
    baseline_latency = _measure(baseline, warmup=warmup, iterations=iterations)
    native_latency = _measure(
        native_baseline, warmup=warmup, iterations=iterations
    )
    packed_latency = _measure(split_packed, warmup=warmup, iterations=iterations)
    packed_bsh_latency = _measure(
        split_packed_bsh, warmup=warmup, iterations=iterations
    )
    return {
        "cis_steps": cis_steps,
        "candidate_count": candidate_count,
        "rollout_count": rollout_count,
        "cis_branches": branches,
        "ordinary_requests": ordinary,
        "total_batch": total_batch,
        "trunk_tokens": trunk_tokens,
        "candidate_tokens": candidate_tokens,
        "unique_tokens": unique_tokens,
        "baseline": baseline_latency,
        "native_paged_attention_baseline": native_latency,
        "split_packed_candidate_group": packed_latency,
        "split_packed_candidate_group_bsh": packed_bsh_latency,
        "p50_speedup": baseline_latency["p50_ms"] / packed_latency["p50_ms"],
        "packed_vs_native_pa_p50_speedup": (
            native_latency["p50_ms"] / packed_latency["p50_ms"]
        ),
        "packed_bsh_vs_native_pa_p50_speedup": (
            native_latency["p50_ms"] / packed_bsh_latency["p50_ms"]
        ),
        "maximum_output_error": float(
            (baseline_output.float() - packed_output.float()).abs().max().cpu()
        ),
        "maximum_lse_error": float(
            (baseline_lse.float() - packed_lse.float()).abs().max().cpu()
        ),
        "native_maximum_output_error": float(
            (baseline_output.float() - native_output.float()).abs().max().cpu()
        ),
        "bsh_maximum_output_error": float(
            (baseline_output.float() - packed_bsh_output.float()).abs().max().cpu()
        ),
        "bsh_maximum_lse_error": float(
            (baseline_lse.float() - packed_bsh_lse.float()).abs().max().cpu()
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
    parser.add_argument(
        "--paged-cis-forest",
        action="store_true",
        help="measure candidate-group factoring on physically shared paged KV",
    )
    parser.add_argument(
        "--mixed-decode",
        action="store_true",
        help="measure packed CIS groups inside a fixed saturated decode batch",
    )
    parser.add_argument("--cis-steps", default="1,2,3,4,5")
    parser.add_argument("--total-decode-batch", type=int, default=256)
    parser.add_argument("--candidate-counts", default="4,8,15")
    parser.add_argument("--rollout-count", type=int, default=3)
    parser.add_argument("--candidate-tokens", type=int, default=128)
    parser.add_argument("--unique-tokens", type=int, default=128)
    args = parser.parse_args()

    results = []
    if args.mixed_decode:
        if not args.paged_cis_forest:
            parser.error("--mixed-decode requires --paged-cis-forest")
        candidate_counts = list(map(int, args.candidate_counts.split(",")))
        prefix_lengths = list(map(int, args.prefix_lengths.split(",")))
        if len(candidate_counts) != 1 or len(prefix_lengths) != 1:
            parser.error(
                "--mixed-decode requires one candidate count and one prefix length"
            )
        for cis_steps in map(int, args.cis_steps.split(",")):
            results.append(
                _screen_paged_mixed_decode_shape(
                    cis_steps=cis_steps,
                    total_batch=args.total_decode_batch,
                    candidate_count=candidate_counts[0],
                    rollout_count=args.rollout_count,
                    trunk_tokens=prefix_lengths[0],
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
            torch.npu.empty_cache()
    elif args.paged_cis_forest:
        for prefix_tokens in map(int, args.prefix_lengths.split(",")):
            for candidate_count in map(int, args.candidate_counts.split(",")):
                results.append(
                    _screen_paged_candidate_group_shape(
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
    elif args.cis_forest:
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
