from __future__ import annotations

import pytest

from inference_scaling.arllm.backends.packed_forest_attention import (
    ForestBranchLayout,
    ForestWindowBranch,
    build_packed_forest_plan,
    estimate_forest_decode_window,
)


def _branch(
    request_index: int,
    branch_index: int,
    blocks: tuple[int, ...],
    *,
    group_id: str = "job:step:0:candidate:0",
    group_size: int = 3,
    sequence_length: int = 266,
) -> ForestBranchLayout:
    return ForestBranchLayout(
        request_index=request_index,
        query_start=request_index,
        query_length=1,
        sequence_length=sequence_length,
        block_ids=blocks,
        group_id=group_id,
        branch_index=branch_index,
        group_size=group_size,
    )


def test_builds_exact_candidate_sibling_layout() -> None:
    branches = [
        _branch(0, 0, (10, 11, 20)),
        _branch(1, 1, (10, 11, 21)),
        _branch(2, 2, (10, 11, 22)),
        ForestBranchLayout(3, 3, 1, 300, (30, 31, 32)),
    ]

    plan = build_packed_forest_plan(
        branches,
        block_size=128,
        minimum_shared_tokens=128,
        minimum_saved_kv_token_reads=1,
    )

    assert plan.enabled
    assert plan.normal_request_indices == (3,)
    assert plan.estimated_saved_kv_token_reads == 512
    assert plan.estimated_masked_tail_token_slots == 60
    assert plan.padded_to_logical_block_ratio == 1.0
    assert len(plan.buckets) == 1
    bucket = plan.buckets[0]
    assert bucket.sibling_count == 3
    assert bucket.query_indices == (0, 1, 2)
    assert bucket.request_indices == (0, 1, 2)
    assert bucket.block_tables == ((10, 11, 20, 21, 22),)
    assert bucket.kv_sequence_lengths == (640,)
    assert bucket.shared_prefix_tokens == (256,)
    assert bucket.tail_tokens == ((10, 10, 10),)

    mask = bucket.attention_masks[0]
    assert not any(mask[0][:256])
    assert not any(mask[0][256:266])
    assert all(mask[0][266:])
    assert not any(mask[1][:256])
    assert not any(mask[1][384:394])
    assert all(mask[1][256:384])
    assert all(mask[1][394:])


def test_pads_variable_physical_layout_but_masks_padding() -> None:
    branches = [
        _branch(0, 0, (1, 2, 10), sequence_length=266),
        _branch(1, 1, (1, 2, 11, 12), sequence_length=394),
        _branch(2, 2, (1, 2, 13), sequence_length=266),
    ]

    bucket = build_packed_forest_plan(
        branches,
        block_size=128,
        minimum_shared_tokens=1,
        minimum_saved_kv_token_reads=1,
    ).buckets[0]

    assert bucket.block_tables == ((1, 2, 10, 1, 11, 12, 13, 1),)
    assert bucket.kv_sequence_lengths == (1024,)
    assert bucket.tail_tokens == ((10, 138, 10),)
    mask = bucket.attention_masks[0]
    assert not any(mask[1][512:650])
    assert all(mask[1][256:512])
    assert all(mask[1][650:])


def test_dynamic_gate_counts_avoided_shared_token_reads() -> None:
    branches = [
        _branch(0, 0, (10, 11, 20)),
        _branch(1, 1, (10, 11, 21)),
        _branch(2, 2, (10, 11, 22)),
    ]

    plan = build_packed_forest_plan(
        branches,
        block_size=128,
        minimum_shared_tokens=128,
        minimum_saved_kv_token_reads=513,
    )

    assert not plan.enabled
    assert plan.estimated_saved_kv_token_reads == 512
    assert plan.normal_request_indices == (0, 1, 2)


def test_runtime_plan_can_skip_host_attention_mask_materialization() -> None:
    branches = [
        _branch(0, 0, (10, 11, 20)),
        _branch(1, 1, (10, 11, 21)),
        _branch(2, 2, (10, 11, 22)),
    ]

    bucket = build_packed_forest_plan(
        branches,
        block_size=128,
        minimum_shared_tokens=128,
        minimum_saved_kv_token_reads=1,
        materialize_attention_masks=False,
    ).buckets[0]

    assert bucket.attention_masks is None
    assert bucket.max_shared_blocks == 2
    assert bucket.max_tail_blocks == 1
    assert bucket.kv_sequence_lengths == (640,)


def test_skips_sparse_packed_subset_that_would_split_the_batch() -> None:
    branches = [
        _branch(0, 0, (10, 11, 20)),
        _branch(1, 1, (10, 11, 21)),
        _branch(2, 2, (10, 11, 22)),
    ] + [
        ForestBranchLayout(index, index, 1, 266, (index, index + 1, index + 2))
        for index in range(3, 8)
    ]

    plan = build_packed_forest_plan(
        branches,
        block_size=128,
        minimum_shared_tokens=128,
        minimum_saved_kv_token_reads=1,
    )

    assert not plan.enabled
    assert plan.normal_request_indices == tuple(range(8))


@pytest.mark.parametrize(
    "mutation",
    [
        {"query_length": 2},
        {"group_size": 4},
        {"branch_index": None},
    ],
)
def test_incomplete_or_non_decode_group_stays_on_normal_path(mutation) -> None:
    values = {
        "request_index": 0,
        "query_start": 0,
        "query_length": 1,
        "sequence_length": 266,
        "block_ids": (10, 11, 20),
        "group_id": "group",
        "branch_index": 0,
        "group_size": 3,
    }
    values.update(mutation)
    branches = [
        ForestBranchLayout(**values),
        _branch(1, 1, (10, 11, 21), group_id="group"),
        _branch(2, 2, (10, 11, 22), group_id="group"),
    ]

    plan = build_packed_forest_plan(
        branches,
        minimum_shared_tokens=1,
        minimum_saved_kv_token_reads=1,
    )

    assert not plan.enabled
    assert plan.normal_request_indices == (0, 1, 2)


def test_duplicate_request_index_is_rejected() -> None:
    branches = [
        _branch(0, 0, (10, 11, 20)),
        _branch(0, 1, (10, 11, 21)),
    ]
    with pytest.raises(ValueError, match="request indices must be unique"):
        build_packed_forest_plan(branches)


def test_forest_window_requires_complete_dense_sibling_groups() -> None:
    branches = [
        ForestWindowBranch("a", index, 3, 1, 8193) for index in range(3)
    ] + [
        ForestWindowBranch("b", index, 3, 1, 8193) for index in range(3)
    ]

    estimate = estimate_forest_decode_window(
        branches,
        minimum_saved_kv_token_reads=32_768,
    )

    assert estimate.enabled
    assert estimate.complete_groups == 2
    assert estimate.packed_queries == 6
    assert estimate.total_queries == 6
    assert estimate.estimated_saved_kv_token_reads == 32_768
    assert estimate.packed_query_fraction == 1.0


def test_forest_window_counts_incomplete_and_ordinary_decode_as_normal() -> None:
    branches = [
        ForestWindowBranch("a", 0, 3, 1, 16_384),
        ForestWindowBranch("a", 1, 3, 1, 16_384),
        ForestWindowBranch(None, None, None, 1, 16_384),
    ]

    estimate = estimate_forest_decode_window(
        branches,
        minimum_saved_kv_token_reads=1,
        minimum_packed_query_fraction=0.5,
    )

    assert not estimate.enabled
    assert estimate.complete_groups == 0
    assert estimate.packed_queries == 0
    assert estimate.total_queries == 3
