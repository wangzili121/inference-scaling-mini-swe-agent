"""Plan exact candidate-sibling packing for paged decode attention."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ForestBranchLayout:
    """One scheduled rollout branch and its physical paged-KV layout."""

    request_index: int
    query_start: int
    query_length: int
    sequence_length: int
    block_ids: tuple[int, ...]
    group_id: str | None = None
    branch_index: int | None = None
    group_size: int | None = None


@dataclass(frozen=True, slots=True)
class PackedForestBucket:
    """Groups with one sibling count that fit one BNSD FIA invocation."""

    sibling_count: int
    query_indices: tuple[int, ...]
    request_indices: tuple[int, ...]
    block_tables: tuple[tuple[int, ...], ...]
    attention_masks: tuple[tuple[tuple[bool, ...], ...], ...] | None
    kv_sequence_lengths: tuple[int, ...]
    shared_prefix_tokens: tuple[int, ...]
    tail_tokens: tuple[tuple[int, ...], ...]
    max_shared_blocks: int
    max_tail_blocks: int
    estimated_saved_kv_token_reads: int
    estimated_masked_tail_token_slots: int
    padded_to_logical_block_ratio: float


@dataclass(frozen=True, slots=True)
class PackedForestPlan:
    """A lossless partition of scheduled requests into packed and normal work."""

    buckets: tuple[PackedForestBucket, ...]
    normal_request_indices: tuple[int, ...]
    estimated_saved_kv_token_reads: int
    estimated_masked_tail_token_slots: int
    padded_to_logical_block_ratio: float

    @property
    def enabled(self) -> bool:
        return bool(self.buckets)


@dataclass(frozen=True, slots=True)
class ForestWindowBranch:
    """Scheduler-visible rollout branch before physical KV inspection."""

    group_id: str | None
    branch_index: int | None
    group_size: int | None
    query_length: int
    shared_prefix_tokens: int


@dataclass(frozen=True, slots=True)
class ForestWindowEstimate:
    """Whether a decode-only window has enough complete sibling work."""

    enabled: bool
    complete_groups: int
    packed_queries: int
    total_queries: int
    estimated_saved_kv_token_reads: int
    packed_query_fraction: float


def estimate_forest_decode_window(
    branches: Sequence[ForestWindowBranch],
    *,
    block_size: int = 128,
    minimum_saved_kv_token_reads: int = 196_608,
    minimum_packed_query_fraction: float = 0.75,
) -> ForestWindowEstimate:
    """Estimate when holding new prefills can expose useful Forest decode.

    This deliberately uses only scheduler-visible metadata. The attention
    planner remains the final authority because it additionally verifies the
    physical block-table sharing before changing the kernel path.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if minimum_saved_kv_token_reads < 0:
        raise ValueError("minimum saved reads must be non-negative")
    if not 0.0 <= minimum_packed_query_fraction <= 1.0:
        raise ValueError("minimum packed query fraction must be in [0, 1]")

    grouped: dict[str, list[ForestWindowBranch]] = {}
    for branch in branches:
        if branch.group_id is not None and branch.query_length == 1:
            grouped.setdefault(branch.group_id, []).append(branch)

    complete_groups = 0
    packed_queries = 0
    saved_reads = 0
    for group in grouped.values():
        expected = group[0].group_size
        if expected is None or expected < 2 or len(group) != expected:
            continue
        if any(
            branch.group_size != expected
            or branch.branch_index is None
            or branch.shared_prefix_tokens <= 0
            for branch in group
        ):
            continue
        indices = sorted(int(branch.branch_index) for branch in group)
        if indices != list(range(expected)):
            continue
        shared_tokens = min(branch.shared_prefix_tokens for branch in group)
        shared_tokens = (shared_tokens // block_size) * block_size
        if shared_tokens <= 0:
            continue
        complete_groups += 1
        packed_queries += expected
        saved_reads += shared_tokens * (expected - 1)

    total_queries = len(branches)
    packed_fraction = packed_queries / total_queries if total_queries else 0.0
    return ForestWindowEstimate(
        enabled=(
            saved_reads >= minimum_saved_kv_token_reads
            and packed_fraction >= minimum_packed_query_fraction
        ),
        complete_groups=complete_groups,
        packed_queries=packed_queries,
        total_queries=total_queries,
        estimated_saved_kv_token_reads=saved_reads,
        packed_query_fraction=packed_fraction,
    )


@dataclass(frozen=True, slots=True)
class _EligibleGroup:
    sibling_count: int
    branches: tuple[ForestBranchLayout, ...]
    shared_blocks: tuple[int, ...]
    tail_blocks: tuple[tuple[int, ...], ...]
    tail_tokens: tuple[int, ...]


def _common_full_blocks(
    branches: Sequence[ForestBranchLayout],
    block_size: int,
) -> tuple[int, ...]:
    full_counts = [branch.sequence_length // block_size for branch in branches]
    common_limit = min(full_counts, default=0)
    common = 0
    while common < common_limit:
        block_id = branches[0].block_ids[common]
        if any(branch.block_ids[common] != block_id for branch in branches[1:]):
            break
        common += 1
    return branches[0].block_ids[:common]


def _eligible_groups(
    branches: Sequence[ForestBranchLayout],
    *,
    block_size: int,
    minimum_shared_tokens: int,
) -> tuple[_EligibleGroup, ...]:
    by_group: dict[str, list[ForestBranchLayout]] = {}
    for branch in branches:
        if branch.group_id is None:
            continue
        by_group.setdefault(branch.group_id, []).append(branch)

    eligible = []
    for group in by_group.values():
        expected = group[0].group_size
        if expected is None or expected < 2 or len(group) != expected:
            continue
        if any(
            branch.group_size != expected
            or branch.branch_index is None
            or branch.query_length != 1
            or branch.sequence_length <= 0
            for branch in group
        ):
            continue
        ordered = sorted(group, key=lambda branch: int(branch.branch_index or 0))
        if [branch.branch_index for branch in ordered] != list(range(expected)):
            continue
        if any(
            len(branch.block_ids) < ceil(branch.sequence_length / block_size)
            for branch in ordered
        ):
            continue
        shared_blocks = _common_full_blocks(ordered, block_size)
        if not shared_blocks:
            continue
        shared_tokens = len(shared_blocks) * block_size
        if shared_tokens < minimum_shared_tokens:
            continue
        tail_blocks = []
        tail_tokens = []
        for branch in ordered:
            used_blocks = ceil(branch.sequence_length / block_size)
            tail_blocks.append(
                branch.block_ids[len(shared_blocks) : used_blocks]
            )
            tail_tokens.append(branch.sequence_length - shared_tokens)
        eligible.append(
            _EligibleGroup(
                sibling_count=expected,
                branches=tuple(ordered),
                shared_blocks=tuple(shared_blocks),
                tail_blocks=tuple(tail_blocks),
                tail_tokens=tuple(tail_tokens),
            )
        )
    return tuple(eligible)


def _build_bucket(
    groups: Sequence[_EligibleGroup],
    *,
    block_size: int,
    materialize_attention_masks: bool,
) -> PackedForestBucket:
    sibling_count = groups[0].sibling_count
    max_shared_blocks = max(len(group.shared_blocks) for group in groups)
    max_tail_blocks = max(
        len(tail) for group in groups for tail in group.tail_blocks
    )
    row_blocks = max_shared_blocks + sibling_count * max_tail_blocks
    row_tokens = row_blocks * block_size
    block_tables = []
    attention_masks = []
    query_indices = []
    request_indices = []
    shared_prefix_tokens = []
    tail_tokens_by_group = []
    saved_token_reads = 0
    masked_tail_token_slots = 0
    logical_row_blocks = 0

    for group in groups:
        padding_block = group.shared_blocks[0]
        row = list(group.shared_blocks)
        row.extend([padding_block] * (max_shared_blocks - len(row)))
        for tail in group.tail_blocks:
            row.extend(tail)
            row.extend([padding_block] * (max_tail_blocks - len(tail)))
        assert len(row) == row_blocks
        block_tables.append(tuple(row))

        shared_tokens = len(group.shared_blocks) * block_size
        if materialize_attention_masks:
            mask = [[True] * row_tokens for _ in range(sibling_count)]
            for branch_index, tail_tokens in enumerate(group.tail_tokens):
                mask[branch_index][:shared_tokens] = [False] * shared_tokens
                tail_start = (
                    max_shared_blocks + branch_index * max_tail_blocks
                ) * block_size
                mask[branch_index][tail_start : tail_start + tail_tokens] = (
                    [False] * tail_tokens
                )
            attention_masks.append(tuple(tuple(row_mask) for row_mask in mask))
        query_indices.extend(branch.query_start for branch in group.branches)
        request_indices.extend(branch.request_index for branch in group.branches)
        shared_prefix_tokens.append(shared_tokens)
        tail_tokens_by_group.append(group.tail_tokens)
        saved_token_reads += shared_tokens * (sibling_count - 1)
        masked_tail_token_slots += (sibling_count - 1) * sum(
            group.tail_tokens
        )
        logical_row_blocks += len(group.shared_blocks) + sibling_count * max(
            (len(tail) for tail in group.tail_blocks),
            default=0,
        )

    padded_row_blocks = len(groups) * row_blocks

    return PackedForestBucket(
        sibling_count=sibling_count,
        query_indices=tuple(query_indices),
        request_indices=tuple(request_indices),
        block_tables=tuple(block_tables),
        attention_masks=(
            tuple(attention_masks) if materialize_attention_masks else None
        ),
        kv_sequence_lengths=(row_tokens,) * len(groups),
        shared_prefix_tokens=tuple(shared_prefix_tokens),
        tail_tokens=tuple(tail_tokens_by_group),
        max_shared_blocks=max_shared_blocks,
        max_tail_blocks=max_tail_blocks,
        estimated_saved_kv_token_reads=saved_token_reads,
        estimated_masked_tail_token_slots=masked_tail_token_slots,
        padded_to_logical_block_ratio=(
            padded_row_blocks / logical_row_blocks
            if logical_row_blocks
            else 1.0
        ),
    )


def build_packed_forest_plan(
    branches: Sequence[ForestBranchLayout],
    *,
    block_size: int = 128,
    minimum_shared_tokens: int = 4096,
    minimum_saved_kv_token_reads: int = 196_608,
    minimum_packed_query_fraction: float = 0.75,
    maximum_padding_ratio: float = 1.25,
    maximum_masked_tail_to_saved_ratio: float = 1.0,
    materialize_attention_masks: bool = True,
) -> PackedForestPlan:
    """Build a dynamically gated exact packed-attention plan.

    The plan combines each candidate's R rollout block tables into one logical
    KV row. A per-group mask lets query ``r`` see the common physical prefix
    and only tail ``r``. Positional information is already encoded in Q/K, so
    this layout changes neither logits nor the Conditional IS distribution.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if minimum_shared_tokens < 0 or minimum_saved_kv_token_reads < 0:
        raise ValueError("forest attention thresholds must be non-negative")
    if not 0.0 <= minimum_packed_query_fraction <= 1.0:
        raise ValueError("minimum_packed_query_fraction must be in [0, 1]")
    if maximum_padding_ratio < 1.0:
        raise ValueError("maximum_padding_ratio must be at least 1")
    if maximum_masked_tail_to_saved_ratio < 0.0:
        raise ValueError(
            "maximum_masked_tail_to_saved_ratio must be non-negative"
        )
    request_indices = [branch.request_index for branch in branches]
    if len(set(request_indices)) != len(request_indices):
        raise ValueError("request indices must be unique")

    eligible = _eligible_groups(
        branches,
        block_size=block_size,
        minimum_shared_tokens=minimum_shared_tokens,
    )
    total_saved = sum(
        len(group.shared_blocks) * block_size * (group.sibling_count - 1)
        for group in eligible
    )
    total_masked_tail_slots = sum(
        (group.sibling_count - 1) * sum(group.tail_tokens)
        for group in eligible
    )
    eligible_query_count = sum(group.sibling_count for group in eligible)
    packed_query_fraction = (
        eligible_query_count / len(branches) if branches else 0.0
    )
    if (
        total_saved < minimum_saved_kv_token_reads
        or packed_query_fraction < minimum_packed_query_fraction
        or total_masked_tail_slots
        > total_saved * maximum_masked_tail_to_saved_ratio
    ):
        return PackedForestPlan(
            (),
            tuple(sorted(request_indices)),
            total_saved,
            total_masked_tail_slots,
            1.0,
        )

    groups_by_siblings: dict[int, list[_EligibleGroup]] = {}
    packed_request_indices = set()
    for group in eligible:
        groups_by_siblings.setdefault(group.sibling_count, []).append(group)
        packed_request_indices.update(
            branch.request_index for branch in group.branches
        )
    buckets = tuple(
        _build_bucket(
            groups,
            block_size=block_size,
            materialize_attention_masks=materialize_attention_masks,
        )
        for _, groups in sorted(groups_by_siblings.items())
    )
    normal = tuple(
        sorted(index for index in request_indices if index not in packed_request_indices)
    )
    logical_weight = sum(len(bucket.block_tables) for bucket in buckets)
    padding_ratio = (
        sum(
            bucket.padded_to_logical_block_ratio * len(bucket.block_tables)
            for bucket in buckets
        )
        / logical_weight
        if logical_weight
        else 1.0
    )
    if padding_ratio > maximum_padding_ratio:
        return PackedForestPlan(
            (),
            tuple(sorted(request_indices)),
            total_saved,
            total_masked_tail_slots,
            padding_ratio,
        )
    return PackedForestPlan(
        buckets,
        normal,
        total_saved,
        total_masked_tail_slots,
        padding_ratio,
    )
