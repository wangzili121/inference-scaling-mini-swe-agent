"""Pure policy for admitting forked Conditional-IS rollout bundles.

A tree owns dependencies and admission credits. Individual rollout requests
still enter vLLM's ordinary continuous-batching scheduler after admission.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import ceil


@dataclass(frozen=True, slots=True)
class ReadyBundle:
    tree_id: str
    parent_handle: str
    child_count: int


@dataclass(slots=True)
class TreeState:
    tree_id: str
    order: int
    candidate_count: int
    candidates_completed: int = 0
    terminal_candidates: int = 0
    ready: deque[ReadyBundle] = field(default_factory=deque)
    active_children: int = 0
    released_children: int = 0
    finished_children: int = 0

    @property
    def candidate_phase_closed(self) -> bool:
        return self.candidates_completed == self.candidate_count

    @property
    def remaining_children(self) -> int:
        return self.active_children + sum(
            bundle.child_count for bundle in self.ready
        )


class TreeFrontierPolicy:
    """Bound release by per-tree branches while refilling idle engine capacity."""

    def __init__(
        self,
        *,
        max_num_seqs: int,
        target_fraction: float = 1.0,
        kv_high_watermark: float = 0.92,
        max_bundles_per_tick: int = 8,
    ) -> None:
        if max_num_seqs <= 0 or max_bundles_per_tick <= 0:
            raise ValueError("sequence and bundle limits must be positive")
        if not 0 < target_fraction <= 1 or not 0 < kv_high_watermark < 1:
            raise ValueError("fractions must be between zero and one")
        self.max_num_seqs = max_num_seqs
        self.target_fraction = target_fraction
        self.kv_high_watermark = kv_high_watermark
        self.max_bundles_per_tick = max_bundles_per_tick
        self.trees: dict[str, TreeState] = {}
        self._next_order = 0

    def register_tree(self, tree_id: str, candidate_count: int) -> TreeState:
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        existing = self.trees.get(tree_id)
        if existing is not None:
            if existing.candidate_count != candidate_count:
                raise ValueError("inconsistent candidate count for CIS tree")
            return existing
        state = TreeState(tree_id, self._next_order, candidate_count)
        self._next_order += 1
        self.trees[tree_id] = state
        return state

    def parent_completed(
        self,
        tree_id: str,
        parent_handle: str,
        *,
        child_count: int,
        terminal: bool,
    ) -> TreeState:
        state = self.trees[tree_id]
        if state.candidate_phase_closed:
            raise RuntimeError("CIS tree received too many completed candidates")
        if child_count < 0:
            raise ValueError("fork parent cannot have negative surviving children")
        state.candidates_completed += 1
        if terminal:
            state.terminal_candidates += 1
        elif child_count:
            state.ready.append(ReadyBundle(tree_id, parent_handle, child_count))
        self._retire_if_done(state)
        return state

    def parked_child_cancelled(self, tree_id: str, parent_handle: str) -> None:
        """Remove one dormant child from a not-yet-released parent bundle."""

        state = self.trees.get(tree_id)
        if state is None:
            return
        for index, bundle in enumerate(state.ready):
            if bundle.parent_handle != parent_handle:
                continue
            if bundle.child_count == 1:
                del state.ready[index]
            else:
                state.ready[index] = ReadyBundle(
                    tree_id, parent_handle, bundle.child_count - 1
                )
            self._retire_if_done(state)
            return

    def child_finished(self, tree_id: str) -> None:
        state = self.trees.get(tree_id)
        if state is None or state.active_children <= 0:
            raise RuntimeError("finished CIS child has no active tree credit")
        state.active_children -= 1
        state.finished_children += 1
        self._retire_if_done(state)

    def choose_bundles(
        self,
        *,
        running: int,
        waiting: int,
        kv_usage: float,
    ) -> list[ReadyBundle]:
        if running < 0 or waiting < 0 or not 0 <= kv_usage <= 1:
            raise ValueError("invalid engine telemetry")
        ready_trees = [state for state in self.trees.values() if state.ready]
        if not ready_trees:
            return []
        target = max(1, ceil(self.max_num_seqs * self.target_fraction))
        runnable = running + waiting
        first_width = min(state.ready[0].child_count for state in ready_trees)
        deficit = max(0, target - runnable)
        tick_budget = (
            min(self.max_bundles_per_tick, max(1, ceil(deficit / first_width)))
            if deficit
            else 0
        )
        admitted: list[ReadyBundle] = []
        for _ in range(tick_budget):
            if runnable >= target:
                break
            active_trees = max(1, len(self.trees))
            base_quota = max(1, ceil(target / active_trees))
            # Refill real idle capacity without imposing a fixed step cap.
            refill = max(0, ceil((target - runnable) / active_trees))
            quota = base_quota + refill
            eligible: list[TreeState] = []
            for state in ready_trees:
                if not state.ready:
                    continue
                width = state.ready[0].child_count
                progress_needed = (
                    state.candidate_phase_closed
                    and state.active_children == 0
                )
                if kv_usage >= self.kv_high_watermark and not progress_needed:
                    continue
                # Preserve the candidate-phase barrier while the engine is
                # already fed. Early fork is only a work-conserving refill.
                if (
                    not state.candidate_phase_closed
                    and (runnable >= target or waiting > 0)
                ):
                    continue
                if (
                    state.active_children + width <= quota
                    or progress_needed
                    or (runnable < target and waiting == 0)
                ):
                    eligible.append(state)
            if not eligible:
                break
            chosen = min(
                eligible,
                key=lambda state: (
                    0
                    if state.candidate_phase_closed
                    and state.remaining_children <= 2 * state.ready[0].child_count
                    else 1,
                    state.active_children / max(1, quota),
                    state.order,
                ),
            )
            bundle = chosen.ready.popleft()
            chosen.active_children += bundle.child_count
            chosen.released_children += bundle.child_count
            runnable += bundle.child_count
            admitted.append(bundle)
        return admitted

    def _retire_if_done(self, state: TreeState) -> None:
        if (
            state.candidate_phase_closed
            and not state.ready
            and state.active_children == 0
        ):
            self.trees.pop(state.tree_id, None)
