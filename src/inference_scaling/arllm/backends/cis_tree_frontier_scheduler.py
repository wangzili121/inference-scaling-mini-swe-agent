"""vLLM 0.18 adapter for a bounded CIS fork-tree ready frontier.

Requires the existing, separately validated fork-waiter runtime patch. The
adapter changes only when completed parent bundles enter vLLM waiting; the
native scheduler still chooses individual branches for each forward batch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from .cis_tree_frontier_policy import TreeFrontierPolicy


class CISTreeFrontierScheduler(Scheduler):
    """Release physical-KV-forked rollout siblings under a tree-level budget."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not getattr(Scheduler, "cis_fork_waiter_supported", False):
            raise RuntimeError(
                "CISTreeFrontierScheduler requires the vLLM 0.18 CIS fork-waiter patch"
            )
        super().__init__(*args, **kwargs)
        if self.policy != SchedulingPolicy.PRIORITY:
            raise ValueError("CIS tree frontier requires scheduling_policy='priority'")
        self._cis_frontier = TreeFrontierPolicy(
            max_num_seqs=self.max_num_running_reqs,
            target_fraction=float(
                os.environ.get("VLLM_CIS_TREE_TARGET_FRACTION", "1.0")
            ),
            kv_high_watermark=float(
                os.environ.get("VLLM_CIS_TREE_KV_HIGH_WATERMARK", "0.92")
            ),
            max_bundles_per_tick=int(
                os.environ.get("VLLM_CIS_TREE_MAX_BUNDLES_PER_TICK", "8")
            ),
        )
        self._cis_cancelled_before_parent: dict[str, int] = {}
        trace = os.environ.get("VLLM_CIS_TREE_FRONTIER_TRACE")
        self._cis_frontier_trace = Path(trace) if trace else None
        if self._cis_frontier_trace is not None:
            self._cis_frontier_trace.parent.mkdir(parents=True, exist_ok=True)
        self._cis_last_stall_recorded = 0.0

    @staticmethod
    def _extra_args(request: Request) -> dict[str, Any]:
        params = request.sampling_params
        value = None if params is None else params.extra_args
        return value if isinstance(value, dict) else {}

    def _record(self, event: str, **data: Any) -> None:
        if self._cis_frontier_trace is None:
            return
        record = {
            "event": event,
            "time_monotonic": time.monotonic(),
            "time_wall": time.time(),
            "running": len(self.running),
            "waiting": len(self.waiting) + len(self.skipped_waiting),
            "kv_usage": self.kv_cache_manager.usage,
            "trees": len(self._cis_frontier.trees),
            **data,
        }
        with self._cis_frontier_trace.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _cancel_fork_hint_child(self, parent_handle: str) -> None:
        hints = getattr(self.kv_cache_manager, "_cis_fork_hints", None)
        hint = None if hints is None else hints.get(parent_handle)
        if hint is None:
            return
        hint.remaining_children -= 1
        if hint.remaining_children < 0:
            raise RuntimeError("CIS fork hint over-counted cancelled children")
        if hint.remaining_children == 0:
            self.kv_cache_manager._release_cis_fork_hint(parent_handle)

    def add_request(self, request: Request) -> None:
        extra = self._extra_args(request)
        group_id = extra.get("cis_fork_group_id")
        if group_id is not None and not extra.get("cis_fork_wait_for_parent", False):
            cis = extra.get("cis_request")
            if not isinstance(cis, dict) or cis.get("schema_version") != 1:
                raise ValueError("CIS fork parent requires v1 tree metadata")
            expected_group = f"{cis['job_id']}:step:{int(cis['step_index'])}"
            if cis.get("node_type") != "candidate" or str(group_id) != expected_group:
                raise ValueError("CIS fork group does not match candidate metadata")
            group_size = int(extra["cis_fork_group_size"])
            self._cis_frontier.register_tree(str(group_id), group_size)
        super().add_request(request)

    def _complete_cis_fork_parent(
        self, parent: Request, fork_handle: str
    ) -> None:
        extra = self._extra_args(parent)
        group_id = extra.get("cis_fork_group_id")
        if group_id is None:
            super()._complete_cis_fork_parent(parent, fork_handle)
            return
        tree_id = str(group_id)
        expected = int(extra.get("cis_fork_expected_children", 0))
        if expected <= 0 or fork_handle in self._cis_fork_completed_parents:
            raise RuntimeError("CIS fork parent has invalid or duplicate child state")
        remaining = self._cis_fork_group_remaining[tree_id] - 1
        if remaining < 0:
            raise RuntimeError("CIS fork group completed too many parents")
        cancelled = self._cis_cancelled_before_parent.pop(fork_handle, 0)
        if cancelled > expected:
            raise RuntimeError("CIS fork parent has more cancellations than children")
        self._cis_fork_completed_parents[fork_handle] = parent
        self._cis_fork_parent_remaining_children[fork_handle] = expected
        self._cis_fork_group_remaining[tree_id] = remaining
        for _ in range(cancelled):
            self._cancel_fork_hint_child(fork_handle)
        if cancelled:
            self._account_cis_fork_children(fork_handle, cancelled)
        terminal = parent.status != RequestStatus.FINISHED_LENGTH_CAPPED
        state = self._cis_frontier.parent_completed(
            tree_id,
            fork_handle,
            child_count=expected - cancelled,
            terminal=terminal,
        )
        self._record(
            "candidate_completed",
            tree_id=tree_id,
            parent_handle=fork_handle,
            terminal=terminal,
            cancelled_children=cancelled,
            candidate_phase_closed=state.candidate_phase_closed,
            ready_bundles=len(state.ready),
            active_children=state.active_children,
        )
        if terminal and cancelled < expected:
            # Preserve the runtime patch's zero-compute child cancellation.
            self._release_cis_fork_parent(parent, fork_handle)
        if remaining == 0:
            self._cis_fork_group_remaining.pop(tree_id, None)
            self._cis_fork_group_size.pop(tree_id, None)
            self._cis_fork_group_release_remaining.pop(tree_id, None)
            self._cis_fork_group_adaptive.pop(tree_id, None)
            self._cis_fork_group_runnable_fraction.pop(tree_id, None)

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        extra = self._extra_args(request)
        cis = extra.get("cis_request")
        rollout_tree_id = None
        if isinstance(cis, dict) and cis.get("node_type") == "rollout":
            rollout_tree_id = f"{cis['job_id']}:step:{int(cis['step_index'])}"
        parked_parent = self._cis_fork_parent_by_child.get(request.request_id)
        result = super()._free_request(request, delay_free_blocks=delay_free_blocks)
        if parked_parent is not None:
            if parked_parent in self._cis_fork_completed_parents:
                if rollout_tree_id is not None:
                    self._cis_frontier.parked_child_cancelled(
                        rollout_tree_id, parked_parent
                    )
                self._account_cis_fork_children(parked_parent, 1)
                self._cancel_fork_hint_child(parked_parent)
            else:
                self._cis_cancelled_before_parent[parked_parent] = (
                    self._cis_cancelled_before_parent.get(parked_parent, 0) + 1
                )
            self._record(
                "parked_child_cancelled",
                parent_handle=parked_parent,
                request_id=request.request_id,
            )
        elif rollout_tree_id is not None:
            self._cis_frontier.child_finished(rollout_tree_id)
            self._record(
                "rollout_finished",
                tree_id=rollout_tree_id,
                request_id=request.request_id,
            )
        return result

    def schedule(self):
        ready_before = sum(
            len(state.ready) for state in self._cis_frontier.trees.values()
        )
        bundles = self._cis_frontier.choose_bundles(
            running=len(self.running),
            waiting=len(self.waiting) + len(self.skipped_waiting),
            kv_usage=float(self.kv_cache_manager.usage),
        )
        for bundle in bundles:
            parent = self._cis_fork_completed_parents.get(bundle.parent_handle)
            if parent is None:
                raise RuntimeError("CIS tree frontier lost a held fork parent")
            self._release_cis_fork_parent(parent, bundle.parent_handle)
            self._record(
                "rollout_bundle_admitted",
                tree_id=bundle.tree_id,
                parent_handle=bundle.parent_handle,
                children=bundle.child_count,
            )
        if ready_before and not bundles and self._cis_frontier_trace is not None:
            now = time.monotonic()
            if now - self._cis_last_stall_recorded >= 1.0:
                self._cis_last_stall_recorded = now
                self._record(
                    "frontier_wait",
                    ready_bundles=ready_before,
                    ready_children=sum(
                        sum(bundle.child_count for bundle in state.ready)
                        for state in self._cis_frontier.trees.values()
                    ),
                )
        return super().schedule()
