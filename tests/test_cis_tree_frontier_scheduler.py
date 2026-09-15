"""Contract tests for the pinned vLLM 0.18 fork-waiter scheduler adapter.

Run inside the vLLM-Ascend image after applying the CIS fork patches. These
tests do not construct model workers or use NPUs.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus
    from inference_scaling.arllm.backends.cis_tree_frontier_policy import (
        TreeFrontierPolicy,
    )
    from inference_scaling.arllm.backends.cis_tree_frontier_scheduler import (
        CISTreeFrontierScheduler,
    )
except (ImportError, ModuleNotFoundError):
    Scheduler = None


@unittest.skipIf(Scheduler is None, "requires the pinned vLLM-Ascend runtime")
class TreeFrontierSchedulerContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = CISTreeFrontierScheduler.__new__(CISTreeFrontierScheduler)
        self.scheduler._cis_frontier = TreeFrontierPolicy(
            max_num_seqs=8, target_fraction=0.75,
            max_bundles_per_tick=1,
        )
        self.scheduler._cis_frontier.register_tree("j:step:0", 2)
        self.scheduler._cis_frontier_trace = None
        self.scheduler._cis_last_stall_recorded = 0.0
        self.scheduler._cis_cancelled_before_parent = {}
        self.scheduler._cis_fork_completed_parents = {}
        self.scheduler._cis_fork_parent_remaining_children = {}
        self.scheduler._cis_fork_group_remaining = {"j:step:0": 2}
        self.scheduler._cis_fork_group_size = {"j:step:0": 2}
        self.scheduler._cis_fork_group_release_remaining = {"j:step:0": 0}
        self.scheduler._cis_fork_group_adaptive = {"j:step:0": False}
        self.scheduler._cis_fork_group_runnable_fraction = {"j:step:0": 0.5}
        self.scheduler._cis_fork_parent_by_child = {}
        self.scheduler._cis_fork_released_parents = set()
        self.scheduler.running = [object()] * 8
        self.scheduler.waiting = [object()] * 4
        self.scheduler.skipped_waiting = []
        self.scheduler.kv_cache_manager = SimpleNamespace(usage=0.1)
        self.released: list[str] = []
        self.scheduler._release_cis_fork_parent = (
            lambda _parent, handle: self.released.append(handle)
        )

    @staticmethod
    def parent(handle: str, *, terminal: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            request_id=handle,
            status=(
                RequestStatus.FINISHED_ABORTED
                if terminal
                else RequestStatus.FINISHED_LENGTH_CAPPED
            ),
            sampling_params=SimpleNamespace(
                extra_args={
                    "cis_fork_group_id": "j:step:0",
                    "cis_fork_expected_children": 3,
                }
            ),
        )

    def test_completed_parents_are_held_then_bounded_released(self) -> None:
        for index in range(2):
            self.scheduler._complete_cis_fork_parent(
                self.parent(f"p{index}"), f"p{index}"
            )
        self.assertEqual(self.released, [])
        self.assertEqual(
            len(self.scheduler._cis_frontier.trees["j:step:0"].ready), 2
        )
        self.scheduler.running = []
        self.scheduler.waiting = []
        with patch.object(Scheduler, "schedule", return_value="native"):
            self.assertEqual(self.scheduler.schedule(), "native")
            self.assertEqual(self.released, ["p0"])
            self.assertEqual(self.scheduler.schedule(), "native")
        self.assertEqual(self.released, ["p0", "p1"])
        self.assertEqual(self.scheduler._cis_fork_group_remaining, {})

    def test_terminal_parent_cancels_children_without_ready_credit(self) -> None:
        self.scheduler._complete_cis_fork_parent(
            self.parent("terminal", terminal=True), "terminal"
        )
        self.assertEqual(self.released, ["terminal"])
        self.assertEqual(
            len(self.scheduler._cis_frontier.trees["j:step:0"].ready), 0
        )

    def test_parked_child_abort_does_not_consume_active_credit(self) -> None:
        self.scheduler._cis_fork_parent_by_child["child"] = "parent"
        child = SimpleNamespace(
            request_id="child",
            sampling_params=SimpleNamespace(
                extra_args={
                    "cis_request": {
                        "job_id": "j",
                        "step_index": 0,
                        "node_type": "rollout",
                    }
                }
            ),
        )
        with patch.object(Scheduler, "_free_request", return_value=None):
            self.assertIsNone(self.scheduler._free_request(child))
        self.assertEqual(
            self.scheduler._cis_frontier.trees["j:step:0"].active_children,
            0,
        )

    def test_mismatched_tree_metadata_is_rejected_before_enqueue(self) -> None:
        parent = SimpleNamespace(
            request_id="bad",
            sampling_params=SimpleNamespace(
                extra_args={
                    "cis_fork_group_id": "other:step:0",
                    "cis_fork_group_size": 2,
                    "cis_request": {
                        "schema_version": 1,
                        "job_id": "j",
                        "step_index": 0,
                        "node_type": "candidate",
                    },
                }
            ),
        )
        with patch.object(Scheduler, "add_request") as native_add:
            with self.assertRaisesRegex(ValueError, "does not match"):
                self.scheduler.add_request(parent)
            native_add.assert_not_called()

    def test_cancel_after_parent_completion_releases_child_and_hint_credit(self) -> None:
        hint = SimpleNamespace(remaining_children=3)
        released_hints: list[str] = []
        self.scheduler.kv_cache_manager = SimpleNamespace(
            usage=0.1,
            _cis_fork_hints={"p0": hint},
            _release_cis_fork_hint=lambda handle: released_hints.append(handle),
        )
        self.scheduler._complete_cis_fork_parent(self.parent("p0"), "p0")
        self.scheduler._cis_fork_parent_by_child["child"] = "p0"
        child = SimpleNamespace(
            request_id="child",
            sampling_params=SimpleNamespace(
                extra_args={
                    "cis_request": {
                        "job_id": "j",
                        "step_index": 0,
                        "node_type": "rollout",
                    }
                }
            ),
        )
        with patch.object(Scheduler, "_free_request", return_value=None):
            self.scheduler._free_request(child)
        self.assertEqual(hint.remaining_children, 2)
        self.assertEqual(
            self.scheduler._cis_fork_parent_remaining_children["p0"], 2
        )
        self.assertEqual(
            self.scheduler._cis_frontier.trees["j:step:0"].ready[0].child_count,
            2,
        )
        self.assertEqual(released_hints, [])

    def test_cancel_before_parent_completion_is_applied_after_capture(self) -> None:
        hint = SimpleNamespace(remaining_children=3)
        self.scheduler.kv_cache_manager = SimpleNamespace(
            usage=0.1,
            _cis_fork_hints={"p0": hint},
            _release_cis_fork_hint=lambda _handle: None,
        )
        self.scheduler._cis_fork_parent_by_child["child"] = "p0"
        child = SimpleNamespace(
            request_id="child",
            sampling_params=SimpleNamespace(extra_args={}),
        )
        with patch.object(Scheduler, "_free_request", return_value=None):
            self.scheduler._free_request(child)
        self.assertEqual(self.scheduler._cis_cancelled_before_parent["p0"], 1)
        self.scheduler._complete_cis_fork_parent(self.parent("p0"), "p0")
        self.assertEqual(hint.remaining_children, 2)
        self.assertEqual(
            self.scheduler._cis_fork_parent_remaining_children["p0"], 2
        )
        self.assertEqual(
            self.scheduler._cis_frontier.trees["j:step:0"].ready[0].child_count,
            2,
        )
        self.assertEqual(self.scheduler._cis_cancelled_before_parent, {})


if __name__ == "__main__":
    unittest.main()
