from __future__ import annotations

import unittest

from inference_scaling.arllm.backends.cis_tree_frontier_policy import (
    TreeFrontierPolicy,
)


class TreeFrontierPolicyTest(unittest.TestCase):
    def test_terminal_tree_retires_without_rollout_credit(self) -> None:
        policy = TreeFrontierPolicy(max_num_seqs=16)
        policy.register_tree("j:step:0", 2)
        policy.parent_completed("j:step:0", "p0", child_count=3, terminal=True)
        policy.parent_completed("j:step:0", "p1", child_count=3, terminal=True)
        self.assertEqual(policy.trees, {})
        self.assertEqual(
            policy.choose_bundles(running=0, waiting=0, kv_usage=0.1), []
        )

    def test_bounded_sibling_release_and_completion(self) -> None:
        policy = TreeFrontierPolicy(
            max_num_seqs=8, target_fraction=0.75, max_bundles_per_tick=1
        )
        policy.register_tree("j:step:0", 3)
        for index in range(3):
            policy.parent_completed(
                "j:step:0", f"p{index}", child_count=3, terminal=False
            )
        first = policy.choose_bundles(running=0, waiting=0, kv_usage=0.1)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].child_count, 3)
        self.assertEqual(policy.trees["j:step:0"].active_children, 3)
        second = policy.choose_bundles(running=3, waiting=0, kv_usage=0.1)
        self.assertEqual(len(second), 1)
        self.assertEqual(
            policy.choose_bundles(running=6, waiting=0, kv_usage=0.1), []
        )
        for _ in range(6):
            policy.child_finished("j:step:0")
        third = policy.choose_bundles(running=0, waiting=0, kv_usage=0.1)
        self.assertEqual(len(third), 1)
        for _ in range(3):
            policy.child_finished("j:step:0")
        self.assertEqual(policy.trees, {})

    def test_high_kv_still_allows_closed_tree_to_progress(self) -> None:
        policy = TreeFrontierPolicy(max_num_seqs=32, kv_high_watermark=0.9)
        policy.register_tree("j:step:0", 2)
        policy.parent_completed("j:step:0", "p0", child_count=2, terminal=False)
        self.assertEqual(
            policy.choose_bundles(running=32, waiting=20, kv_usage=0.95), []
        )
        policy.parent_completed("j:step:0", "p1", child_count=2, terminal=False)
        self.assertEqual(
            len(policy.choose_bundles(running=0, waiting=0, kv_usage=0.95)),
            1,
        )
        self.assertEqual(
            policy.choose_bundles(running=32, waiting=20, kv_usage=0.95), []
        )

    def test_open_candidate_phase_only_refills_idle_engine(self) -> None:
        policy = TreeFrontierPolicy(max_num_seqs=16, target_fraction=0.75)
        policy.register_tree("j:step:0", 2)
        policy.parent_completed("j:step:0", "p0", child_count=3, terminal=False)
        self.assertEqual(
            policy.choose_bundles(running=16, waiting=4, kv_usage=0.1), []
        )
        self.assertEqual(
            len(policy.choose_bundles(running=2, waiting=0, kv_usage=0.1)),
            1,
        )

    def test_critical_tree_precedes_other_ready_tree(self) -> None:
        policy = TreeFrontierPolicy(max_num_seqs=64)
        policy.register_tree("older", 3)
        for index in range(3):
            policy.parent_completed(
                "older", f"old{index}", child_count=3, terminal=False
            )
        policy.register_tree("critical", 1)
        policy.parent_completed(
            "critical", "new0", child_count=3, terminal=False
        )
        admitted = policy.choose_bundles(
            running=0, waiting=0, kv_usage=0.1
        )
        self.assertEqual(admitted[0].tree_id, "critical")

    def test_saturated_engine_does_not_drain_ready_frontier(self) -> None:
        policy = TreeFrontierPolicy(max_num_seqs=256, target_fraction=0.80)
        policy.register_tree("j", 2)
        policy.parent_completed("j", "p0", child_count=3, terminal=False)
        policy.parent_completed("j", "p1", child_count=3, terminal=False)
        for _ in range(100):
            self.assertEqual(
                policy.choose_bundles(running=256, waiting=20, kv_usage=0.5),
                [],
            )
        self.assertEqual(len(policy.trees["j"].ready), 2)
        admitted = policy.choose_bundles(
            running=196, waiting=0, kv_usage=0.5
        )
        self.assertEqual(len(admitted), 2)

    def test_sibling_bundle_refills_near_target_without_queue_flood(self) -> None:
        policy = TreeFrontierPolicy(max_num_seqs=256, target_fraction=0.80)
        policy.register_tree("j", 2)
        policy.parent_completed("j", "p0", child_count=3, terminal=False)
        policy.parent_completed("j", "p1", child_count=3, terminal=False)
        admitted = policy.choose_bundles(
            running=204, waiting=0, kv_usage=0.63
        )
        self.assertEqual([bundle.parent_handle for bundle in admitted], ["p0"])
        self.assertEqual(
            policy.choose_bundles(running=207, waiting=0, kv_usage=0.63),
            [],
        )
        self.assertEqual(len(policy.trees["j"].ready), 1)

    def test_rejects_extra_candidate_and_uncredited_child(self) -> None:
        policy = TreeFrontierPolicy(max_num_seqs=16)
        policy.register_tree("j", 2)
        policy.parent_completed("j", "p0", child_count=2, terminal=True)
        policy.parent_completed("j", "p1", child_count=2, terminal=False)
        with self.assertRaises(RuntimeError):
            policy.parent_completed("j", "p2", child_count=2, terminal=False)
        with self.assertRaises(RuntimeError):
            policy.child_finished("j")

    def test_cancel_parked_child_shrinks_ready_bundle(self) -> None:
        policy = TreeFrontierPolicy(max_num_seqs=16)
        policy.register_tree("j", 2)
        policy.parent_completed("j", "p0", child_count=3, terminal=False)
        policy.parked_child_cancelled("j", "p0")
        self.assertEqual(policy.trees["j"].ready[0].child_count, 2)
        policy.parked_child_cancelled("j", "p0")
        policy.parked_child_cancelled("j", "p0")
        self.assertEqual(len(policy.trees["j"].ready), 0)
        policy.parent_completed("j", "p1", child_count=0, terminal=True)
        self.assertEqual(policy.trees, {})


if __name__ == "__main__":
    unittest.main()
