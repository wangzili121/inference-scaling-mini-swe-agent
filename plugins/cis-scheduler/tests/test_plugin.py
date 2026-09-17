from concurrent.futures import ThreadPoolExecutor, TimeoutError
from types import SimpleNamespace
import unittest

from cis_scheduler import (
    CISNode,
    CISPriorityPolicy,
    CISSchedulerPlugin,
    RuntimeKVCapacityProvider,
    TokenBudgetAdmissionController,
    VLLMRequestPolicy,
    derive_pressure_step_limit,
)


def candidate(job="job-a", step=0, index=0):
    return CISNode(
        job_id=job,
        step_index=step,
        node_type="candidate",
        candidate_index=index,
        candidate_count=2,
        expected_rollouts=2,
        candidate_max_tokens=128,
        rollout_max_tokens=512,
    )


class MetadataTests(unittest.TestCase):
    def test_current_runtime_mapping_round_trip(self):
        node = CISNode(
            job_id="job-a",
            step_index=0,
            node_type="rollout",
            candidate_index=1,
            candidate_count=2,
            rollout_index=0,
            expected_rollouts=2,
            step_rollout_count=4,
            candidate_max_tokens=128,
            rollout_max_tokens=512,
        )
        self.assertEqual(CISNode.from_mapping(node.to_mapping()), node)
        self.assertEqual(node.to_mapping()["step_rollout_count"], 4)

    def test_rejects_invalid_schema_and_nodes(self):
        mapping = candidate().to_mapping()
        mapping["schema_version"] = True
        with self.assertRaises(ValueError):
            CISNode.from_mapping(mapping)
        mapping = candidate().to_mapping()
        mapping["step_rollout_count"] = 2
        with self.assertRaises(ValueError):
            CISNode.from_mapping(mapping)
        mapping = candidate().to_mapping()
        mapping["candidate_max_tokens"] = 0
        with self.assertRaises(ValueError):
            CISNode.from_mapping(mapping)


class PriorityTests(unittest.TestCase):
    def test_job_fifo_groups_all_steps_and_branches(self):
        policy = CISPriorityPolicy("job_fifo")
        first = policy.priority_for(candidate())
        self.assertEqual(policy.priority_for(candidate(step=1)), first)
        self.assertEqual(policy.priority_for(candidate(index=1)), first)
        self.assertGreater(policy.priority_for(candidate(job="job-b")), first)

    def test_step_fifo_separates_steps(self):
        policy = CISPriorityPolicy("step_fifo")
        first = policy.priority_for(candidate())
        second = policy.priority_for(candidate(step=1))
        self.assertGreater(second, first)
        self.assertEqual(policy.priority_for(candidate(index=1)), first)
        policy.finish_step("job-a", 0)
        self.assertGreater(policy.priority_for(candidate()), second)

    def test_concurrent_group_assignment_is_stable(self):
        policy = CISPriorityPolicy()
        with ThreadPoolExecutor(max_workers=16) as pool:
            values = list(pool.map(policy.priority_for, [candidate()] * 100))
        self.assertEqual(len(set(values)), 1)

    def test_registered_job_order_does_not_depend_on_arrival(self):
        policy = CISPriorityPolicy()
        policy.register_job_order("job-b", 1)
        policy.register_job_order("job-a", 0)
        self.assertEqual(policy.priority_for(candidate(job="job-a")), -1_000_000)
        self.assertEqual(policy.priority_for(candidate(job="job-b")), -999_999)

    def test_registered_job_order_cannot_collide_with_dynamic_priority(self):
        policy = CISPriorityPolicy()
        with self.assertRaises(ValueError):
            policy.register_job_order("job-a", 1_000_000)


class AdapterTests(unittest.TestCase):
    def test_preserves_extra_args_and_attaches_metadata(self):
        adapter = VLLMRequestPolicy(CISPriorityPolicy())
        priority, args = adapter.prepare(candidate(), {"existing": 42})
        self.assertEqual(priority, 0)
        self.assertEqual(args["existing"], 42)
        self.assertEqual(args["cis_request"]["job_id"], "job-a")

    def test_mapping_adapter_validates_metadata(self):
        adapter = VLLMRequestPolicy(CISPriorityPolicy())
        mapping = candidate().to_mapping()
        mapping["schema_version"] = 2
        with self.assertRaises(ValueError):
            adapter.attach_mapping(mapping)

    def test_non_cis_request_keeps_runtime_behavior(self):
        adapter = VLLMRequestPolicy(CISPriorityPolicy())
        self.assertEqual(adapter.prepare(None, {"existing": 42}), (0, {"existing": 42}))


class _Runtime:
    kv_cache_geometry = {
        "token_capacity": 1000,
        "block_size": 100,
    }

    def bind_cis_admission_controller(self, controller):
        self.controller = controller

    def register_cis_job(self, job_id, order):
        self.registered = job_id, order

    def finish_cis_job(self, job_id):
        self.finished = job_id


class RuntimeFacadeTests(unittest.TestCase):
    def plugin(self):
        return CISSchedulerPlugin(
            _Runtime(),
            candidate_count=2,
            rollout_count=2,
            max_num_seqs=4,
            max_active_steps=4,
        )

    def test_context_manages_admission_and_request_metadata(self):
        runtime = _Runtime()
        plugin = CISSchedulerPlugin(
            runtime,
            candidate_count=2,
            rollout_count=2,
            max_num_seqs=4,
            max_active_steps=4,
        )
        self.assertIs(runtime.controller, plugin.admission)
        with plugin.step("job-a", 0, 400) as step:
            priority, args = step.prepare_request("candidate", 1)
            self.assertEqual(priority, 0)
            self.assertEqual(args["cis_request"]["job_id"], "job-a")
            self.assertEqual(args["cis_request"]["candidate_index"], 1)
            self.assertEqual(plugin.admission.snapshot().active_steps, 1)
        self.assertEqual(plugin.admission.snapshot().active_steps, 0)
        plugin.register_job("job-b", 1)
        plugin.finish_job("job-b")
        self.assertEqual(runtime.registered, ("job-b", 1))
        self.assertEqual(runtime.finished, "job-b")

    def test_context_releases_after_exception(self):
        plugin = self.plugin()
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with plugin.step("job-a", 0, 400):
                raise RuntimeError("failed")
        self.assertEqual(plugin.admission.snapshot().active_steps, 0)

    def test_transition_updates_request_step(self):
        plugin = self.plugin()
        with plugin.step("job-a", 0, 400) as step:
            self.assertTrue(step.transition(1, 300))
            _, args = step.prepare_request("rollout", 0, rollout_index=1)
            self.assertEqual(args["cis_request"]["step_index"], 1)
        self.assertEqual(plugin.admission.snapshot().active_steps, 0)

    def test_reads_vllm_cache_config_without_custom_backend_adapter(self):
        runtime = SimpleNamespace(
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(
                    kv_cache_size_tokens=2000,
                    effective_attention_block_size=100,
                )
            )
        )
        plugin = CISSchedulerPlugin(
            runtime,
            candidate_count=2,
            rollout_count=2,
            max_num_seqs=4,
        )
        self.assertEqual(plugin.admission.token_budget, 1600)


class CapacityTests(unittest.TestCase):
    def test_runtime_geometry_derives_budget(self):
        provider = RuntimeKVCapacityProvider(
            {"num_gpu_blocks": 3976, "block_size": 128}, fraction=0.8
        )
        snapshot = provider.snapshot()
        self.assertEqual(snapshot.token_capacity, 508928)
        self.assertEqual(snapshot.block_size, 128)
        self.assertEqual(snapshot.budget_tokens, 407142)

    def test_runtime_geometry_is_read_on_each_snapshot(self):
        geometry = {"token_capacity": 1000, "block_size": 100}
        provider = RuntimeKVCapacityProvider(lambda: geometry, fraction=0.9)
        self.assertEqual(provider.snapshot().budget_tokens, 900)
        geometry["token_capacity"] = 2000
        self.assertEqual(provider.snapshot().budget_tokens, 1800)

    def test_hybrid_kv_layout_uses_bottleneck_pool(self):
        provider = RuntimeKVCapacityProvider(
            {
                "kv_pools": [
                    {
                        "name": "full-attention",
                        "num_blocks": 500,
                        "block_size": 64,
                    },
                    {
                        "name": "latent-attention",
                        "token_capacity": 24000,
                        "block_size": 128,
                    },
                ]
            },
            fraction=0.8,
        )
        snapshot = provider.snapshot()
        self.assertEqual(snapshot.token_capacity, 24000)
        self.assertEqual(snapshot.block_size, 128)
        self.assertEqual(snapshot.budget_tokens, 19200)
        self.assertEqual(
            [pool.name for pool in snapshot.pools],
            ["full-attention", "latent-attention"],
        )

    def test_hybrid_kv_layout_uses_common_reservation_granularity(self):
        provider = RuntimeKVCapacityProvider(
            {
                "kv_pools": [
                    {"token_capacity": 1000, "block_size": 48},
                    {"token_capacity": 2000, "block_size": 64},
                ]
            },
            fraction=1.0,
        )
        self.assertEqual(provider.snapshot().block_size, 192)

    def test_rejects_invalid_geometry(self):
        with self.assertRaises(ValueError):
            RuntimeKVCapacityProvider({}, fraction=0.8).snapshot()
        with self.assertRaises(ValueError):
            RuntimeKVCapacityProvider(
                {"token_capacity": 1000, "block_size": 100}, fraction=0
            )
        with self.assertRaises(ValueError):
            RuntimeKVCapacityProvider({"kv_pools": []}, fraction=0.8).snapshot()


class AdmissionTests(unittest.TestCase):
    def test_pressure_step_limit_scales_with_engine_and_fanout(self):
        self.assertEqual(derive_pressure_step_limit(256, 8, 3), 16)
        self.assertEqual(derive_pressure_step_limit(512, 8, 3), 32)
        self.assertEqual(derive_pressure_step_limit(256, 15, 3), 8)
        with self.assertRaises(ValueError):
            derive_pressure_step_limit(256, 8, 0)

    def test_pressure_gate_bypasses_priority_below_budget(self):
        provider = RuntimeKVCapacityProvider(
            {"token_capacity": 1000, "block_size": 100}, fraction=0.8
        )
        controller = TokenBudgetAdmissionController(
            provider, max_active_steps=4, pressure_gate=True
        )
        controller.acquire("job-a:step:0", 300)
        controller.acquire("job-b:step:0", 300)
        self.assertFalse(controller.priority_enabled("job-a:step:0"))
        self.assertFalse(controller.priority_enabled("job-b:step:0"))
        snapshot = controller.snapshot()
        self.assertFalse(snapshot.pressure_active)
        self.assertEqual(snapshot.gate_bypasses, 2)

    def test_pressure_gate_enables_existing_steps_at_budget_boundary(self):
        provider = RuntimeKVCapacityProvider(
            {"token_capacity": 1000, "block_size": 100}, fraction=0.8
        )
        controller = TokenBudgetAdmissionController(
            provider,
            max_active_steps=4,
            pressure_gate=True,
            poll_seconds=0.01,
        )
        controller.acquire("job-a:step:0", 400)
        controller.acquire("job-b:step:0", 400)
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(controller.acquire, "job-c:step:0", 100)
            with self.assertRaises(TimeoutError):
                waiting.result(timeout=0.05)
            self.assertTrue(controller.priority_enabled("job-a:step:0"))
            self.assertTrue(controller.priority_enabled("job-b:step:0"))
            self.assertTrue(controller.snapshot().pressure_active)
            controller.release("job-a:step:0")
            waiting.result(timeout=1.0)
        self.assertTrue(controller.priority_enabled("job-c:step:0"))
        self.assertEqual(controller.snapshot().gate_activations, 1)

    def test_pressure_gate_uses_runtime_sequence_limit(self):
        provider = RuntimeKVCapacityProvider(
            {"token_capacity": 10000, "block_size": 100}, fraction=0.8
        )
        controller = TokenBudgetAdmissionController(
            provider,
            max_active_steps=8,
            pressure_gate=True,
            pressure_step_limit=2,
            poll_seconds=0.01,
        )
        controller.acquire("a", 100)
        controller.acquire("b", 100)
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(controller.acquire, "c", 100)
            with self.assertRaises(TimeoutError):
                waiting.result(timeout=0.05)
            self.assertTrue(controller.priority_enabled("a"))
            self.assertTrue(controller.priority_enabled("b"))
            self.assertEqual(controller.snapshot().pressure_step_limit, 2)
            controller.release("a")
            waiting.result(timeout=1.0)

    def test_pressure_gate_deactivates_with_hysteresis(self):
        provider = RuntimeKVCapacityProvider(
            {"token_capacity": 1000, "block_size": 100}, fraction=0.8
        )
        controller = TokenBudgetAdmissionController(
            provider,
            max_active_steps=4,
            pressure_gate=True,
            pressure_deactivate_fraction=0.5,
            poll_seconds=0.01,
        )
        controller.acquire("a", 400)
        controller.acquire("b", 400)
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(controller.acquire, "c", 100)
            with self.assertRaises(TimeoutError):
                waiting.result(timeout=0.05)
            controller.release("a")
            waiting.result(timeout=1.0)
        controller.release("b")
        self.assertFalse(controller.snapshot().pressure_active)
        controller.acquire("d", 100)
        self.assertFalse(controller.priority_enabled("d"))

    def test_transition_preserves_latched_priority(self):
        provider = RuntimeKVCapacityProvider(
            {"token_capacity": 1000, "block_size": 100}, fraction=0.8
        )
        controller = TokenBudgetAdmissionController(
            provider, max_active_steps=2, pressure_gate=True
        )
        controller.acquire("a", 900)
        self.assertTrue(controller.priority_enabled("a"))
        self.assertTrue(controller.transition("a", "next", 300))
        self.assertTrue(controller.priority_enabled("next"))

    def test_budget_blocks_and_release_unblocks_fifo_waiter(self):
        provider = RuntimeKVCapacityProvider(
            {"token_capacity": 1000, "block_size": 100}, fraction=0.8
        )
        controller = TokenBudgetAdmissionController(provider, max_active_steps=2)
        controller.acquire("a", 350)
        controller.acquire("b", 350)
        self.assertEqual(controller.snapshot().active_tokens, 800)
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(controller.acquire, "c", 100)
            with self.assertRaises(TimeoutError):
                waiting.result(timeout=0.05)
            controller.release("a")
            waiting.result(timeout=1.0)
        snapshot = controller.snapshot()
        self.assertEqual(snapshot.active_steps, 2)
        self.assertEqual(snapshot.peak_active_steps, 2)
        self.assertEqual(snapshot.peak_active_tokens, 800)
        self.assertEqual(snapshot.admissions, 3)
        self.assertGreater(snapshot.wait_seconds, 0)

    def test_runtime_capacity_growth_unblocks_waiter(self):
        geometry = {"token_capacity": 500, "block_size": 100}
        provider = RuntimeKVCapacityProvider(lambda: geometry, fraction=1.0)
        controller = TokenBudgetAdmissionController(
            provider, max_active_steps=2, poll_seconds=0.01
        )
        controller.acquire("a", 500)
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(controller.acquire, "b", 500)
            with self.assertRaises(TimeoutError):
                waiting.result(timeout=0.05)
            geometry["token_capacity"] = 1000
            waiting.result(timeout=1.0)

    def test_resize_transition_and_oversized_progress(self):
        provider = RuntimeKVCapacityProvider(
            {"token_capacity": 500, "block_size": 100}, fraction=1.0
        )
        controller = TokenBudgetAdmissionController(provider, max_active_steps=2)
        controller.acquire("oversized", 600)
        controller.resize("oversized", 300)
        self.assertTrue(controller.transition("oversized", "next", 400))
        self.assertEqual(controller.transitions, 1)
        self.assertEqual(controller.transition_failures, 0)
        self.assertEqual(controller.snapshot().active_tokens, 400)
        controller.release("next")
        self.assertEqual(controller.snapshot().active_steps, 0)

    def test_rejects_unnamed_claims(self):
        provider = RuntimeKVCapacityProvider(
            {"token_capacity": 500, "block_size": 100}, fraction=1.0
        )
        controller = TokenBudgetAdmissionController(provider, max_active_steps=2)
        with self.assertRaises(ValueError):
            controller.acquire("", 100)
        controller.acquire("step", 100)
        with self.assertRaises(ValueError):
            controller.transition("step", "", 100)


if __name__ == "__main__":
    unittest.main()
