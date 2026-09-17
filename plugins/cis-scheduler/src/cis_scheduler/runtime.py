"""Ready-to-use CIS scheduling facade for vLLM-compatible runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType

from .adapter import VLLMRequestPolicy
from .admission import TokenBudgetAdmissionController, derive_pressure_step_limit
from .capacity import GeometrySource, RuntimeKVCapacityProvider
from .metadata import CISNode
from .priority import CISPriorityPolicy, PriorityMode


def _runtime_geometry(runtime: object) -> Mapping[str, object] | None:
    value = getattr(runtime, "kv_cache_geometry", None)
    value = value() if callable(value) else value
    if isinstance(value, Mapping):
        return value
    config = getattr(runtime, "vllm_config", None)
    cache = getattr(config, "cache_config", None)
    if cache is None:
        return None
    token_capacity = getattr(cache, "kv_cache_size_tokens", None)
    block_size = getattr(cache, "effective_attention_block_size", None)
    if type(block_size) is not int or block_size <= 0:
        block_size = getattr(cache, "block_size", None)
    num_blocks = getattr(cache, "num_gpu_blocks", None)
    if type(token_capacity) is int and token_capacity > 0:
        return {"token_capacity": token_capacity, "block_size": block_size}
    if (
        type(num_blocks) is int
        and num_blocks > 0
        and type(block_size) is int
        and block_size > 0
    ):
        return {"num_gpu_blocks": num_blocks, "block_size": block_size}
    return None


class CISStepLease:
    """One admitted CIS step with automatic release on scope exit."""

    def __init__(
        self,
        plugin: CISSchedulerPlugin,
        job_id: str,
        step_index: int,
        estimated_tokens: int,
    ) -> None:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        if type(step_index) is not int or step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        self.plugin = plugin
        self.job_id = job_id
        self.step_index = step_index
        self.estimated_tokens = estimated_tokens
        self._claim_id = self._make_claim_id(step_index)
        self._active = False

    def _make_claim_id(self, step_index: int) -> str:
        return f"{self.job_id}:step:{step_index}"

    def __enter__(self) -> CISStepLease:
        if self._active:
            raise RuntimeError("CIS step lease is already active")
        self.plugin.admission.acquire(self._claim_id, self.estimated_tokens)
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("CIS step lease is not active")

    def prepare_request(
        self,
        node_type: str,
        candidate_index: int,
        *,
        rollout_index: int | None = None,
        step_rollout_count: int | None = None,
        candidate_max_tokens: int | None = None,
        rollout_max_tokens: int | None = None,
        extra_args: Mapping[str, object] | None = None,
    ) -> tuple[int, dict[str, object]]:
        """Return the runtime priority and metadata for one child request."""

        self._require_active()
        node = CISNode(
            job_id=self.job_id,
            step_index=self.step_index,
            node_type=node_type,
            candidate_index=candidate_index,
            candidate_count=self.plugin.candidate_count,
            rollout_index=rollout_index,
            expected_rollouts=self.plugin.rollout_count,
            step_rollout_count=step_rollout_count,
            candidate_max_tokens=candidate_max_tokens,
            rollout_max_tokens=rollout_max_tokens,
        )
        args = self.plugin.request_policy.attach_metadata(node, extra_args)
        priority = (
            self.plugin.request_policy.priority.priority_for(node)
            if self.plugin.admission.priority_enabled(self._claim_id)
            else 0
        )
        return priority, args

    def resize(self, estimated_tokens: int) -> None:
        self._require_active()
        self.plugin.admission.resize(self._claim_id, estimated_tokens)
        self.estimated_tokens = estimated_tokens

    def transition(self, next_step_index: int, estimated_tokens: int) -> bool:
        """Move the reservation to the next step without release/reacquire."""

        self._require_active()
        if type(next_step_index) is not int or next_step_index < 0:
            raise ValueError("next_step_index must be a non-negative integer")
        next_claim_id = self._make_claim_id(next_step_index)
        transitioned = self.plugin.admission.transition(
            self._claim_id, next_claim_id, estimated_tokens
        )
        if transitioned:
            self._claim_id = next_claim_id
            self.step_index = next_step_index
            self.estimated_tokens = estimated_tokens
        return transitioned

    def close(self) -> None:
        if not self._active:
            return
        self.plugin.admission.release(self._claim_id)
        self._active = False


class CISSchedulerPlugin:
    """Bind CIS admission and request metadata to an existing runtime."""

    def __init__(
        self,
        runtime: object,
        *,
        candidate_count: int,
        rollout_count: int,
        max_num_seqs: int,
        max_active_steps: int | None = None,
        capacity_fraction: float = 0.8,
        priority_mode: PriorityMode = "job_fifo",
        sequence_multiplier: float = 1.5,
        pressure_activate_fraction: float = 1.0,
        pressure_deactivate_fraction: float = 0.7,
        geometry: GeometrySource | None = None,
    ) -> None:
        if (
            type(candidate_count) is not int
            or type(rollout_count) is not int
            or candidate_count <= 0
            or rollout_count <= 0
        ):
            raise ValueError("candidate_count and rollout_count must be positive")
        self.runtime = runtime
        self.candidate_count = candidate_count
        self.rollout_count = rollout_count
        self.request_policy = VLLMRequestPolicy(CISPriorityPolicy(priority_mode))
        capacity = RuntimeKVCapacityProvider(
            geometry if geometry is not None else lambda: _runtime_geometry(runtime),
            fraction=capacity_fraction,
        )
        step_limit = derive_pressure_step_limit(
            max_num_seqs,
            self.candidate_count,
            self.rollout_count,
            sequence_multiplier=sequence_multiplier,
        )
        self.admission = TokenBudgetAdmissionController(
            capacity,
            max_active_steps=(
                max_num_seqs if max_active_steps is None else max_active_steps
            ),
            pressure_gate=True,
            pressure_activate_fraction=pressure_activate_fraction,
            pressure_deactivate_fraction=pressure_deactivate_fraction,
            pressure_step_limit=step_limit,
        )
        bind = getattr(runtime, "bind_cis_admission_controller", None)
        if callable(bind):
            bind(self.admission)

    def step(
        self, job_id: str, step_index: int, estimated_tokens: int
    ) -> CISStepLease:
        return CISStepLease(self, job_id, step_index, estimated_tokens)

    def register_job(self, job_id: str, order: int) -> None:
        self.request_policy.priority.register_job_order(job_id, order)
        register = getattr(self.runtime, "register_cis_job", None)
        if callable(register):
            register(job_id, order)

    def finish_job(self, job_id: str) -> None:
        self.request_policy.priority.finish_job(job_id)
        finish = getattr(self.runtime, "finish_cis_job", None)
        if callable(finish):
            finish(job_id)
