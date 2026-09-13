"""Conditional-IS-aware admission for the vLLM V1 EngineCore scheduler.

This module deliberately subclasses the vLLM 0.18 scheduler instead of copying
its scheduling loop.  The policy controls which complete CIS steps may enter
the ordinary vLLM waiting queue; vLLM remains responsible for token budgets,
continuous batching, KV allocation, and preemption.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any

from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request


_METADATA_KEY = "cis_request"


def _float_env(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return value


@dataclass(slots=True)
class _StepState:
    key: str
    order: int
    candidate_count: int
    rollouts_per_candidate: int
    active: bool = False
    phase: str = "candidate"
    seen_candidates: set[int] = field(default_factory=set)
    seen_rollouts: set[tuple[int, int]] = field(default_factory=set)
    rollout_request_count: int | None = None
    live_request_ids: set[str] = field(default_factory=set)
    deferred: dict[str, Request] = field(default_factory=dict)
    idle_since: float | None = None


class CISTreeScheduler(Scheduler):
    """Experimental EngineCore admission policies for complete CIS steps.

    Adaptive-window mode grows a conservative initial window from observed
    engine pressure. Fractional-work mode lends capacity as rollout branches
    complete. Requests inside admitted steps still use vLLM's native priority
    scheduler and continuous batching. Both modes are opt-in experiments.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.policy != SchedulingPolicy.PRIORITY:
            raise ValueError("CISTreeScheduler requires scheduling_policy='priority'")
        self._cis_steps: dict[str, _StepState] = {}
        self._cis_active_steps: set[str] = set()
        self._cis_next_order = 0
        self._cis_window: int | None = None
        self._cis_initial_window: int | None = None
        self._cis_admission_mode = os.environ.get(
            "VLLM_CIS_ADMISSION_MODE", "adaptive_window"
        )
        if self._cis_admission_mode not in {"adaptive_window", "fractional_work"}:
            raise ValueError(
                "VLLM_CIS_ADMISSION_MODE must be adaptive_window or fractional_work"
            )
        configured_window = os.environ.get("VLLM_CIS_INITIAL_WINDOW")
        self._cis_configured_initial_window = (
            None if configured_window is None else int(configured_window)
        )
        if (
            self._cis_configured_initial_window is not None
            and self._cis_configured_initial_window <= 0
        ):
            raise ValueError("VLLM_CIS_INITIAL_WINDOW must be positive")
        self._cis_target_running_fraction = _float_env(
            "VLLM_CIS_TARGET_RUNNING_FRACTION", 0.50
        )
        self._cis_kv_low_watermark = _float_env(
            "VLLM_CIS_KV_LOW_WATERMARK", 0.80
        )
        self._cis_kv_high_watermark = _float_env(
            "VLLM_CIS_KV_HIGH_WATERMARK", 0.92
        )
        if self._cis_kv_low_watermark >= self._cis_kv_high_watermark:
            raise ValueError("CIS KV low watermark must be below the high watermark")
        self._cis_adjust_interval = float(
            os.environ.get("VLLM_CIS_WINDOW_ADJUST_INTERVAL", "1.0")
        )
        if self._cis_adjust_interval <= 0:
            raise ValueError("VLLM_CIS_WINDOW_ADJUST_INTERVAL must be positive")
        self._cis_candidate_idle_grace = float(
            os.environ.get("VLLM_CIS_CANDIDATE_IDLE_GRACE", "5.0")
        )
        if self._cis_candidate_idle_grace <= 0:
            raise ValueError("VLLM_CIS_CANDIDATE_IDLE_GRACE must be positive")
        self._cis_last_adjusted = time.monotonic()
        self._cis_preemptions = 0
        self._cis_observed_preemptions = 0
        self._cis_completed_steps = 0
        self._cis_last_growth_completion = 0
        trace_path = os.environ.get("VLLM_CIS_SCHEDULER_TRACE")
        self._cis_trace_path = Path(trace_path) if trace_path else None
        if self._cis_trace_path is not None:
            self._cis_trace_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _metadata(request: Request) -> dict[str, Any] | None:
        params = request.sampling_params
        extra_args = None if params is None else params.extra_args
        value = None if extra_args is None else extra_args.get(_METADATA_KEY)
        if not isinstance(value, dict):
            return None
        if ":warmup:" in str(value.get("job_id", "")):
            return None
        return value

    @staticmethod
    def _step_key(metadata: dict[str, Any]) -> str:
        return f"{metadata['job_id']}:step:{int(metadata['step_index'])}"

    def _trace(self, event: str, **values: Any) -> None:
        if self._cis_trace_path is None:
            return
        payload = {
            "schema_version": 1,
            "event": event,
            "monotonic_seconds": time.monotonic(),
            "wall_time_seconds": time.time(),
            "window": self._cis_window,
            "initial_window": self._cis_initial_window,
            "active_steps": len(self._cis_active_steps),
            "deferred_steps": sum(
                bool(state.deferred) for state in self._cis_steps.values()
            ),
            "deferred_requests": sum(
                len(state.deferred) for state in self._cis_steps.values()
            ),
            "running_requests": len(self.running),
            "waiting_requests": len(self.waiting) + len(self.skipped_waiting),
            "kv_usage": self.kv_cache_manager.usage,
            "preemptions": self._cis_preemptions,
            "admission_mode": self._cis_admission_mode,
            "normalized_active_load": self._normalized_active_load(),
            **values,
        }
        with self._cis_trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def _state_for(self, metadata: dict[str, Any]) -> _StepState:
        key = self._step_key(metadata)
        state = self._cis_steps.get(key)
        if state is not None:
            return state
        candidate_count = int(metadata["candidate_count"])
        rollouts = max(0, int(metadata.get("expected_rollouts", 0) or 0))
        if candidate_count <= 0:
            raise ValueError("CIS scheduler received an invalid candidate count")
        state = _StepState(
            key=key,
            order=self._cis_next_order,
            candidate_count=candidate_count,
            rollouts_per_candidate=rollouts,
        )
        self._cis_next_order += 1
        self._cis_steps[key] = state
        if self._cis_window is None:
            fanout = candidate_count * max(1, rollouts)
            self._cis_initial_window = (
                self._cis_configured_initial_window
                if self._cis_configured_initial_window is not None
                else max(1, ceil(self.max_num_running_reqs / fanout))
            )
            self._cis_window = self._cis_initial_window
            self._trace(
                "window_initialized",
                candidate_count=candidate_count,
                rollouts_per_candidate=rollouts,
            )
        return state

    def _activate(self, state: _StepState, *, reason: str) -> None:
        if state.active:
            return
        state.active = True
        self._cis_active_steps.add(state.key)
        deferred = sorted(state.deferred.values(), key=lambda req: req.arrival_time)
        state.deferred.clear()
        for request in deferred:
            self._enqueue_waiting_request(request)
        self._trace(
            "step_activated",
            step_key=state.key,
            reason=reason,
            released_requests=len(deferred),
        )

    def _defer(self, request: Request, state: _StepState) -> None:
        state.deferred[request.request_id] = request
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    @staticmethod
    def _normalized_step_load(state: _StepState) -> float:
        if state.phase != "rollout" or state.rollout_request_count is None:
            return 1.0
        full_fanout = state.candidate_count * max(1, state.rollouts_per_candidate)
        finished = len(state.seen_rollouts) - len(state.live_request_ids)
        remaining = max(0, state.rollout_request_count - finished)
        return min(1.0, remaining / full_fanout)

    def _normalized_active_load(self) -> float:
        return sum(
            self._normalized_step_load(self._cis_steps[key])
            for key in self._cis_active_steps
            if key in self._cis_steps
        )

    def _fractional_borrow_is_safe(self) -> bool:
        assert self._cis_initial_window is not None
        if len(self._cis_active_steps) < self._cis_initial_window:
            return True
        return (
            len(self.waiting) + len(self.skipped_waiting) == 0
            and self.kv_cache_manager.usage < self._cis_kv_low_watermark
        )

    def _admit_deferred_steps(self, *, reason: str) -> None:
        assert self._cis_window is not None
        if self._cis_admission_mode == "fractional_work":
            assert self._cis_initial_window is not None
            if not self._fractional_borrow_is_safe():
                return
            available = int(
                self._cis_initial_window - self._normalized_active_load() + 1e-9
            )
        else:
            available = self._cis_window - len(self._cis_active_steps)
        if available <= 0:
            return
        deferred_states = sorted(
            (state for state in self._cis_steps.values() if state.deferred),
            key=lambda state: state.order,
        )
        for state in deferred_states[:available]:
            self._activate(state, reason=reason)

    def _can_activate_step(self) -> bool:
        assert self._cis_window is not None
        if self._cis_admission_mode == "fractional_work":
            assert self._cis_initial_window is not None
            return self._fractional_borrow_is_safe() and (
                self._normalized_active_load() + 1.0
                <= self._cis_initial_window + 1e-9
            )
        return len(self._cis_active_steps) < self._cis_window

    def add_request(self, request: Request) -> None:
        metadata = self._metadata(request)
        if metadata is None:
            super().add_request(request)
            return
        state = self._state_for(metadata)
        node_type = str(metadata["node_type"])
        candidate_index = int(metadata["candidate_index"])
        state.live_request_ids.add(request.request_id)
        state.idle_since = None
        if node_type == "rollout":
            state.phase = "rollout"
            state.rollouts_per_candidate = max(
                state.rollouts_per_candidate,
                int(metadata.get("expected_rollouts", 0) or 1),
            )
            state.seen_rollouts.add(
                (candidate_index, int(metadata.get("rollout_index", 0) or 0))
            )
            step_rollout_count = metadata.get("step_rollout_count")
            if step_rollout_count is not None:
                exact_count = int(step_rollout_count)
                if exact_count <= 0:
                    raise ValueError("a CIS rollout requires a positive step count")
                if (
                    state.rollout_request_count is not None
                    and state.rollout_request_count != exact_count
                ):
                    raise ValueError("inconsistent CIS step rollout counts")
                state.rollout_request_count = exact_count
            if not state.active:
                self._activate(state, reason="rollout_ready")
            super().add_request(request)
            return
        if node_type != "candidate":
            raise ValueError(f"unknown CIS node type: {node_type}")
        state.seen_candidates.add(candidate_index)
        assert self._cis_window is not None
        if state.active or self._can_activate_step():
            self._activate(state, reason="within_window")
            super().add_request(request)
        else:
            self._defer(request, state)

    def _step_finished(self, state: _StepState) -> bool:
        rollout_batch_complete = bool(state.seen_rollouts) and (
            state.rollout_request_count is None
            or len(state.seen_rollouts) >= state.rollout_request_count
        )
        return (
            (
                (state.phase == "rollout" and rollout_batch_complete)
                or (
                    state.rollouts_per_candidate == 0
                    and len(state.seen_candidates) == state.candidate_count
                )
            )
            and not state.live_request_ids
            and not state.deferred
        )

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        metadata = self._metadata(request)
        result = super()._free_request(request, delay_free_blocks=delay_free_blocks)
        if metadata is None:
            return result
        key = self._step_key(metadata)
        state = self._cis_steps.get(key)
        if state is None:
            return result
        state.live_request_ids.discard(request.request_id)
        state.deferred.pop(request.request_id, None)
        if (
            state.phase == "candidate"
            and state.rollouts_per_candidate > 0
            and not state.live_request_ids
            and not state.deferred
        ):
            state.idle_since = time.monotonic()
        if self._step_finished(state):
            self._cis_active_steps.discard(key)
            del self._cis_steps[key]
            self._cis_completed_steps += 1
            self._trace("step_completed", step_key=key)
            self._admit_deferred_steps(reason="step_completed")
        elif self._cis_admission_mode == "fractional_work":
            self._admit_deferred_steps(reason="rollout_capacity_released")
        return result

    def _expire_idle_candidate_steps(self) -> None:
        now = time.monotonic()
        expired = [
            state
            for state in self._cis_steps.values()
            if state.active
            and state.phase == "candidate"
            and state.idle_since is not None
            and now - state.idle_since >= self._cis_candidate_idle_grace
            and not state.live_request_ids
            and not state.deferred
        ]
        for state in expired:
            self._cis_active_steps.discard(state.key)
            del self._cis_steps[state.key]
            self._cis_completed_steps += 1
            self._trace("step_expired_without_rollout", step_key=state.key)
        if expired:
            self._admit_deferred_steps(reason="idle_step_expired")

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        self._cis_preemptions += 1
        super()._preempt_request(request, timestamp)

    def _adjust_window(self) -> None:
        if self._cis_window is None or self._cis_initial_window is None:
            return
        if self._cis_admission_mode == "fractional_work":
            return
        now = time.monotonic()
        if now - self._cis_last_adjusted < self._cis_adjust_interval:
            return
        self._cis_last_adjusted = now
        if not any(state.deferred for state in self._cis_steps.values()):
            self._cis_observed_preemptions = self._cis_preemptions
            return
        kv_usage = self.kv_cache_manager.usage
        new_preemptions = self._cis_preemptions - self._cis_observed_preemptions
        self._cis_observed_preemptions = self._cis_preemptions
        old_window = self._cis_window
        reason = None
        if new_preemptions > 0 or kv_usage >= self._cis_kv_high_watermark:
            self._cis_window = max(self._cis_initial_window, self._cis_window - 1)
            reason = "preemption" if new_preemptions > 0 else "kv_high"
        else:
            target = ceil(
                self.max_num_running_reqs * self._cis_target_running_fraction
            )
            waiting = len(self.waiting) + len(self.skipped_waiting)
            if (
                self._cis_completed_steps > self._cis_last_growth_completion
                and len(self.running) < target
                and waiting == 0
                and kv_usage < self._cis_kv_low_watermark
            ):
                self._cis_window += 1
                self._cis_last_growth_completion = self._cis_completed_steps
                reason = "underfilled"
        if self._cis_window != old_window:
            self._trace(
                "window_adjusted",
                old_window=old_window,
                reason=reason,
                new_preemptions=new_preemptions,
            )
            if self._cis_window > old_window:
                self._admit_deferred_steps(reason="window_grew")

    def schedule(self):
        self._expire_idle_candidate_steps()
        output = super().schedule()
        self._adjust_window()
        return output

    def get_request_counts(self) -> tuple[int, int]:
        running, waiting = super().get_request_counts()
        deferred = sum(len(state.deferred) for state in self._cis_steps.values())
        return running, waiting + deferred

    def get_num_unfinished_requests(self) -> int:
        unfinished = super().get_num_unfinished_requests()
        deferred = sum(len(state.deferred) for state in self._cis_steps.values())
        return unfinished + deferred
