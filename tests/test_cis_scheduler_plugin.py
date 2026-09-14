from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler

from inference_scaling.arllm.backends.cis_scheduler import (
    CISTreeScheduler,
    _StepState,
)


def _stub_scheduler_init(self, *args, **kwargs) -> None:
    del args, kwargs
    self.policy = SchedulingPolicy.PRIORITY
    self.cache_config = SimpleNamespace(num_gpu_blocks=3976)
    self.block_size = 128
    self.max_num_running_reqs = 256
    self.kv_cache_manager = SimpleNamespace(usage=0.0)
    self.running = []
    self.waiting = []
    self.skipped_waiting = []
    self.log_stats = False
    self.requests = {}


def test_runtime_kv_budget_uses_engine_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Scheduler, "__init__", _stub_scheduler_init)
    monkeypatch.setenv("VLLM_CIS_ADMISSION_MODE", "runtime_kv_budget")
    monkeypatch.setenv("VLLM_CIS_KV_CAPACITY_FRACTION", "0.9")
    monkeypatch.delenv("VLLM_CIS_PEAK_TOKEN_BUDGET", raising=False)
    monkeypatch.delenv("VLLM_CIS_SCHEDULER_TRACE", raising=False)

    scheduler = CISTreeScheduler()
    state = _StepState(
        key="job:step:0",
        job_id="job",
        order=0,
        candidate_count=2,
        rollouts_per_candidate=3,
        prompt_tokens=129,
        candidate_max_tokens=65,
        rollout_max_tokens=129,
    )

    assert scheduler._cis_runtime_kv_capacity_tokens == 508928
    assert scheduler._cis_peak_token_budget == 458035
    assert scheduler._peak_step_token_load(state) == 2048
    assert scheduler._peak_budget_can_fit(state) is True


def test_scheduler_rejects_unknown_metadata_schema() -> None:
    request = SimpleNamespace(
        sampling_params=SimpleNamespace(
            extra_args={"cis_request": {"schema_version": 2}}
        )
    )

    with pytest.raises(ValueError, match="unsupported CIS request metadata schema"):
        CISTreeScheduler._metadata(request)
