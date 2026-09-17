"""CIS group ordering over an engine's existing integer priority primitive."""

from __future__ import annotations

from threading import Lock
from typing import Literal

from .metadata import CISNode


PriorityMode = Literal["job_fifo", "step_fifo"]
_STABLE_PRIORITY_BASE = -1_000_000


class CISPriorityPolicy:
    def __init__(self, mode: PriorityMode = "job_fifo") -> None:
        if mode not in {"job_fifo", "step_fifo"}:
            raise ValueError("mode must be job_fifo or step_fifo")
        self.mode = mode
        self._lock = Lock()
        self._next_priority = 0
        self._priorities: dict[str | tuple[str, int], int] = {}

    def register_job_order(self, job_id: str, order: int) -> None:
        if self.mode != "job_fifo":
            raise ValueError("stable job order requires job_fifo mode")
        if (
            not job_id
            or type(order) is not int
            or not 0 <= order < -_STABLE_PRIORITY_BASE
        ):
            raise ValueError("job order must be an integer in [0, 1000000)")
        priority = _STABLE_PRIORITY_BASE + order
        with self._lock:
            existing = self._priorities.get(job_id)
            if existing is not None and existing != priority:
                raise ValueError("job order cannot change after registration")
            self._priorities[job_id] = priority

    def priority_for(self, node: CISNode | None) -> int:
        if node is None:
            return 0
        return self.priority_for_group(node.job_id, node.step_index)

    def priority_for_group(self, job_id: str, step_index: int) -> int:
        if not job_id or step_index < 0:
            raise ValueError("a priority group needs a job_id and step_index")
        key: str | tuple[str, int] = (
            job_id if self.mode == "job_fifo" else (job_id, step_index)
        )
        with self._lock:
            value = self._priorities.get(key)
            if value is None:
                value = self._next_priority
                self._next_priority += 1
                self._priorities[key] = value
            return value

    def finish_step(self, job_id: str, step_index: int) -> None:
        if self.mode == "step_fifo":
            with self._lock:
                self._priorities.pop((job_id, step_index), None)

    def finish_job(self, job_id: str) -> None:
        with self._lock:
            if self.mode == "job_fifo":
                self._priorities.pop(job_id, None)
            else:
                for key in tuple(self._priorities):
                    if isinstance(key, tuple) and key[0] == job_id:
                        del self._priorities[key]
