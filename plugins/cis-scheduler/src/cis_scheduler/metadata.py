"""Versioned dependency metadata exchanged with inference runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


_SCHEMA_VERSION = 1
_NODE_TYPES = frozenset({"candidate", "rollout"})


@dataclass(frozen=True, slots=True)
class CISNode:
    job_id: str
    step_index: int
    node_type: str
    candidate_index: int
    candidate_count: int
    rollout_index: int | None = None
    expected_rollouts: int = 0
    step_rollout_count: int | None = None
    candidate_max_tokens: int | None = None
    rollout_max_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.job_id, str) or not self.job_id:
            raise ValueError("job_id must be a non-empty string")
        for name in (
            "step_index",
            "candidate_index",
            "candidate_count",
            "expected_rollouts",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.step_index < 0 or self.candidate_count <= 0:
            raise ValueError(
                "step_index must be non-negative and candidate_count positive"
            )
        if not 0 <= self.candidate_index < self.candidate_count:
            raise ValueError("candidate_index is outside candidate_count")
        if self.node_type not in _NODE_TYPES or self.expected_rollouts < 0:
            raise ValueError("invalid node_type or expected_rollouts")
        if self.node_type == "candidate" and self.rollout_index is not None:
            raise ValueError("candidate nodes cannot have a rollout_index")
        if self.node_type == "candidate" and self.step_rollout_count is not None:
            raise ValueError("candidate nodes cannot declare step_rollout_count")
        if self.node_type == "rollout" and (
            type(self.rollout_index) is not int or self.rollout_index < 0
        ):
            raise ValueError("rollout nodes require a non-negative rollout_index")
        if self.node_type == "rollout" and (
            self.expected_rollouts <= 0 or self.rollout_index >= self.expected_rollouts
        ):
            raise ValueError("rollout_index is outside expected_rollouts")
        if self.step_rollout_count is not None and (
            type(self.step_rollout_count) is not int or self.step_rollout_count < 0
        ):
            raise ValueError("step_rollout_count must be a non-negative integer")
        for name in ("candidate_max_tokens", "rollout_max_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.candidate_max_tokens == 0:
            raise ValueError("candidate_max_tokens must be positive")

    @property
    def step_key(self) -> tuple[str, int]:
        return self.job_id, self.step_index

    def to_mapping(self) -> dict[str, str | int | None]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "job_id": self.job_id,
            "step_index": self.step_index,
            "node_type": self.node_type,
            "candidate_index": self.candidate_index,
            "candidate_count": self.candidate_count,
            "rollout_index": self.rollout_index,
            "expected_rollouts": self.expected_rollouts,
            "step_rollout_count": self.step_rollout_count,
            "candidate_max_tokens": self.candidate_max_tokens,
            "rollout_max_tokens": self.rollout_max_tokens,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CISNode:
        if (
            type(value.get("schema_version")) is not int
            or value["schema_version"] != _SCHEMA_VERSION
        ):
            raise ValueError("unsupported CIS metadata schema_version")
        return cls(
            job_id=value["job_id"],
            step_index=value["step_index"],
            node_type=value["node_type"],
            candidate_index=value["candidate_index"],
            candidate_count=value["candidate_count"],
            rollout_index=value.get("rollout_index"),
            expected_rollouts=value.get("expected_rollouts", 0),
            step_rollout_count=value.get("step_rollout_count"),
            candidate_max_tokens=value.get("candidate_max_tokens"),
            rollout_max_tokens=value.get("rollout_max_tokens"),
        )
