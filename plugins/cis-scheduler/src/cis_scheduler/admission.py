"""Token-budget admission for complete CIS steps."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Condition
from time import perf_counter

from .capacity import CapacityProvider


def derive_pressure_step_limit(
    max_num_seqs: int,
    candidate_count: int,
    rollout_count: int,
    *,
    sequence_multiplier: float = 1.5,
) -> int:
    """Derive a step window from engine capacity and peak CIS fanout."""

    if max_num_seqs <= 0 or candidate_count <= 0 or rollout_count <= 0:
        raise ValueError("sequence capacity and CIS fanout must be positive")
    if sequence_multiplier <= 0:
        raise ValueError("sequence_multiplier must be positive")
    peak_branches = candidate_count * max(1, rollout_count)
    return max(1, int(max_num_seqs * sequence_multiplier) // peak_branches)


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    active_steps: int
    active_tokens: int
    waiting_steps: int
    token_budget: int
    max_active_steps: int
    peak_active_steps: int
    peak_active_tokens: int
    admissions: int
    wait_seconds: float
    pressure_gate: bool
    pressure_active: bool
    priority_enabled_steps: int
    gate_activations: int
    gate_bypasses: int
    pressure_step_limit: int


class TokenBudgetAdmissionController:
    """FIFO admission bounded by runtime KV capacity and active step count.

    The controller reserves estimated sequence-token demand. The runtime remains
    the final allocator and may apply a stricter limit.
    """

    def __init__(
        self,
        capacity: CapacityProvider,
        *,
        max_active_steps: int,
        poll_seconds: float = 0.05,
        pressure_gate: bool = False,
        pressure_activate_fraction: float = 1.0,
        pressure_deactivate_fraction: float = 0.70,
        pressure_step_limit: int | None = None,
    ) -> None:
        if max_active_steps <= 0:
            raise ValueError("max_active_steps must be positive")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if not 0.0 < pressure_activate_fraction <= 1.0:
            raise ValueError("pressure_activate_fraction must be in (0, 1]")
        if not 0.0 < pressure_deactivate_fraction <= pressure_activate_fraction:
            raise ValueError(
                "pressure_deactivate_fraction must be in "
                "(0, pressure_activate_fraction]"
            )
        if pressure_step_limit is not None and pressure_step_limit <= 0:
            raise ValueError("pressure_step_limit must be positive")
        self.capacity = capacity
        self.max_active_steps = int(max_active_steps)
        self.poll_seconds = float(poll_seconds)
        self.pressure_gate = bool(pressure_gate)
        self.pressure_activate_fraction = float(pressure_activate_fraction)
        self.pressure_deactivate_fraction = float(pressure_deactivate_fraction)
        self.pressure_step_limit = min(
            self.max_active_steps,
            self.max_active_steps
            if pressure_step_limit is None
            else int(pressure_step_limit),
        )
        self._condition = Condition()
        self._claims: dict[str, int] = {}
        self._waiters: dict[str, tuple[int, int]] = {}
        self._priority_claims: set[str] = set()
        self._next_waiter = 0
        self._active_tokens = 0
        self._peak_active_steps = 0
        self._peak_active_tokens = 0
        self._admissions = 0
        self._wait_seconds = 0.0
        self._pressure_active = not self.pressure_gate
        self._gate_activations = 0
        self._gate_bypasses = 0
        self.transitions = 0
        self.transition_failures = 0

    @property
    def limit(self) -> int:
        return self.max_active_steps

    @property
    def active_tokens(self) -> int:
        with self._condition:
            return self._active_tokens

    @property
    def token_budget(self) -> int:
        return self.capacity.snapshot().budget_tokens

    @property
    def token_block_size(self) -> int:
        return self.capacity.snapshot().block_size

    @staticmethod
    def _round_up(tokens: int, block_size: int) -> int:
        return ((tokens + block_size - 1) // block_size) * block_size

    def _normalized_tokens(self, estimated_tokens: int) -> int:
        if type(estimated_tokens) is not int or estimated_tokens <= 0:
            raise ValueError("estimated_tokens must be a positive integer")
        return self._round_up(
            estimated_tokens,
            self.capacity.snapshot().block_size,
        )

    def _can_admit(self, claim_id: str) -> bool:
        step_limit = (
            self.pressure_step_limit
            if self.pressure_gate and self._pressure_active
            else self.max_active_steps
        )
        if len(self._claims) >= step_limit:
            return False
        first = min(self._waiters.items(), key=lambda item: item[1][1])[0]
        if first != claim_id:
            return False
        tokens = self._waiters[claim_id][0]
        budget = self.capacity.snapshot().budget_tokens
        return self._active_tokens + tokens <= budget or not self._claims

    def _activate_pressure(self) -> None:
        if self._pressure_active:
            return
        self._pressure_active = True
        self._priority_claims.update(self._claims)
        self._gate_activations += 1

    def _maybe_deactivate_pressure(self) -> None:
        if not self.pressure_gate or not self._pressure_active or self._waiters:
            return
        budget = self.capacity.snapshot().budget_tokens
        step_floor = max(
            1, int(self.pressure_step_limit * self.pressure_deactivate_fraction)
        )
        if (
            self._active_tokens <= int(budget * self.pressure_deactivate_fraction)
            and len(self._claims) <= step_floor
        ):
            self._pressure_active = False

    def priority_enabled(self, claim_id: str) -> bool:
        """Return whether subsequent children of this step use CIS priority."""

        if not isinstance(claim_id, str) or not claim_id:
            return False
        if not self.pressure_gate:
            return True
        with self._condition:
            return claim_id in self._priority_claims

    def acquire(
        self,
        claim_id: str | None = None,
        estimated_tokens: int | None = None,
        reference_sample: bool = False,
    ) -> float:
        del reference_sample
        if not isinstance(claim_id, str) or not claim_id:
            raise ValueError("claim_id must be a non-empty string")
        if estimated_tokens is None:
            raise ValueError("estimated_tokens must be a positive integer")
        tokens = self._normalized_tokens(estimated_tokens)
        started = perf_counter()
        with self._condition:
            if claim_id in self._claims or claim_id in self._waiters:
                raise RuntimeError("duplicate admission claim")
            if self.pressure_gate and not self._pressure_active:
                budget = self.capacity.snapshot().budget_tokens
                activation_budget = int(
                    budget * self.pressure_activate_fraction
                )
                projected_steps = len(self._claims) + 1
                projected_tokens = self._active_tokens + tokens
                if (
                    projected_steps <= self.pressure_step_limit
                    and projected_tokens <= activation_budget
                ):
                    self._claims[claim_id] = tokens
                    self._active_tokens = projected_tokens
                    self._peak_active_steps = max(
                        self._peak_active_steps, len(self._claims)
                    )
                    self._peak_active_tokens = max(
                        self._peak_active_tokens, self._active_tokens
                    )
                    self._admissions += 1
                    self._gate_bypasses += 1
                    self._wait_seconds += perf_counter() - started
                    self._condition.notify_all()
                    return perf_counter() - started
                self._activate_pressure()
            self._waiters[claim_id] = (tokens, self._next_waiter)
            self._next_waiter += 1
            try:
                while not self._can_admit(claim_id):
                    self._condition.wait(timeout=self.poll_seconds)
            except BaseException:
                self._waiters.pop(claim_id, None)
                self._condition.notify_all()
                raise
            del self._waiters[claim_id]
            self._claims[claim_id] = tokens
            self._active_tokens += tokens
            if self.pressure_gate:
                self._priority_claims.add(claim_id)
            self._peak_active_steps = max(self._peak_active_steps, len(self._claims))
            self._peak_active_tokens = max(
                self._peak_active_tokens, self._active_tokens
            )
            self._admissions += 1
            self._wait_seconds += perf_counter() - started
            self._condition.notify_all()
        return perf_counter() - started

    def resize(self, claim_id: str, estimated_tokens: int) -> None:
        tokens = self._normalized_tokens(estimated_tokens)
        with self._condition:
            previous = self._claims.get(claim_id)
            if previous is None:
                raise RuntimeError("unknown admission claim")
            if tokens > previous:
                raise RuntimeError("admission claim cannot exceed its reservation")
            self._claims[claim_id] = tokens
            self._active_tokens -= previous - tokens
            self._maybe_deactivate_pressure()
            self._condition.notify_all()

    def transition(
        self,
        claim_id: str,
        next_claim_id: str,
        estimated_tokens: int,
    ) -> bool:
        if not isinstance(next_claim_id, str) or not next_claim_id:
            raise ValueError("next_claim_id must be a non-empty string")
        tokens = self._normalized_tokens(estimated_tokens)
        with self._condition:
            previous = self._claims.get(claim_id)
            if previous is None or next_claim_id in self._claims:
                self.transition_failures += 1
                return False
            next_total = self._active_tokens - previous + tokens
            if self.pressure_gate and not self._pressure_active:
                activation_budget = int(
                    self.capacity.snapshot().budget_tokens
                    * self.pressure_activate_fraction
                )
                if next_total > activation_budget:
                    self._activate_pressure()
            if (
                tokens > previous
                and next_total > self.capacity.snapshot().budget_tokens
                and len(self._claims) > 1
            ):
                self.transition_failures += 1
                return False
            del self._claims[claim_id]
            self._claims[next_claim_id] = tokens
            priority_enabled = claim_id in self._priority_claims
            self._priority_claims.discard(claim_id)
            if priority_enabled or self._pressure_active:
                self._priority_claims.add(next_claim_id)
            self._active_tokens = next_total
            self.transitions += 1
            self._maybe_deactivate_pressure()
            self._condition.notify_all()
            return True

    def release(self, claim_id: str | None = None) -> None:
        with self._condition:
            tokens = self._claims.pop(claim_id, None)
            if tokens is None:
                raise RuntimeError("unknown admission claim")
            self._priority_claims.discard(claim_id)
            self._active_tokens -= tokens
            self._maybe_deactivate_pressure()
            self._condition.notify_all()

    def snapshot(self) -> AdmissionSnapshot:
        with self._condition:
            return AdmissionSnapshot(
                active_steps=len(self._claims),
                active_tokens=self._active_tokens,
                waiting_steps=len(self._waiters),
                token_budget=self.capacity.snapshot().budget_tokens,
                max_active_steps=self.max_active_steps,
                peak_active_steps=self._peak_active_steps,
                peak_active_tokens=self._peak_active_tokens,
                admissions=self._admissions,
                wait_seconds=self._wait_seconds,
                pressure_gate=self.pressure_gate,
                pressure_active=self._pressure_active,
                priority_enabled_steps=len(self._priority_claims),
                gate_activations=self._gate_activations,
                gate_bypasses=self._gate_bypasses,
                pressure_step_limit=self.pressure_step_limit,
            )
