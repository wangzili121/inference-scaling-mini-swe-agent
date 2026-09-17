"""Runtime capacity contracts without owning an engine scheduler."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import lcm
from typing import Protocol


@dataclass(frozen=True, slots=True)
class KVPoolSnapshot:
    """One resident KV pool normalized to sequence-token capacity."""

    name: str
    token_capacity: int
    block_size: int


@dataclass(frozen=True, slots=True)
class CapacitySnapshot:
    """Physical KV geometry and the policy budget derived from it."""

    token_capacity: int
    block_size: int
    budget_tokens: int
    fraction: float
    pools: tuple[KVPoolSnapshot, ...]


class CapacityProvider(Protocol):
    def snapshot(self) -> CapacitySnapshot:
        """Return the capacity visible to one CIS admission domain."""


GeometrySource = Mapping[str, object] | Callable[[], Mapping[str, object] | None]


def _pool_snapshot(value: Mapping[str, object], index: int) -> KVPoolSnapshot:
    block_size = value.get("block_size")
    token_capacity = value.get("token_capacity")
    num_blocks = value.get("num_gpu_blocks", value.get("num_blocks"))
    name = value.get("name", f"pool-{index}")
    if not isinstance(name, str) or not name:
        raise ValueError("runtime KV pool name must be a non-empty string")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("runtime KV block_size must be a positive integer")
    if token_capacity is None and type(num_blocks) is int and num_blocks > 0:
        token_capacity = num_blocks * block_size
    if type(token_capacity) is not int or token_capacity <= 0:
        raise ValueError("runtime KV token_capacity must be a positive integer")
    return KVPoolSnapshot(name, token_capacity, block_size)


class RuntimeKVCapacityProvider:
    """Adapt runtime KV geometry to the CIS token-budget controller.

    The provider does not reserve blocks and does not create a second scheduler.
    vLLM or MARS remains the owner of physical KV allocation.
    """

    def __init__(self, geometry: GeometrySource, *, fraction: float) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("capacity fraction must be in (0, 1]")
        self._geometry = geometry
        self.fraction = float(fraction)

    def snapshot(self) -> CapacitySnapshot:
        value = self._geometry() if callable(self._geometry) else self._geometry
        if not isinstance(value, Mapping):
            raise ValueError("runtime KV capacity is unavailable")
        raw_pools = value.get("kv_pools")
        if raw_pools is None:
            pools = (_pool_snapshot(value, 0),)
        else:
            if (
                not isinstance(raw_pools, Sequence)
                or isinstance(raw_pools, (str, bytes))
                or not raw_pools
            ):
                raise ValueError("runtime kv_pools must be a non-empty sequence")
            if not all(isinstance(pool, Mapping) for pool in raw_pools):
                raise ValueError("each runtime KV pool must be a mapping")
            pools = tuple(
                _pool_snapshot(pool, index) for index, pool in enumerate(raw_pools)
            )
        token_capacity = min(pool.token_capacity for pool in pools)
        block_size = lcm(*(pool.block_size for pool in pools))
        return CapacitySnapshot(
            token_capacity=token_capacity,
            block_size=block_size,
            budget_tokens=int(token_capacity * self.fraction),
            fraction=self.fraction,
            pools=pools,
        )
