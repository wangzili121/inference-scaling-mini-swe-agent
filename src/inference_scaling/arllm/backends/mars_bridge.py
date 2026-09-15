"""Optional Conditional-IS policy bridge for the Muyuan MARS scheduler.

The module deliberately depends on MARS at runtime instead of copying its
scheduler.  MARS remains the only EngineCore scheduler; this bridge only adds
CIS step locality as an ordering key and exposes the read-only KV geometry
used by the outer CIS admission controller.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any


_METADATA_KEY = "cis_request"
_ORDER_STRIDE = 1_000_000
_LEVEL_STRIDE = 100_000
_VALID_ORDER_MODES = frozenset({"mars_first", "cis_first"})


def cis_metadata(request: Any) -> Mapping[str, Any] | None:
    """Return validated v1 CIS metadata from a vLLM request, if present."""

    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, Mapping):
        return None
    value = extra_args.get(_METADATA_KEY)
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        return None
    if "job_id" not in value or "step_index" not in value:
        return None
    return value


def cis_priority(request: Any) -> int:
    """Read the priority assigned by the CIS adapter.

    Non-CIS requests remain at priority zero so installing the bridge does not
    require every request source to emit CIS metadata.
    """

    if cis_metadata(request) is None:
        return 0
    value = getattr(request, "priority", 0)
    return int(value) if value is not None else 0


def bridge_order_mode() -> str:
    value = os.environ.get("CIS_MARS_ORDER", "mars_first").strip().lower()
    if value not in _VALID_ORDER_MODES:
        choices = ", ".join(sorted(_VALID_ORDER_MODES))
        raise ValueError(f"CIS_MARS_ORDER must be one of: {choices}")
    return value


def running_sort_key(
    request: Any,
    *,
    mars_level: int,
    last_scheduled: float,
    mode: str,
) -> tuple[int | float, ...]:
    """Compose MARS policy and CIS locality without replacing either one."""

    priority = cis_priority(request)
    arrival = float(getattr(request, "arrival_time", 0.0))
    if mode == "mars_first":
        return (int(mars_level), priority, float(last_scheduled), arrival)
    if mode == "cis_first":
        return (priority, int(mars_level), float(last_scheduled), arrival)
    raise ValueError(f"unknown CIS/MARS order mode: {mode}")


try:
    from mars_offloading_plugin.request_queue import MarsRequestQueue
    from mars_offloading_plugin.scheduler import MarsScheduler
except ModuleNotFoundError as _mars_import_error:
    _MARS_IMPORT_CAUSE = _mars_import_error

    class MarsCISScheduler:  # type: ignore[no-redef]
        """Fail with an actionable error when the optional MARS wheel is absent."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "MarsCISScheduler requires the Muyuan mars-offloading-plugin "
                "for vLLM/vLLM-Ascend 0.18"
            ) from _MARS_IMPORT_CAUSE

else:

    class _CISMarsRequestQueue(MarsRequestQueue):
        """MARS prompt tiers with an explicit CIS locality tie-break."""

        def __init__(self, *, mode: str) -> None:
            super().__init__()
            self._cis_mode = mode

        def add_request(self, request: Any) -> None:
            self._seq += 1
            mars_level = self._get_level(request)
            priority = cis_priority(request)
            if self._cis_mode == "mars_first":
                level = mars_level
                order = priority * _ORDER_STRIDE + self._seq
            else:
                level = 0
                order = (
                    priority * _ORDER_STRIDE
                    + mars_level * _LEVEL_STRIDE
                    + self._seq
                )
            self._push(request, level=level, order=order)


    class MarsCISScheduler(MarsScheduler):  # type: ignore[misc,no-redef]
        """MARS scheduler with CIS step-aware locality as a composable policy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._cis_mars_order = bridge_order_mode()

            # All queues are empty during Scheduler construction.  Replacing
            # them here preserves MARS lifecycle behavior while avoiding a
            # source patch to the independently installed MARS wheel.
            self.waiting = _CISMarsRequestQueue(mode=self._cis_mars_order)
            self.skipped_waiting = _CISMarsRequestQueue(mode=self._cis_mars_order)
            # MARS deliberately keeps overflow FCFS, unlike its tiered waiting
            # queues. Keep that handoff policy intact for this optional bridge.

        def _mars_reorder_running(self, now: float) -> None:
            def sort_key(request: Any) -> tuple[int | float, ...]:
                request_id = request.request_id
                level = self._mars_effective_level(request, now)
                last_scheduled = self._mars_last_scheduled_by_req_id.get(request_id)
                if last_scheduled is None:
                    last_scheduled = now - self._mars_aging_interval_s * (
                        len(self._mars_level_quanta) + 1
                    )
                return running_sort_key(
                    request,
                    mars_level=level,
                    last_scheduled=last_scheduled,
                    mode=self._cis_mars_order,
                )

            self.running.sort(key=sort_key)

        def cis_kv_cache_geometry(self) -> dict[str, int]:
            """Expose the same read-only geometry contract as stock CIS runs."""

            num_gpu_blocks = self.cache_config.num_gpu_blocks
            if num_gpu_blocks is None or num_gpu_blocks <= 0:
                raise RuntimeError("MARS scheduler has no positive GPU KV capacity")
            return {
                "num_gpu_blocks": int(num_gpu_blocks),
                "block_size": int(self.block_size),
                "token_capacity": int(num_gpu_blocks) * int(self.block_size),
            }
