from __future__ import annotations

from dataclasses import dataclass
import os
import unittest
from unittest.mock import patch

from inference_scaling.arllm.backends.mars_bridge import (
    bridge_order_mode,
    cis_metadata,
    cis_priority,
    running_sort_key,
)


@dataclass
class _SamplingParams:
    extra_args: object


@dataclass
class _Request:
    priority: int
    arrival_time: float
    sampling_params: _SamplingParams


def _request(priority: int, *, arrival: float = 0.0) -> _Request:
    return _Request(
        priority=priority,
        arrival_time=arrival,
        sampling_params=_SamplingParams(
            {
                "cis_request": {
                    "schema_version": 1,
                    "job_id": "job-0",
                    "step_index": 2,
                }
            }
        ),
    )


class MarsBridgeTest(unittest.TestCase):
    def test_cis_metadata_and_priority(self) -> None:
        request = _request(7)
        self.assertEqual(
            cis_metadata(request),
            {
                "schema_version": 1,
                "job_id": "job-0",
                "step_index": 2,
            },
        )
        self.assertEqual(cis_priority(request), 7)

    def test_non_cis_request_has_neutral_priority(self) -> None:
        request = _Request(9, 1.0, _SamplingParams({}))
        self.assertIsNone(cis_metadata(request))
        self.assertEqual(cis_priority(request), 0)

    def test_invalid_schema_does_not_crash_unrelated_requests(self) -> None:
        request = _Request(
            9,
            1.0,
            _SamplingParams({"cis_request": {"schema_version": "bad"}}),
        )
        self.assertIsNone(cis_metadata(request))
        self.assertEqual(cis_priority(request), 0)

    def test_mars_first_prefers_mars_level(self) -> None:
        older_step = _request(1, arrival=10.0)
        newer_step = _request(2, arrival=20.0)
        self.assertLess(
            running_sort_key(
                newer_step,
                mars_level=0,
                last_scheduled=5.0,
                mode="mars_first",
            ),
            running_sort_key(
                older_step,
                mars_level=1,
                last_scheduled=1.0,
                mode="mars_first",
            ),
        )

    def test_cis_first_prefers_older_step(self) -> None:
        older_step = _request(1, arrival=10.0)
        newer_step = _request(2, arrival=20.0)
        self.assertLess(
            running_sort_key(
                older_step,
                mars_level=3,
                last_scheduled=5.0,
                mode="cis_first",
            ),
            running_sort_key(
                newer_step,
                mars_level=0,
                last_scheduled=1.0,
                mode="cis_first",
            ),
        )

    def test_bridge_order_mode_rejects_unknown_value(self) -> None:
        with patch.dict(os.environ, {"CIS_MARS_ORDER": "unknown"}):
            with self.assertRaisesRegex(ValueError, "CIS_MARS_ORDER"):
                bridge_order_mode()
