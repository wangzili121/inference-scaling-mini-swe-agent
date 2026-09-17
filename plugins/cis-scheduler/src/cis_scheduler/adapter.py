"""A narrow adapter for vLLM-compatible request submission."""

from __future__ import annotations

from collections.abc import Mapping

from .metadata import CISNode
from .priority import CISPriorityPolicy


class VLLMRequestPolicy:
    def __init__(self, priority: CISPriorityPolicy) -> None:
        self.priority = priority

    def prepare(
        self,
        node: CISNode | None,
        extra_args: Mapping[str, object] | None = None,
    ) -> tuple[int, dict[str, object]]:
        args = self.attach_metadata(node, extra_args)
        return self.priority.priority_for(node), args

    @staticmethod
    def attach_metadata(
        node: CISNode | None,
        extra_args: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        args = dict(extra_args or {})
        if node is not None:
            args["cis_request"] = node.to_mapping()
        return args

    @staticmethod
    def attach_mapping(
        metadata: Mapping[str, object],
        extra_args: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        CISNode.from_mapping(metadata)
        args = dict(extra_args or {})
        args["cis_request"] = dict(metadata)
        return args
