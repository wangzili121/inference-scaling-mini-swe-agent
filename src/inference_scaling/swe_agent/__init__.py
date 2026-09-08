"""Conditional IS integration for mini-SWE-agent."""

from __future__ import annotations

from typing import Any


__all__ = ["ConditionalISModel", "ConditionalISRunner", "load_service_config"]


def __getattr__(name: str) -> Any:
    if name == "ConditionalISModel":
        from inference_scaling.swe_agent.model import ConditionalISModel

        return ConditionalISModel
    if name in {"ConditionalISRunner", "load_service_config"}:
        from inference_scaling.swe_agent.service import (
            ConditionalISRunner,
            load_service_config,
        )

        return {
            "ConditionalISRunner": ConditionalISRunner,
            "load_service_config": load_service_config,
        }[name]
    raise AttributeError(name)
