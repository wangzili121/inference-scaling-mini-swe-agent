"""Lightweight chat-template helpers shared by runtime and offline tools."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


def public_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove private trace fields before applying the model chat template."""

    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    normalized = []
    for message in messages:
        public = {key: value for key, value in message.items() if key in allowed}
        tool_calls = public.get("tool_calls")
        if isinstance(tool_calls, list):
            public["tool_calls"] = [
                _normalize_tool_call(tool_call) for tool_call in tool_calls
            ]
        normalized.append(public)
    return normalized


def _normalize_tool_call(tool_call: Any) -> Any:
    if not isinstance(tool_call, Mapping):
        return tool_call
    normalized = dict(tool_call)
    function = normalized.get("function")
    if not isinstance(function, Mapping):
        return normalized
    normalized_function = dict(function)
    arguments = normalized_function.get("arguments")
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError("tool-call arguments are not valid JSON") from error
        if not isinstance(decoded, Mapping):
            raise ValueError("tool-call arguments must decode to an object")
        normalized_function["arguments"] = dict(decoded)
    normalized["function"] = normalized_function
    return normalized


__all__ = ["BASH_TOOL", "public_messages"]
