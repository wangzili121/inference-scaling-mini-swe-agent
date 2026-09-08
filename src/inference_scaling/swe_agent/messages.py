"""Lightweight chat-template helpers shared by runtime and offline tools."""

from __future__ import annotations

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
    return [
        {key: value for key, value in message.items() if key in allowed}
        for message in messages
    ]


__all__ = ["BASH_TOOL", "public_messages"]
