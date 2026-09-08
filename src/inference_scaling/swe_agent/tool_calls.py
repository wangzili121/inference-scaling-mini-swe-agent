"""Parse Qwen tool-call text into mini-SWE-agent's OpenAI-style messages."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any


class ToolCallParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedAssistant:
    content: str | None
    tool_calls: tuple[dict[str, Any], ...]
    actions: tuple[dict[str, str], ...]


_TOOL_PATTERNS = (
    re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL),
    re.compile(r"<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>", re.DOTALL),
)
_TRAILING_SPECIAL = re.compile(
    r"(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|</s>)+\s*$"
)
_FUNCTION_PAYLOAD = re.compile(
    r"^\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*$", re.DOTALL
)
_PARAMETER_PAYLOAD = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)


def _tool_call_id(request_id: str, index: int) -> str:
    digest = blake2b(f"{request_id}:{index}".encode("utf-8"), digest_size=8).hexdigest()
    return f"call_{digest}"


def _decode_payload(raw: str) -> dict[str, Any]:
    value = raw.strip()
    function_match = _FUNCTION_PAYLOAD.fullmatch(value)
    if function_match is not None:
        body = function_match.group(2)
        parameters = {
            match.group(1).strip(): match.group(2).strip()
            for match in _PARAMETER_PAYLOAD.finditer(body)
        }
        remainder = _PARAMETER_PAYLOAD.sub("", body).strip()
        if remainder or not parameters:
            raise ToolCallParseError("invalid Qwen function/parameter payload")
        return {
            "name": function_match.group(1).strip(),
            "arguments": parameters,
        }
    if value.startswith("```json"):
        value = value[7:]
    elif value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    try:
        payload = json.loads(value.strip())
    except json.JSONDecodeError as error:
        raise ToolCallParseError(f"invalid tool-call JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ToolCallParseError("tool-call payload must be a JSON object")
    return payload


def _command(payload: dict[str, Any]) -> str:
    function = payload.get("function")
    function = function if isinstance(function, dict) else {}
    name = payload.get("name") or function.get("name")
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ToolCallParseError(
                f"invalid tool-call arguments JSON: {error}"
            ) from error
    if name not in {"bash", "functions.bash"}:
        raise ToolCallParseError(f"unsupported tool {name!r}; expected 'bash'")
    if not isinstance(arguments, dict) or not isinstance(arguments.get("command"), str):
        raise ToolCallParseError("bash tool call requires a string 'command'")
    return arguments["command"]


def parse_assistant_text(text: str, *, request_id: str) -> ParsedAssistant:
    matches: list[tuple[int, int, str]] = []
    for pattern in _TOOL_PATTERNS:
        matches.extend(
            (match.start(), match.end(), match.group(1))
            for match in pattern.finditer(text)
        )
    matches.sort(key=lambda item: item[0])
    non_overlapping: list[tuple[int, int, str]] = []
    for match in matches:
        if not non_overlapping or match[0] >= non_overlapping[-1][1]:
            non_overlapping.append(match)

    tool_calls: list[dict[str, Any]] = []
    actions: list[dict[str, str]] = []
    for index, (_, _, raw) in enumerate(non_overlapping):
        command = _command(_decode_payload(raw))
        call_id = _tool_call_id(request_id, index)
        arguments = json.dumps({"command": command}, separators=(",", ":"))
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": arguments},
            }
        )
        actions.append({"command": command, "tool_call_id": call_id})

    content = text
    for start, end, _ in reversed(non_overlapping):
        content = content[:start] + content[end:]
    content = _TRAILING_SPECIAL.sub("", content).strip()
    return ParsedAssistant(
        content=content or None,
        tool_calls=tuple(tool_calls),
        actions=tuple(actions),
    )


__all__ = [
    "ParsedAssistant",
    "ToolCallParseError",
    "parse_assistant_text",
]
