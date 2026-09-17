"""Token-level thinking/content boundaries independent of a model backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from inference_scaling.shared.types import TokenSequence


def find_token_sequence(
    tokens: TokenSequence, marker: TokenSequence, *, start: int = 0
) -> int | None:
    if not marker:
        raise ValueError("a token marker must be nonempty")
    for index in range(start, len(tokens) - len(marker) + 1):
        if tuple(tokens[index : index + len(marker)]) == tuple(marker):
            return index
    return None


@dataclass(frozen=True, slots=True)
class OutputSegments:
    thinking_token_ids: TokenSequence
    content_token_ids: TokenSequence
    thinking_start: int | None
    thinking_end: int | None
    boundary_end: int | None
    status: str
    ended_by_eos: bool
    format_name: str | None = None

    @property
    def has_complete_thinking(self) -> bool:
        return self.status == "complete"


class OutputParser(Protocol):
    def split(
        self,
        prompt: TokenSequence,
        completion: TokenSequence,
        *,
        eos_token_id: int | None = None,
    ) -> OutputSegments: ...

    def describe(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class ThinkingFormat:
    end_token_ids: TokenSequence
    start_token_ids: TokenSequence | None = None
    starts_in_thinking: bool | None = None
    name: str = "custom"

    def __post_init__(self) -> None:
        for name in ("end_token_ids", "start_token_ids"):
            marker = getattr(self, name)
            if marker is None and name == "start_token_ids":
                continue
            if not marker or any(not isinstance(token, int) or token < 0 for token in marker):
                raise ValueError(f"{name} must contain non-negative token IDs")
            object.__setattr__(self, name, tuple(marker))
        if self.start_token_ids == self.end_token_ids:
            raise ValueError("thinking start and end markers must differ")

    def prompt_is_thinking(self, prompt: TokenSequence) -> bool:
        if self.starts_in_thinking is not None:
            return self.starts_in_thinking
        if self.start_token_ids is None:
            return True
        latest_start = latest_end = -1
        for index in range(len(prompt)):
            if prompt[index : index + len(self.start_token_ids)] == self.start_token_ids:
                latest_start = index
            if prompt[index : index + len(self.end_token_ids)] == self.end_token_ids:
                latest_end = index
        return latest_start > latest_end

    def split(
        self,
        prompt: TokenSequence,
        completion: TokenSequence,
        *,
        eos_token_id: int | None = None,
    ) -> OutputSegments:
        tokens = tuple(completion)
        ended = eos_token_id is not None and eos_token_id in tokens
        if ended:
            tokens = tokens[: tokens.index(eos_token_id)]
        if self.prompt_is_thinking(tuple(prompt)):
            start = 0
        elif self.start_token_ids is not None:
            marker_start = find_token_sequence(tokens, self.start_token_ids)
            start = None if marker_start is None else marker_start + len(self.start_token_ids)
        else:
            start = None
        if start is None:
            return OutputSegments((), tokens, None, None, None, "absent", ended)
        end = find_token_sequence(tokens, self.end_token_ids, start=start)
        if end is None:
            return OutputSegments(tokens[start:], (), start, None, None, "incomplete", ended)
        boundary_end = end + len(self.end_token_ids)
        thinking = tokens[start:end]
        return OutputSegments(
            thinking,
            tokens[boundary_end:],
            start,
            end,
            boundary_end,
            "complete" if thinking else "empty",
            ended,
        )

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "start_token_ids": self.start_token_ids,
            "end_token_ids": self.end_token_ids,
            "starts_in_thinking": self.starts_in_thinking,
        }


def full_sequence_segments(
    completion: TokenSequence, *, eos_token_id: int | None = None, reason: str
) -> OutputSegments:
    tokens = tuple(completion)
    ended = eos_token_id is not None and eos_token_id in tokens
    if ended:
        tokens = tokens[: tokens.index(eos_token_id)]
    return OutputSegments((), tokens, None, None, None, reason, ended)


@dataclass(frozen=True, slots=True)
class ThinkingParser:
    formats: tuple[ThinkingFormat, ...] = ()
    mode: Literal["auto", "enabled", "disabled"] = "auto"
    is_blank: Callable[[TokenSequence], bool] | None = field(
        default=None, compare=False, repr=False
    )

    def split(
        self,
        prompt: TokenSequence,
        completion: TokenSequence,
        *,
        eos_token_id: int | None = None,
    ) -> OutputSegments:
        def fallback(reason: str) -> OutputSegments:
            return full_sequence_segments(
                completion, eos_token_id=eos_token_id, reason=reason
            )

        if self.mode == "disabled":
            return fallback("disabled")
        if not self.formats:
            return fallback("unrecognized_format")
        matches: list[tuple[OutputSegments, ThinkingFormat]] = []
        for format_ in self.formats:
            resolved = format_
            if (
                self.is_blank is not None
                and format_.starts_in_thinking is None
                and format_.start_token_ids is not None
                and format_.prompt_is_thinking(prompt)
            ):
                last_open = max(
                    index
                    for index in range(len(prompt))
                    if prompt[index : index + len(format_.start_token_ids)]
                    == format_.start_token_ids
                )
                resolved = replace(
                    format_,
                    starts_in_thinking=self.is_blank(
                        prompt[last_open + len(format_.start_token_ids) :]
                    ),
                )
            item = resolved.split(prompt, completion, eos_token_id=eos_token_id)
            if item.thinking_start is not None:
                matches.append((item, resolved))
        if not matches:
            return fallback("absent")
        item, format_ = min(matches, key=lambda match: match[0].thinking_start or 0)
        assert item.thinking_start is not None
        if item.thinking_start and format_.start_token_ids is not None:
            leading = completion[
                : item.thinking_start - len(format_.start_token_ids)
            ]
            if leading and self.is_blank is not None and not self.is_blank(leading):
                return fallback("leading_content")
        for other in self.formats:
            if other.start_token_ids is not None:
                nested = find_token_sequence(
                    completion, other.start_token_ids, start=item.thinking_start
                )
                if nested is not None and (
                    item.thinking_end is None or nested < item.thinking_end
                ):
                    return fallback("malformed")
        if item.status != "complete":
            return fallback(item.status)
        if self.is_blank is not None and self.is_blank(item.thinking_token_ids):
            return fallback("empty")
        return replace(item, format_name=format_.name)

    def describe(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "formats": [format_.describe() for format_ in self.formats],
            "fallback": "full_sequence",
        }


__all__ = [
    "OutputParser",
    "OutputSegments",
    "ThinkingFormat",
    "ThinkingParser",
    "find_token_sequence",
    "full_sequence_segments",
]
