"""Resolve thinking markers from explicit settings and tokenizer metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inference_scaling.shared.output import ThinkingFormat, ThinkingParser


_THINKING_MARKERS = (
    ("think", "<think>", "</think>"),
    ("thinking", "<thinking>", "</thinking>"),
    ("bracket_think", "[THINK]", "[/THINK]"),
    ("reasoning", "<reasoning>", "</reasoning>"),
)


def output_settings_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    options = dict(config.get("output", {}))
    template = config.get("prompt", {}).get("chat_template_kwargs", {})
    if options.get("thinking_mode", "auto") == "auto" and "enable_thinking" in template:
        enabled = template["enable_thinking"]
        if not isinstance(enabled, bool):
            raise ValueError("prompt.chat_template_kwargs.enable_thinking must be boolean")
        options["thinking_mode"] = "enabled" if enabled else "disabled"
    return options


def thinking_format_from_backend(
    backend: Any,
    settings: Mapping[str, Any] | None = None,
    *,
    required: bool = False,
) -> ThinkingParser:
    options = settings or {}
    mode = str(options.get("thinking_mode", "auto"))
    if mode not in {"auto", "enabled", "disabled"}:
        raise ValueError("thinking_mode must be auto, enabled or disabled")
    tokenizer = getattr(backend, "tokenizer", None)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocabulary = get_vocab() if callable(get_vocab) else {}
    template = str(getattr(tokenizer, "chat_template", "") or "")
    configured = options.get("thinking_formats")
    if configured is not None:
        definitions = list(configured)
    elif options.get("thinking_end_text") is not None:
        definitions = [
            {
                "name": "configured",
                "start_text": options.get("thinking_start_text"),
                "end_text": options["thinking_end_text"],
                "starts_in_thinking": options.get("starts_in_thinking"),
            }
        ]
    else:
        definitions = [
            {
                "name": name,
                "start_text": start,
                "end_text": end,
                "starts_in_thinking": options.get("starts_in_thinking"),
            }
            for name, start, end in _THINKING_MARKERS
            if (start in vocabulary and end in vocabulary)
            or (start in template and end in template)
        ]
    encode = getattr(backend, "encode", None) or getattr(tokenizer, "encode", None)
    formats: list[ThinkingFormat] = []
    for definition in definitions:
        if not isinstance(definition, Mapping):
            raise ValueError("each thinking format must be a table")
        start = definition.get("start_text")
        end = definition.get("end_text")
        if not isinstance(end, str) or not end:
            raise ValueError("thinking end_text must be a nonempty string")
        if start is not None and (not isinstance(start, str) or not start):
            raise ValueError("thinking start_text must be a nonempty string")
        if encode is None:
            raise ValueError("resolving thinking markers requires a tokenizer encoder")
        formats.append(
            ThinkingFormat(
                end_token_ids=tuple(encode(end, add_special_tokens=False)),
                start_token_ids=(
                    None
                    if start is None
                    else tuple(encode(start, add_special_tokens=False))
                ),
                starts_in_thinking=definition.get("starts_in_thinking"),
                name=str(definition.get("name", "configured")),
            )
        )
    if required and not formats:
        raise ValueError(
            "thinking requires a configured format or recognized tokenizer delimiters"
        )

    decode = getattr(tokenizer, "decode", None) or getattr(backend, "decode", None)

    def is_blank(tokens: tuple[int, ...]) -> bool:
        if not tokens:
            return True
        if decode is None:
            return False
        try:
            return str(
                decode(
                    tokens,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            ).strip() == ""
        except TypeError:
            return str(decode(tokens, skip_special_tokens=False)).strip() == ""

    return ThinkingParser(
        tuple(formats),
        mode="disabled" if mode == "disabled" else "enabled" if mode == "enabled" else "auto",
        is_blank=is_blank if decode is not None else None,
    )


__all__ = ["output_settings_from_config", "thinking_format_from_backend"]
