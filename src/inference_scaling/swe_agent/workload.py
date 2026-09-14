"""Freeze real mini-SWE-agent model calls into deterministic tuning manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence

from inference_scaling.swe_agent.messages import BASH_TOOL, public_messages


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as stream:
        for record in records:
            payload = json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            stream.write(payload + b"\n")
            digest.update(payload + b"\n")
    return digest.hexdigest()


def _trajectory_messages(
    value: dict[str, Any], *, source: Path
) -> list[dict[str, Any]]:
    candidates = (
        value.get("messages"),
        (value.get("trajectory") or {}).get("messages")
        if isinstance(value.get("trajectory"), dict)
        else None,
        (value.get("data") or {}).get("messages")
        if isinstance(value.get("data"), dict)
        else None,
    )
    messages = next((item for item in candidates if isinstance(item, list)), None)
    if messages is None:
        raise ValueError(f"{source} has no supported messages array")
    if not all(isinstance(item, dict) and item.get("role") for item in messages):
        raise ValueError(f"{source} contains an invalid message")
    return public_messages(messages)


def extract_call_snapshots(
    trajectory: dict[str, Any],
    *,
    source: Path,
    token_count: Callable[[Sequence[dict[str, Any]]], int] | None = None,
) -> list[dict[str, Any]]:
    """Extract each real assistant invocation without padding or truncation."""

    messages = _trajectory_messages(trajectory, source=source)
    instance_id = str(
        trajectory.get("instance_id")
        or (trajectory.get("info") or {}).get("instance_id", "")
        or source.name.removesuffix(".traj.json")
    )
    snapshots = []
    call_index = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not messages[:index]:
            continue
        prefix = messages[:index]
        diagnostics: dict[str, Any] = {
            "source": str(source),
            "instance_id": instance_id,
            "call_index": call_index,
        }
        if token_count is not None:
            diagnostics["prompt_tokens"] = token_count(prefix)
        snapshots.append(
            {
                "request_id": f"public:{instance_id}:call-{call_index}",
                "messages": prefix,
                "diagnostics": diagnostics,
            }
        )
        call_index += 1
    return snapshots


def _token_counter(model: str) -> Callable[[Sequence[dict[str, Any]]], int]:
    # Tokenization is an offline CPU task. Ascend images otherwise auto-load
    # torch_npu while importing transformers and require mounted driver libs.
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ModuleNotFoundError(
            "exact public-workload token counts require transformers"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=Path(model).exists(),
        trust_remote_code=True,
    )

    def count(messages: Sequence[dict[str, Any]]) -> int:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            tools=[BASH_TOOL],
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(tokenizer.encode(str(rendered), add_special_tokens=False))

    return count


def build_public_workload(
    source_directory: str | Path,
    output_directory: str | Path,
    *,
    model: str,
    seed: int = 20260908,
    total: int = 64,
    max_new_tokens: int = 512,
    context_margin: int = 256,
    maximum_context: int = 65536,
    minimum_prompt_tokens: int = 0,
    maximum_prompt_tokens: int | None = None,
) -> dict[str, Any]:
    if minimum_prompt_tokens < 0:
        raise ValueError("minimum_prompt_tokens must not be negative")
    if (
        maximum_prompt_tokens is not None
        and maximum_prompt_tokens <= minimum_prompt_tokens
    ):
        raise ValueError(
            "maximum_prompt_tokens must be greater than minimum_prompt_tokens"
        )
    source = Path(source_directory)
    paths = sorted(source.rglob("*.traj.json"))
    if not paths:
        raise ValueError(f"no *.traj.json files found below {source}")
    snapshots: list[dict[str, Any]] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{path} is not a JSON object")
        snapshots.extend(
            extract_call_snapshots(value, source=path)
        )
    if len(snapshots) < total:
        raise ValueError(f"need {total} public calls, found {len(snapshots)}")
    rng = random.Random(seed)
    rng.shuffle(snapshots)
    counter = _token_counter(model)
    selected: list[dict[str, Any]] = []
    lengths = []
    oversized_calls_skipped = 0
    below_range_skipped = 0
    above_range_skipped = 0
    examined_calls = 0
    for item in snapshots:
        length = counter(item["messages"])
        examined_calls += 1
        if length + max_new_tokens + context_margin > maximum_context:
            oversized_calls_skipped += 1
            continue
        if length < minimum_prompt_tokens:
            below_range_skipped += 1
            continue
        if maximum_prompt_tokens is not None and length >= maximum_prompt_tokens:
            above_range_skipped += 1
            continue
        item["diagnostics"]["prompt_tokens"] = length
        selected.append(item)
        lengths.append(length)
        if len(selected) == total:
            break
    if len(selected) < total:
        raise ValueError(
            f"need {total} calls within context {maximum_context}, found {len(selected)}"
        )
    required = max(lengths) + max_new_tokens + context_margin
    max_model_len = next(
        (
            value
            for value in (16384, 32768, 65536, 131072, 262144)
            if required <= value <= maximum_context
        ),
        None,
    )
    if max_model_len is None:
        raise ValueError(
            f"selected workload requires max_model_len >= {required}, "
            f"outside supported buckets up to {maximum_context}"
        )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    workload_path = output / f"public-{total}.jsonl"
    digest = _write_jsonl(workload_path, selected)
    metadata = {
        "schema_version": 2,
        "source": str(source.resolve()),
        "model": model,
        "seed": seed,
        "count": total,
        "workload": {"path": workload_path.name, "sha256": digest},
        "prompt_tokens": {
            "minimum": min(lengths),
            "median": statistics.median(lengths),
            "p95": sorted(lengths)[max(0, int(0.95 * len(lengths)) - 1)],
            "maximum": max(lengths),
        },
        "max_new_tokens": max_new_tokens,
        "context_margin": context_margin,
        "maximum_context": maximum_context,
        "prompt_range": {
            "minimum": minimum_prompt_tokens,
            "maximum_exclusive": maximum_prompt_tokens,
        },
        "examined_calls": examined_calls,
        "below_range_skipped": below_range_skipped,
        "above_range_skipped": above_range_skipped,
        "oversized_calls_skipped": oversized_calls_skipped,
        "required_context": required,
        "selected_max_model_len": max_model_len,
        "truncated": False,
        "padded": False,
    }
    (output / "public-manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def freeze_workload(
    trace_path: str | Path,
    output_directory: str | Path,
    *,
    seed: int = 20260908,
    total: int = 128,
) -> dict[str, Any]:
    if total <= 0 or total % 2:
        raise ValueError("total must be a positive even number")
    records = _load_jsonl(Path(trace_path))
    successful: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        request_id = str(record.get("request_id", ""))
        actions = record.get("message", {}).get("extra", {}).get("actions", [])
        if not request_id or request_id in seen or not actions:
            continue
        if not isinstance(record.get("messages"), list):
            continue
        seen.add(request_id)
        successful.append(record)
    if len(successful) < total:
        raise ValueError(
            f"need {total} unique successful calls, found {len(successful)}"
        )
    rng = random.Random(seed)
    rng.shuffle(successful)
    selected = successful[:total]
    midpoint = total // 2
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tune_path = output / f"tune-{midpoint}.jsonl"
    holdout_path = output / f"holdout-{midpoint}.jsonl"
    tune_sha = _write_jsonl(tune_path, selected[:midpoint])
    holdout_sha = _write_jsonl(holdout_path, selected[midpoint:])
    metadata = {
        "schema_version": 1,
        "source": str(Path(trace_path).resolve()),
        "seed": seed,
        "total": total,
        "tune": {"path": tune_path.name, "count": midpoint, "sha256": tune_sha},
        "holdout": {
            "path": holdout_path.name,
            "count": midpoint,
            "sha256": holdout_sha,
        },
        "prompt_tokens": {
            "minimum": min(
                record["diagnostics"]["prompt_tokens"] for record in selected
            ),
            "maximum": max(
                record["diagnostics"]["prompt_tokens"] for record in selected
            ),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace")
    parser.add_argument("--public-trajectories")
    parser.add_argument("--model")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--total", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--context-margin", type=int, default=256)
    parser.add_argument("--maximum-context", type=int, default=65536)
    parser.add_argument("--minimum-prompt-tokens", type=int, default=0)
    parser.add_argument("--maximum-prompt-tokens", type=int)
    args = parser.parse_args()
    if bool(args.trace) == bool(args.public_trajectories):
        parser.error("provide exactly one of --trace or --public-trajectories")
    if args.public_trajectories:
        if not args.model:
            parser.error("--model is required with --public-trajectories")
        result = build_public_workload(
            args.public_trajectories,
            args.output_directory,
            model=args.model,
            seed=args.seed,
            total=args.total,
            max_new_tokens=args.max_new_tokens,
            context_margin=args.context_margin,
            maximum_context=args.maximum_context,
            minimum_prompt_tokens=args.minimum_prompt_tokens,
            maximum_prompt_tokens=args.maximum_prompt_tokens,
        )
    else:
        result = freeze_workload(
            args.trace,
            args.output_directory,
            seed=args.seed,
            total=args.total,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
