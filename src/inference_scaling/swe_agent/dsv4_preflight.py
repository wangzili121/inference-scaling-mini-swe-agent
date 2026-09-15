"""Fail-fast checks for an offline DeepSeek-V4-Flash Conditional IS deployment."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path

from inference_scaling.swe_agent.messages import BASH_TOOL
from inference_scaling.swe_agent.tool_calls import parse_deepseek_v4_text


def inspect_snapshot(model_dir: Path) -> dict[str, object]:
    required = ("config.json", "tokenizer_config.json")
    for name in required:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)
    if not (
        (model_dir / "tokenizer.json").is_file()
        or (model_dir / "tokenizer.model").is_file()
    ):
        raise FileNotFoundError("tokenizer.json or tokenizer.model is missing")
    config = json.loads((model_dir / "config.json").read_text())
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted(path.name for path in model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError("no safetensors weights found")
    missing = [name for name in shards if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} safetensors shards: {missing[:4]}"
        )
    digest = hashlib.sha256()
    for name in (*required, "tokenizer.json", "model.safetensors.index.json"):
        path = model_dir / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return {
        "model_type": config.get("model_type"),
        "quantization_config": config.get("quantization_config"),
        "shard_count": len(shards),
        "total_shard_bytes": sum((model_dir / name).stat().st_size for name in shards),
        "metadata_sha256": digest.hexdigest(),
    }


def inspect_runtime(model_dir: Path) -> dict[str, object]:
    import vllm
    from transformers import AutoTokenizer
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer

    if not vllm.__version__.startswith("0.26."):
        raise RuntimeError(f"expected vLLM 0.26 runtime, found {vllm.__version__}")
    ascend_version = version("vllm-ascend")

    args = inspect.signature(AsyncEngineArgs).parameters
    required = {
        "tokenizer_mode",
        "enable_expert_parallel",
        "disable_hybrid_kv_cache_manager",
        "block_size",
        "model_loader_extra_config",
        "compilation_config",
        "additional_config",
        "logprobs_mode",
    }
    missing = sorted(required - set(args))
    if missing:
        raise RuntimeError(f"AsyncEngineArgs in this image lacks: {missing}")
    tokenizer = get_deepseek_v4_tokenizer(
        AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=True
        )
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Use bash to print ready."}],
        tools=[BASH_TOOL],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not rendered:
        raise RuntimeError("DeepSeek V4 chat template returned an empty prompt")
    synthetic = (
        '<\uff5cDSML\uff5ctool_calls><\uff5cDSML\uff5cinvoke name="bash">'
        '<\uff5cDSML\uff5cparameter name="command" string="true">echo ready'
        '</\uff5cDSML\uff5cparameter></\uff5cDSML\uff5cinvoke></\uff5cDSML\uff5ctool_calls>'
    )
    parsed = parse_deepseek_v4_text(
        synthetic, request_id="preflight", tokenizer=tokenizer
    )
    if len(parsed.actions) != 1 or parsed.actions[0]["command"] != "echo ready":
        raise RuntimeError("DeepSeek V4 parser could not decode a bash tool call")
    return {
        "vllm_version": vllm.__version__,
        "vllm_ascend_version": ascend_version,
        "tokenizer_probe": "ok",
        "tool_parser_probe": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    result = inspect_snapshot(args.model_dir)
    if args.runtime:
        result.update(inspect_runtime(args.model_dir))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
