"""Checks that do not require an NPU or a downloaded 0731 checkpoint."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from inference_scaling.swe_agent.dsv4_preflight import inspect_snapshot
from inference_scaling.swe_agent.service import load_service_config
from inference_scaling.swe_agent.tool_calls import (
    ToolCallParseError,
    parse_deepseek_v4_text,
)


ROOT = Path(__file__).resolve().parents[1]


class DSV4BaselineTests(unittest.TestCase):
    def test_ordinary_cis_config_disables_tree_patches(self) -> None:
        with patch.dict("os.environ", {"CIS_MODEL_PATH": "/models/dsv4"}):
            config = load_service_config(
                ROOT / "configs/dsv4_flash/conditional_is_0731.toml"
            )
        self.assertEqual(config["models"]["base"], "/models/dsv4")
        self.assertEqual(config["vllm"]["tokenizer_mode"], "deepseek_v4")
        self.assertEqual(config["service"]["tool_parser"], "deepseek_v4")
        self.assertEqual(
            config["service"]["trace_path"],
            "/artifacts/traces/dsv4_model_calls.jsonl",
        )
        self.assertEqual(config["vllm"]["quantization"], "ascend")
        for name in (
            "native_kv_fork", "native_packed_forest_attention",
            "native_parallel_sampling", "native_kv_branch_eviction",
        ):
            self.assertFalse(config["vllm"][name])
        self.assertFalse(config["conditional_is"]["engine_fork_candidate_rollouts"])

    def test_snapshot_finds_missing_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text('{"model_type":"deepseek_v4"}')
            (model / "tokenizer_config.json").write_text("{}")
            (model / "tokenizer.json").write_text("{}")
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "model-00001.safetensors"}})
            )
            with self.assertRaises(FileNotFoundError):
                inspect_snapshot(model)
            (model / "model-00001.safetensors").write_bytes(b"test")
            result = inspect_snapshot(model)
            self.assertEqual(result["shard_count"], 1)
            self.assertEqual(result["model_type"], "deepseek_v4")

    def test_vllm_parser_output_is_normalized(self) -> None:
        parser_module = ModuleType("vllm.parser.deepseek_v4")

        class FakeParser:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def extract_tool_calls(self, text, request):
                self.assert_request = request
                return SimpleNamespace(
                    content="Done.",
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="bash", arguments='{"command":"pwd"}'
                            )
                        )
                    ],
                )

        parser_module.DeepSeekV4Parser = FakeParser
        with patch.dict(
            "sys.modules",
            {
                "vllm": ModuleType("vllm"),
                "vllm.parser": ModuleType("vllm.parser"),
                "vllm.parser.deepseek_v4": parser_module,
            },
        ):
            result = parse_deepseek_v4_text(
                "<\uff5cDSML\uff5ctool_calls>",
                request_id="job-1", tokenizer=object(),
            )
        self.assertEqual(result.content, "Done.")
        self.assertEqual(result.actions[0]["command"], "pwd")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "bash")

    def test_parser_rejects_non_bash_calls(self) -> None:
        parser_module = ModuleType("vllm.parser.deepseek_v4")

        class FakeParser:
            def __init__(self, tokenizer):
                pass

            def extract_tool_calls(self, text, request):
                return SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(function=SimpleNamespace(
                        name="python", arguments='{"command":"print(1)"}'
                    ))],
                )

        parser_module.DeepSeekV4Parser = FakeParser
        with patch.dict(
            "sys.modules",
            {
                "vllm": ModuleType("vllm"),
                "vllm.parser": ModuleType("vllm.parser"),
                "vllm.parser.deepseek_v4": parser_module,
            },
        ):
            with self.assertRaises(ToolCallParseError):
                parse_deepseek_v4_text(
                    "<\uff5cDSML\uff5ctool_calls>",
                    request_id="job-1", tokenizer=object(),
                )

    def test_plain_deepseek_text_falls_back_to_decoded_content(self) -> None:
        parser_module = ModuleType("vllm.parser.deepseek_v4")

        class FakeParser:
            def __init__(self, tokenizer):
                pass

            def extract_tool_calls(self, text, request):
                return SimpleNamespace(content=None, tool_calls=[])

        parser_module.DeepSeekV4Parser = FakeParser
        with patch.dict(
            "sys.modules",
            {
                "vllm": ModuleType("vllm"),
                "vllm.parser": ModuleType("vllm.parser"),
                "vllm.parser.deepseek_v4": parser_module,
            },
        ):
            result = parse_deepseek_v4_text(
                "Complete Python solution.",
                request_id="plain-1", tokenizer=object(),
            )
        self.assertEqual(result.content, "Complete Python solution.")
        self.assertEqual(result.tool_calls, ())


if __name__ == "__main__":
    unittest.main()
