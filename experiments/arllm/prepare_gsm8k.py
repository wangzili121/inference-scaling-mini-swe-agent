"""Download pinned public data and model revisions used by the reproduction."""

from __future__ import annotations

import argparse
import time
import tomllib
from pathlib import Path

from huggingface_hub import snapshot_download

from inference_scaling.shared.evaluation import download_gsm8k

from experiments.shared.artifacts import file_sha256 as _sha256


def _download_model(
    source: str,
    revision: str,
    destination: str,
    expected_weight_sha256: str,
    retries: int,
    *,
    modelscope_source: str | None = None,
) -> None:
    weight = Path(destination) / "model.safetensors"
    if weight.is_file() and _sha256(weight) == expected_weight_sha256:
        print(f"verified existing model: {destination}", flush=True)
        return
    error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"model={source} revision={revision} attempt={attempt}", flush=True)
            if modelscope_source is not None:
                from modelscope import snapshot_download as modelscope_download

                modelscope_download(
                    modelscope_source,
                    revision="master",
                    local_dir=destination,
                    max_workers=4,
                    ignore_file_pattern=["*.md"],
                )
            else:
                snapshot_download(
                    source,
                    revision=revision,
                    local_dir=destination,
                    max_workers=1,
                )
            if not weight.is_file():
                raise FileNotFoundError(weight)
            actual = _sha256(weight)
            if actual != expected_weight_sha256:
                raise ValueError(
                    f"weight checksum mismatch for {source}: "
                    f"expected {expected_weight_sha256}, got {actual}"
                )
            return
        except Exception as caught:  # network retries are intentionally broad
            error = caught
            if attempt < retries:
                print(f"download retry after {type(caught).__name__}: {caught}", flush=True)
                time.sleep(2)
    assert error is not None
    raise error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_standard.toml"))
    parser.add_argument("--train-data", type=Path, default=Path("data/gsm8k/train.jsonl"))
    parser.add_argument("--test-data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    train_dataset = download_gsm8k(args.train_data, split="train")
    test_dataset = download_gsm8k(args.test_data, split="test")
    print(f"verified GSM8K train: {train_dataset}")
    print(f"verified GSM8K test: {test_dataset}")
    if args.skip_models:
        return
    for role in ("base", "proposal"):
        _download_model(
            str(config["models"][f"{role}_source"]),
            str(config["models"][f"{role}_revision"]),
            str(config["models"][role]),
            str(config["models"][f"{role}_weight_sha256"]),
            args.retries,
            modelscope_source=config["models"].get(f"{role}_modelscope_source"),
        )
    print(
        "The RL comparison is trained locally with experiments/arllm/train_gsm8k_grpo.py.",
        flush=True,
    )


if __name__ == "__main__":
    main()
