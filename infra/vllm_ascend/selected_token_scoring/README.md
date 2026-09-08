# vLLM 0.18 tiled selected-token scoring fallback

This guarded patch is pinned to upstream vLLM `v0.18.0` commit `bcf2be9`.
It changes only the `prompt_logprobs=0` path and is disabled by default.

Apply it inside the vLLM-Ascend v0.18 environment:

```bash
python infra/vllm_ascend/selected_token_scoring/apply_patch.py
```

Enable it only for the long-context reference/fallback arm:

```bash
export VLLM_TILED_SELECTED_TOKEN_LOGPROBS=1
export VLLM_SELECTED_TOKEN_VOCAB_TILE_SIZE=8192
```

The implementation scans vocabulary tiles and avoids a full FP32
`[tokens, vocabulary]` temporary. Earlier 8K experiments completed without OOM,
but this Python implementation saturated near `0.06-0.065 jobs/s` and uses a
different reduction order from full `log_softmax`. It is therefore not the
default production path. Retention requires numerical replay plus end-to-end
quality and performance gates on the fixed SWE-agent workload.
