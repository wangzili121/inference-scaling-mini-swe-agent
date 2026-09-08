# AR native categorical post-optimization profile

## Scope

This profile uses the real chang AR conditional-IS workload after enabling the
retained native Ascend categorical sampler. The run keeps the existing async
vLLM engines, continuous batching, async scheduling, APC, chunked prefill,
FULL_DECODE_ONLY graph mode, and the original algorithm budgets.

- Workload: 96 GSM8K requests
- Models: Qwen2.5-1.5B base + Qwen2.5-0.5B proposal
- Precision: FP16
- Device: one Ascend NPU
- Profile root: `results/profiling/categorical_native_post_v2`

## Device-time distribution

| Operator | Calls | Device time |
| --- | ---: | ---: |
| MatMulV2 | 870,723 | 44.928% |
| MatMulV3 | 32,844 | 16.797% |
| FusedInferAttentionScore | 222,404 | 9.472% |
| SwiGlu | 222,432 | 5.168% |
| LogSoftmaxV2 | 13,687 | 4.602% |
| AddRmsNormBias | 444,888 | 4.580% |
| `_triton_rope` | 222,432 | 3.060% |
| Slice | 685,636 | 2.361% |
| Cast | 92,308 | 2.059% |
| GreaterEqual | 13,687 | 1.789% |
| ReshapeAndCacheNdKernel | 222,404 | 1.571% |
| CategoricalSample | 7,935 | 1.273% |
| ReduceSum | 13,689 | 0.935% |

## Interpretation

The sampling bottleneck has moved: `CategoricalSample` is now only 1.273% of
device time. Further sampler micro-optimization is no longer the first target.

The next removable overhead is prompt scoring. Chang always requests
`prompt_logprobs=0`, but vLLM 0.18 still materializes a full-vocabulary FP32
`log_softmax` tensor before gathering the observed token. A mathematically
equivalent selected-token path can compute

`logit[target] - logsumexp(logits)`

and derive rank directly from raw logits. This avoids materializing the full
log-softmax result while preserving token IDs, logprobs, and ranks.

## Next experiment

Implement the selected-token prompt-logprob path as a guarded vLLM patch and
compare it against the retained native-categorical baseline on the same NPU.
Keep it only if 96-request end-to-end A/B results preserve outputs and improve
wall time.
