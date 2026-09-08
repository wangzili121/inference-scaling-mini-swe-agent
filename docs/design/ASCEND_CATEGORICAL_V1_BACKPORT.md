# AscendC categorical sampler backport for chang's V1 runtime

## Goal

Backport the native categorical sampler from vLLM-Ascend PR #14141 to the
v0.18 Model Runner V1 stack used by chang's real AR small-proposal workload.
The optimization must remain entirely below the algorithm layer and must be
measured on top of chang's existing asynchronous engines, continuous batching,
APC, chunked prefill, asynchronous scheduling, and decode graph mode.

Upstream implementation:

- https://github.com/vllm-project/vllm-ascend/pull/14141
- native operator commit `ce5d090`

## Why this is different from the retained transform patch

The retained patch batches only the elementwise exponential transform. It
still launches one seeded uniform RNG operation per request row and then
materializes a full `[batch, vocabulary]` random tensor.

The upstream AscendC operator samples directly from stable categorical masses:

1. apply temperature to processed logits;
2. find the row maximum;
3. compute `exp(logit - maximum)` in tiles;
4. generate one Philox draw from `(request_seed, logical_position)`;
5. select the first vocabulary item whose cumulative mass crosses the draw.

It therefore avoids the per-request RNG launch loop and the full-vocabulary
random tensor. The implementation supports FP16, BF16, and FP32 logits and a
vocabulary size up to 1,048,576.

## V1 adapter boundary

v0.18's `AscendTopKTopPSampler.forward_native` already receives processed
`logits`, a row-to-`torch.Generator` dictionary, and top-k/top-p tensors. The
adapter will:

1. keep the existing AscendC top-k/top-p filtering;
2. enable the native categorical path only when every active row has an
   independent request generator;
3. build device seed and logical-position tensors from each generator;
4. invoke `torch.ops._C_ascend.npu_categorical_sample` directly on logits;
5. advance every participating generator by the installed stack's measured
   compatibility increment;
6. fall back to the unmodified sampler for unseeded/mixed rows, unsupported
   layouts, missing operator registration, or unsupported devices.

The first implementation is intentionally conservative. Chang's workload uses
per-request seeds, so the guarded path still exercises the target workload
without changing unrelated vLLM behavior.

## Correctness contract

The native operator uses stateless Philox categorical sampling, whereas the
v0.18 path uses stateful exponential-race sampling. Both sample the same
categorical distribution, but their token streams are not expected to be
bit-exact with each other. Acceptance requires:

- identical output for repeated runs with the same seed and logical position;
- invariance to batch ordering, padding, and unrelated request rows;
- correct generator-position advancement and no advancement for inactive rows;
- statistical agreement with the requested categorical distribution;
- unchanged chang algorithm budget and no material GSM8K accuracy regression;
- end-to-end throughput improvement over the retained batched-transform
  baseline, not merely over the original vLLM sampler.

## Implementation gates

1. Build PR #14141's native ACLNN/AscendC operator against the v0.18 image.
2. Run the operator's NPU correctness suite unchanged where contracts overlap.
3. Add V1 adapter tests for generator mapping and fallback behavior.
4. Run a sampler microbenchmark only as a correctness/per-kernel gate.
5. Run same-card, reversed-order end-to-end A/B on chang's 96-request workload.
6. Retain the backport only if it beats the current `+21.3%` geometric-mean
   sampler baseline.

## Known compatibility work

PR #14141 targets vLLM 0.26 MRV2 and receives explicit request seed and logical
position tensors. V1 exposes mutable `torch.Generator` objects instead. The
adapter must own that translation; silently treating a generator offset as an
upstream MRV2 position without tests is not acceptable. The native build also
touches vLLM-Ascend's CMake, Torch schema registration, and custom OPP loading,
so the operator should be built as a version-matched extension rather than
copying only the Python wrapper.

## Current validation status

The operator and v0.18 extension compile successfully with CANN 8.5.1 for
Ascend 910B. Focused real-NPU tests passed:

- same seed and logical position are deterministic;
- changing row order does not change each row's result;
- a 4,096-sample binary distribution produced probability `0.7278` versus the
  theoretical `0.7311`;
- the V1 adapter advances each participating generator from offset `0` to `12`;
- partially seeded batches fall back to the original v0.18 sampler.

The test container must mount the host Ascend driver devices and libraries.
Importing the extension without those mounts fails during RTS initialization;
that failure is not an operator ABI failure.

## End-to-end Pair 1

The first same-NPU sequential A/B used chang's 96-request GSM8K
`conditional_is_small_proposal` workload with both AR engines, continuous
batching, asynchronous scheduling, APC, chunked prefill, and
FULL_DECODE_ONLY enabled in both arms. The comparator is the retained batched
exponential-transform sampler, not the original vLLM sampler.

| Arm | Elapsed | Throughput | P95 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| retained transform | 241.92 s | 0.3968 req/s | 240.71 s | 44.79% |
| native categorical | 213.08 s | 0.4505 req/s | 212.37 s | 43.75% |

Pair 1 improves throughput by **13.5%** over the retained sampler and reduces
P95 by **11.8%**. Both the base and proposal EngineCore logs contain the
one-time native activation marker.

The reverse-order pair confirmed the result:

| Pair | Order | Retained | Native | Speedup | P95 change | Accuracy retained/native |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | retained, native | 241.92 s | 213.08 s | 1.135x | -11.8% | 44.79% / 43.75% |
| 2 | native, retained | 237.84 s | 217.25 s | 1.095x | -8.5% | 44.79% / 44.79% |

The geometric-mean speedup is **1.115x** and mean throughput rises from
`0.4002` to `0.4462 requests/s` (**+11.5%**). The Pair 1 accuracy difference is
inside the previously measured asynchronous baseline variation, while Pair 2
matches exactly at the answer-accuracy level. The native categorical backport
is therefore retained as the new sampler candidate.

Compounding this result with the retained transform's prior `1.213x`
geometric-mean gain gives an estimated `1.352x` improvement over the original
v0.18 sampling path. This compounded number combines separate A/B campaigns;
the direct, strongest claim from this experiment remains `1.115x` over the
retained sampler.

## Rejected adapter follow-up

A shape-keyed cache for the static row mapping and unit-temperature tensors was
tested on the same 96-request path. The uncached adapter took `218.33s`
(`0.4397 req/s`), while the cached adapter took `218.55s`
(`0.4393 req/s`), a `0.10%` regression with identical `41.67%` accuracy. The
allocation is not material after native sampling, so this cache is not retained.
