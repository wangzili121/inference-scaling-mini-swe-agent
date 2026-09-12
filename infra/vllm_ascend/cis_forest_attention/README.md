# Conditional IS Forest Attention experiments

This directory contains experimental vLLM/vLLM-Ascend v0.18 patches for the
shared KV structure among the `R` rollout siblings of one Conditional IS
candidate. They are research prototypes, not a production recommendation.

## Packed Forest Attention

Ordinary APC lets rollout requests reuse candidate KV blocks at admission, but
decode attention still reads the shared prefix once per query. The experimental
path groups complete, physically shared sibling sets and builds one logical KV
row per candidate:

```text
[shared candidate prefix][rollout tail 0]...[rollout tail R-1]
```

The `R` queries are packed as a contiguous BSH query sequence. A device-side
mask lets query `r` see the shared prefix and only tail `r`, so one FIA call can
replace repeated reads of the same prefix. Ordinary requests remain on native
`torch_npu._npu_paged_attention`.

The v3 bridge transfers only compact shared/tail lengths. It creates the mask
with cached device `arange` tensors and has a direct-return fast path when the
whole decode batch is eligible. It activates only when:

- every branch is a one-token decode;
- all expected siblings are present in the same forward;
- sibling block tables have the same physical shared prefix;
- the shared prefix is at least 4,096 tokens;
- at least 196,608 shared-KV token reads are avoided;
- packed queries are at least 75% of the forward;
- padding and masked-tail overhead pass their configured limits;
- PCP, DCP, speculative decoding and sliding-window attention are inactive.

Apply from the vLLM-Ascend source root:

```bash
git apply --recount vllm-ascend-0.18-packed-forest-attention.patch
```

Enable it in the inference-scaling config:

```text
vllm.native_packed_forest_attention=true
```

The bridge is exact up to expected BF16 kernel differences, but it cannot
remain in the existing `FULL_DECODE_ONLY` graph. Eligible forwards use the
piecewise path, and mixed batches also pay native/Forest split and merge costs.
On the fixed P0 64-request workload, v3 is within about 2.4% of the equivalent
dual-graph control but remains about 12%-14% behind the true FULL-graph baseline
in jobs/s and forward-token-slots/s. A production implementation therefore
requires a graph-capturable fused CANN op that accepts compact segment metadata;
further Python/FIA bridge tuning is not the target architecture.

## Forest decode window

`vllm-0.18-cis-forest-window.patch` is a narrow scheduler experiment. When the
already scheduled running decode set contains enough complete sibling groups to
cross the same saved-read and packed-fraction thresholds, it briefly withholds
new waiting prefills. This attempts to create a dense pure-decode Forest batch
without turning a whole CIS step into a device-monopolizing request.

Apply from the vLLM source root:

```bash
git apply --recount vllm-0.18-cis-forest-window.patch
```

Runtime controls:

```text
VLLM_CIS_FOREST_WINDOW_MAX_STEPS
VLLM_CIS_FOREST_WINDOW_MIN_SAVED_READS
VLLM_CIS_FOREST_WINDOW_MIN_PACKED_FRACTION
```

After `MAX_STEPS` consecutive holds, the scheduler admits waiting work for one
tick to avoid starvation. This patch should be evaluated only together with the
packed attention patch and a fixed request namespace. It is an A/B probe for
mixed prefill/decode interference, not a complete tree-aware scheduler.

The fixed P0 64-request A/B showed no throughput benefit: jobs/s changed by
-0.27% and forward-token-slots/s by -0.12%; Job mean improved by 6.6% while
P95 regressed by 6.5%. Only one scheduler window activated during the full run.
The large gain seen in a four-job smoke did not reproduce under saturated load,
so this patch is retained as a negative experiment and should not be expanded
before a graph-capturable Forest Attention op exists.

## Validation

The local planner and bridge tests live in:

```text
tests/test_packed_forest_attention.py
tests/test_vllm_backend.py
```

The remote experiment runner is:

```text
experiments/swebench/run_cis_forest_ab_remote.sh
```

Use the same workload, request order, `run_namespace`, C/R/B/L, workers and
engine configuration for every A/B. Report jobs/s, forward-token-slots/s,
Job mean/P95, generated work and preemptions together; jobs/s alone is unsafe
when small numerical changes alter later resampling and total generated work.
