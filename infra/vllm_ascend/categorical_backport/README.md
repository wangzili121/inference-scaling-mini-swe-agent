# vLLM-Ascend V1 categorical sampler backport

This directory vendors the native categorical operator from
`vllm-project/vllm-ascend` PR
[#14141](https://github.com/vllm-project/vllm-ascend/pull/14141), commit
`ce5d090`, under its original Apache-2.0 headers. It is isolated here while the
v0.18 build and Model Runner V1 adapter are developed and tested.

The files are not yet a production patch. The upstream operator targets vLLM
0.26 Model Runner V2. Before it can be used by chang's runtime, the following
version-specific pieces must be completed:

1. register `categorical_sample` in the v0.18 ACLNN build;
2. register `npu_categorical_sample` in v0.18's Torch extension;
3. translate V1 per-row `torch.Generator` state to seed/position tensors;
4. retain the original path for mixed seeded/unseeded batches;
5. pass NPU correctness and chang end-to-end A/B gates.

## Native build result

The operator now builds successfully against the v0.18 image, CANN 8.5.1, and
Ascend 910B. The resulting package contains `aclnn_categorical_sample.h`, FP32,
FP16, and BF16 kernel binaries, op tiling libraries, and the dynamic operator
metadata.

The backport required four compatibility changes:

1. replace the newer `add_op_to_compiled_list` / `add_modules_sources` CMake
   API with v0.18's explicit op-host, op-tiling, and op-proto targets;
2. remove the Ascend 950 registration unsupported by the old build parser;
3. provide v0.18's local tiling error header and C++11-compatible literals;
4. replace the newer `Compares` call with v0.18's equivalent
   `CompareScalar` API.

The v0.18 Torch extension, focused native NPU correctness test, and guarded V1
generator adapter pass. Two same-NPU reverse-order runs on chang's real
96-request workload show a `1.115x` geometric-mean speedup over the already
optimized batched-transform sampler, so the backport is retained.

The focused checks are `test_categorical_smoke.py` and
`test_v1_adapter.py`. They require the ordinary Ascend device nodes and host
driver mounts; the extension initializes only after `torch.npu.set_device`.

The implementation and acceptance contract are documented in
`docs/design/ASCEND_CATEGORICAL_V1_BACKPORT.md`.
