# Upstream sources

This repository started from a source-only snapshot of Chang's
`inference_scaling` repository. Upstream Git history was deliberately not
imported.

| Component | Version | Imported on |
| --- | --- | --- |
| inference_scaling | `04492fe1e227e7171f9f20c971240c2d5ac3c535` | 2026-09-08 |
| mini-SWE-agent integration target | `v2.4.6` | 2026-09-08 |
| vLLM-Ascend runtime target | `v0.18`, MRV1 | 2026-09-08 |

The first repository commit, `ca87966`, is the unmodified
`inference_scaling` source snapshot. All SWE-bench and deployment work is kept
in later commits so it can be reviewed and benchmarked independently.
