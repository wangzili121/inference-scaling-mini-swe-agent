# Conditional IS Step Affinity 与 Forest Attention A/B

## 问题与口径

本轮只回答两个问题：

1. 同一个 Conditional IS step 的 candidate/rollout 更集中地完成，能否减少跨 step 互相拖延？
2. 将 rollout attention 按共享 trunk、candidate suffix、unique tail 拆分，当前 Ascend FIA 能否直接获益？

Step 指标分为两层，避免把 admission 排队隐藏掉：

- `active step completion`：该 step 获准执行后，从 candidate 开始到 rollout barrier、reward 和 resample 完成。
- `end-to-end step completion`：`step admission wait + active step completion`。本文选择方案时以此为主。

## 固定条件

- 模型：`Qwen3-Coder-30B-A3B-Instruct` BF16。
- Runtime：vLLM-Ascend v0.18 MRV1，双卡 TP2，物理卡 0/2。
- P0：`C15/R3/B128/L512`，sequence-logprob，`T=1/top_p=1/top_k=None`。
- Engine：`MNS=256`、`MBT=32768`、memory `0.90`、partial prefill `(1,1)`。
- 已启用 persistent AsyncLLM、APC、chunked prefill、native categorical sampler 和 `FULL_DECODE_ONLY + Npugraph_ex`。
- Workload：固定 64 个公开 SWE-agent 调用，workers=32；三组最终数据在同一 0/2 卡对上顺序运行。

对比方案：

- `baseline`：原始 continuous batching/FIFO，不暴露 CIS step 关系。
- `step-gang-6`：最多 6 个 active step；同一 step 的 candidate/rollout 共用 FIFO priority。6 来自 `6 x 45 ~= MNS 256`，不是搜索后的最优值。
- `step-elastic`：不限制 active step，把已就绪 rollout 提到普通 candidate 之前；空余容量仍可被其他 step 借用。

## 端到端结果

| 指标 | Baseline | Step gang 6 | 相对变化 | Elastic rollout-first | 相对变化 |
|---|---:|---:|---:|---:|---:|
| jobs/s | 0.1463 | 0.1688 | +15.3% | **0.1707** | **+16.6%** |
| job mean | 168.5 s | **134.7 s** | **-20.0%** | 147.5 s | -12.5% |
| job P50 | 147.4 s | 141.4 s | -4.0% | **129.3 s** | **-12.2%** |
| job P95 | 333.9 s | **246.1 s** | **-26.3%** | 279.5 s | -16.3% |
| end-to-end step mean | 147.5 s | **119.5 s** | **-18.9%** | 130.8 s | -11.3% |
| end-to-end step P50 | 109.4 s | 137.3 s | +25.6% | 118.0 s | +7.9% |
| end-to-end step P95 | 316.9 s | **238.9 s** | **-24.6%** | 243.9 s | -23.0% |
| rollout barrier-tail mean | 71.0 s | **22.3 s** | **-68.6%** | 60.3 s | -15.0% |
| rollout barrier-tail P95 | 189.4 s | **43.1 s** | **-77.3%** | 153.6 s | -18.9% |
| preemptions | 10 | **0** | -100% | **0** | -100% |
| APC token hit ratio | 96.25% | 97.08% | +0.83 pp | **97.10%** | +0.85 pp |
| engine requests/s | 4.713 | 4.985 | +5.8% | **5.360** | +13.8% |
| generated tokens/s | 543.6 | 560.1 | +3.0% | **597.7** | +9.9% |

`step-gang-6` 的 active step mean 只有 29.8 s，但 admission wait mean 为 89.7 s；因此不能把 29.8 s 当作用户观察到的 step latency。合并后的 119.5 s 才是正确主结果。

## P0 判断

若主目标是平均 step 完成时间和 barrier 长尾，`step-gang-6` 胜出：end-to-end step mean 降 18.9%，P95 降 24.6%，barrier tail mean 降 68.6%。它把 active step 峰值从 32 压到 6，将 engine queue 基本搬到可解释的 step admission queue，并消除了 KV preemption。

若主目标是总体吞吐与 job P50，`step-elastic` 更均衡：jobs/s 比 gang 高约 1.1%，job P50 更低，但 step mean、job mean 和 barrier tail 明显不如 gang。下一版值得做的是 soft step cap：保留 gang 的 step locality，并仅在不会破坏 barrier-critical bundle 时借用空余容量。

本轮是随机 `T=1` workload。相同 request seed 在 native categorical + 不同 batch shape 下没有得到逐 token 相同轨迹；两次纯 baseline 的最终 action 也只有 19/64 完全一致，因此不能把 exact-output mismatch 归因于调度策略。本文不作质量结论，并同时报告 requests/s、generated tokens/s、请求数和 prefill，以降低随机工作量差异带来的误判。两种调度收益均超过历史 baseline 吞吐波动约 2.4%，但正式质量验证仍需单独进行。

## P1/P2 扩展

使用完全相同的模型、runtime、engine 参数、64 个请求和 workers=32，进一步测试：

- P1：`C8/R3/B128/L512`，step-gang cap 为 `round(256/(8x3))=11`。
- P2：`C4/R2/B128/L512`，step-gang cap 为 `256/(4x2)=32`。

| Profile | 方案 | jobs/s | jobs/s 变化 | Step mean | Step mean 变化 | Step P95 变化 | Barrier-tail mean 变化 |
|---|---|---:|---:|---:|---:|---:|---:|
| P1 | Baseline | 0.2352 | - | 94.3 s | - | - | - |
| P1 | Step gang 11 | 0.2248 | **-4.4%** | **84.9 s** | **-10.0%** | -25.0% | -53.8% |
| P1 | Elastic | **0.2502** | **+6.4%** | 86.6 s | -8.2% | -4.2% | -14.3% |
| P2 | Baseline | 0.3329 | - | 60.2 s | - | - | - |
| P2 | Step gang 32 | 0.3564 | +7.0% | 55.3 s | -8.2% | -11.0% | -0.9% |
| P2 | Elastic | **0.3663** | **+10.0%** | **55.1 s** | **-8.6%** | **-11.2%** | -9.5% |

P1 的 hard cap 能显著缩短已获准 step 的 barrier，并把 preemption 从 28 降到 0，但限制过强使完整 burst 的 jobs/s 下降 4.4%。Elastic 将 preemption 降到 17，在不设 admission barrier 的情况下同时改善 jobs/s、job latency 和 step latency，因此是 P1 的胜出方案。P1 elastic 本轮生成 token 数少约 10.9%，所以 `+6.4% jobs/s` 不能单独归因于调度；更保守的证据是 engine requests/s 提升 2.9%、step mean 下降 8.2%。

P2 的 fanout 较小，cap=32 实际不限制 32 个 worker；此时 step-gang 主要等价于 step FIFO priority。两种 priority 都有收益，elastic 的 jobs/s 提升 10.0%，而且本轮生成 token 数更多，收益不是由更短输出造成。不过 P2 的 barrier-tail P95 在 gang/elastic 下分别增加 19.1%/15.9%，说明其收益来自更好的队列、APC locality 和较少 preemption，不是 barrier 长尾改善。

综合三种 profile：

- 高 fanout P0：hard step locality 对平均/P95 step 和 barrier 最有效；elastic 的吞吐略高。
- 中 fanout P1：使用 elastic；hard cap=11 会牺牲总吞吐。
- 低 fanout P2：使用 elastic；hard admission 已无必要。

因此下一版不应提供一个固定 cap，而应依据 `C x R`、MNS、当前 ready frontier 和 barrier-critical work 动态选择：高 fanout 启用 soft cap，低/中 fanout 只保留 elastic priority。

## Attention 结果

Attention 探针保持 P0 的 `C15/R3`，candidate suffix 和 unique tail 均为 128 token，比较原生一次 FIA 与三个现成 FIA 加 exact online-softmax merge：

| 共享 trunk | 原生 FIA P50 | 三段 FIA P50 | 相对速度 |
|---:|---:|---:|---:|
| 8K | 0.427 ms | 0.867 ms | **0.492x** |
| 16K | 0.675 ms | 0.880 ms | **0.767x** |
| 32K | 1.199 ms | 0.883 ms | **1.358x** |

因此当前 6-8K 场景不能直接接入三段 FIA：它会让 attention 约慢一倍。32K 的正收益证明共享 KV IO 存在空间，但真正可用的实现必须是单个融合 Forest Attention CANN op，在 kernel 内完成三段计算和 LSE merge，并按 trunk 长度与 active siblings 动态门控。

## 复现与原始数据

- P0 调度原始目录：`/data/disk/wangzili/cis-step-affinity/ab-20260910/{baseline,step-gang-6,step-elastic}`。
- P1 调度原始目录：`/data/disk/wangzili/cis-step-affinity/p1-ab-20260910/{baseline,step-gang,step-elastic}`。
- P2 调度原始目录：`/data/disk/wangzili/cis-step-affinity/p2-ab-20260910/{baseline,step-gang,step-elastic}`。
- 调度摘要：`docs/experiments/data/cis_step_affinity_ab_20260910.json`，SHA256 `1b5d213a4ac3d1c78789d7babd3d586d584369cb25291d673e84fb7fec3bef3a`。
- P1 摘要：`docs/experiments/data/cis_step_affinity_p1_20260910.json`，SHA256 `8d9b577ce8a42097ca2373c6dfb1c4600566fdb0608abee9b65447556012dd37`。
- P2 摘要：`docs/experiments/data/cis_step_affinity_p2_20260910.json`，SHA256 `3d92315b61cdef9f210bcf0b81d5739dfdd51ad0e741bf8892e296ac96a292e5`。
- Attention 摘要：`docs/experiments/data/cis_forest_attention_rerun_20260910.json`，SHA256 `89df83228197b513b492471b212a36ebc98cb8dec548908d4481371e4a552792`。
- 启动脚本：`experiments/swebench/run_cis_forest_ab_remote.sh`。
- 汇总脚本：`experiments/swebench/summarize_cis_step_ab.py`。

这轮为无 profiler 的性能 A/B，避免 trace 开销改变调度结论。下一次正式双卡 P0 profiling 应同时保留 baseline 与 `step-gang-6`，以算法事件、MS Service Profiler 和 Ascend PyTorch Profiler 解释 barrier、APC locality、preemption 与 NPU batch shape 的变化。
