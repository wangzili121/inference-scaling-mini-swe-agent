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

## 三组配置统一结果

表中的 `Step` 使用各配置当前最佳的已测 hard-step cap：P0 为 6，P1 为 16，P2 为 32。P1 最初按 `MNS/(C x R)` 得到的 cap 11 已被后续 sweep 判定为过紧，不作为最终 Step 代表值。

| 配置 | 调度 | jobs/s（变化） | Job mean / P95（变化） | Step E2E mean / P95（变化） | Barrier mean / P95（变化） | Preemptions（变化） |
|---|---|---:|---:|---:|---:|---:|
| P0 `C15/R3` | Baseline | 0.1463 | 168.5 / 333.9 s | 147.5 / 316.9 s | 71.0 / 189.4 s | 10 |
| P0 `C15/R3` | Step cap 6 | 0.1688 (+15.3%) | **134.7 (-20.0%) / 246.1 s (-26.3%)** | **119.5 (-18.9%) / 238.9 s (-24.6%)** | **22.3 (-68.6%) / 43.1 s (-77.3%)** | **0 (-100%)** |
| P0 `C15/R3` | Elastic | **0.1707 (+16.6%)** | 147.5 (-12.5%) / 279.5 s (-16.3%) | 130.8 (-11.3%) / 243.9 s (-23.0%) | 60.3 (-15.0%) / 153.6 s (-18.9%) | **0 (-100%)** |
| P1 `C8/R3` | Baseline | 0.2352 | 110.7 / 223.6 s | 94.3 / 210.6 s | 45.5 / 114.5 s | 28 |
| P1 `C8/R3` | Step cap 16 | **0.2752 (+17.0%)** | **97.5 (-12.0%) / 173.2 s (-22.6%)** | **85.2 (-9.6%) / 168.3 s (-20.1%)** | **29.7 (-34.6%) / 75.4 s (-34.2%)** | **0 (-100%)** |
| P1 `C8/R3` | Elastic | 0.2502 (+6.4%) | 100.4 (-9.4%) / 205.1 s (-8.3%) | 86.6 (-8.2%) / 201.7 s (-4.2%) | 39.0 (-14.3%) / 130.0 s (+13.5%) | 17 (-39.3%) |
| P2 `C4/R2` | Baseline | 0.3329 | 71.8 / 159.0 s | 60.2 / 150.3 s | 16.2 / **45.8 s** | 4 |
| P2 `C4/R2` | Step cap 32 | 0.3564 (+7.0%) | **64.2 (-10.6%)** / 151.6 s (-4.7%) | 55.3 (-8.2%) / 133.7 s (-11.0%) | 16.0 (-0.9%) / 54.5 s (+19.1%) | 6 (+50.0%) |
| P2 `C4/R2` | Elastic | **0.3663 (+10.0%)** | 65.6 (-8.5%) / **145.5 s (-8.5%)** | **55.1 (-8.6%) / 133.4 s (-11.2%)** | **14.6 (-9.5%)** / 53.1 s (+15.9%) | **1 (-75.0%)** |

`Job latency` 是一次完整 Conditional IS 模型调用从进入到返回最终 assistant/tool-call 的时间。`Step latency` 是该 job 内一个 `B=128` block 的 candidate、rollout、reward 和 resample 完成时间，并包含本文单列的 step admission wait。一个 job 可顺序经历多个 step，因此两者不是同一指标；在聚合层面近似满足 `job mean ~= step mean x 总 step 数 / 总 job 数 + 非 step 开销`。Mean 与 P95 应同时报告：mean 描述总体平均，P95 描述最慢 5% 的尾延迟；P95 不参与 mean 的计算，也不应把多轮实验的 P95 简单再取平均。多次复验时应合并原始样本后重新计算，或同时报告各轮波动。

## P1/P2 扩展

使用完全相同的模型、runtime、engine 参数、64 个请求和 workers=32，进一步测试：

- P1：`C8/R3/B128/L512`，step-gang cap 为 `round(256/(8x3))=11`。
- P2：`C4/R2/B128/L512`，step-gang cap 为 `256/(4x2)=32`。

### P1 完整对照

| 指标 | Baseline | Gang cap 11 | 相对 baseline | Elastic | 相对 baseline |
|---|---:|---:|---:|---:|---:|
| jobs/s | 0.2352 | 0.2248 | -4.4% | **0.2502** | **+6.4%** |
| job mean | 110.7 s | **99.7 s** | -9.9% | 100.4 s | -9.4% |
| job P50 | 95.6 s | 104.3 s | +9.1% | **87.6 s** | -8.4% |
| job P95 | 223.6 s | **190.3 s** | **-14.9%** | 205.1 s | -8.3% |
| end-to-end step mean | 94.3 s | **84.9 s** | **-10.0%** | 86.6 s | -8.2% |
| end-to-end step P50 | 77.8 s | 91.9 s | +18.2% | **72.6 s** | -6.7% |
| end-to-end step P95 | 210.6 s | **157.9 s** | **-25.0%** | 201.7 s | -4.2% |
| barrier-tail mean | 45.5 s | **21.0 s** | **-53.8%** | 39.0 s | -14.3% |
| barrier-tail P95 | 114.5 s | **54.7 s** | **-52.2%** | 130.0 s | +13.5% |
| engine queue mean | 7.648 s | **0.185 s** | **-97.6%** | 5.513 s | -27.9% |
| engine queue P95 | 27.726 s | **1.851 s** | **-93.3%** | 29.593 s | +6.7% |
| preemptions | 28 | **0** | -100% | 17 | -39.3% |
| APC hit ratio | 94.30% | **95.66%** | +1.36 pp | 94.54% | +0.24 pp |
| max in-flight | 284 | **88** | -69.0% | 282 | -0.7% |
| engine requests/s | 4.211 | 3.993 | -5.2% | **4.332** | +2.9% |
| generated tokens/s | **475.0** | 464.8 | -2.2% | 450.1 | -5.2% |

P1 `cap=11` 不是“step 捆绑策略失败”：它将 queue、barrier 和 preemption 大幅压低，但 `11 x C8 = 88` 恰好成为观察到的最大 in-flight。也就是说，按理论最大 `C x R` 推导 cap 高估了同时实际存在的 rollout，硬门限让 engine 长时间喂不满；吞吐损失来自 admission underfill，而不是 locality 本身。

为分离 hard cap 与 step-FIFO priority，又固定 P1 运行了 `cap={16,22,32}`：

| 指标 | Cap 11 | Cap 16 | Cap 22 | Cap 32 | Baseline |
|---|---:|---:|---:|---:|---:|
| jobs/s | 0.2248 | **0.2752** | 0.2687 | 0.2442 | 0.2352 |
| 相对 baseline | -4.4% | **+17.0%** | +14.3% | +3.8% | - |
| job mean | 99.7 s | 97.5 s | **93.1 s** | 104.9 s | 110.7 s |
| job P95 | 190.3 s | **173.2 s** | 186.3 s | 212.1 s | 223.6 s |
| step mean | 84.9 s | 85.2 s | **83.7 s** | 93.0 s | 94.3 s |
| step P95 | **157.9 s** | 168.3 s | 179.0 s | 210.6 s | 210.6 s |
| barrier mean | **21.0 s** | 29.7 s | 32.0 s | 44.5 s | 45.5 s |
| barrier P95 | **54.7 s** | 75.4 s | 97.3 s | 121.5 s | 114.5 s |
| admission wait mean | 50.5 s | 36.5 s | 21.4 s | 0.0 s | 0.0 s |
| engine queue mean | **0.185 s** | 0.304 s | 0.753 s | 6.366 s | 7.648 s |
| preemptions | **0** | **0** | **0** | 15 | 28 |
| APC hit ratio | **95.66%** | 95.09% | 95.02% | 93.64% | 94.30% |
| max in-flight | 88 | 114 | 172 | 286 | 284 |
| forward-token-slots/s | 3464 | 4150 | 4019 | **4763** | 4604 |
| generated tokens | 132349 | 117692 | 111523 | 122009 | 129267 |

这个 sweep 形成清晰的折中曲线：cap 太小会把 engine 饿住；cap 太大则恢复长 engine queue、barrier 和 KV preemption。`16-22` 是当前 P1 的有效区间，但不能仅凭 jobs/s 宣称 cap 16 稳定快 17%，因为不同 batch shape 下的随机采样使生成 token 总量不同。更稳健的结论是：cap 16/22 都把 job/step tail 降低、preemption 清零且没有 cap 11 的明显 underfill；cap 32 的最高工作归一化吞吐说明 engine 本身更饱和，但端到端 barrier 和 queue 已回升。正式实现应采用 soft/elastic cap，而不是固定 11。

### P2 完整对照

| 指标 | Baseline | Gang cap 32 | 相对 baseline | Elastic | 相对 baseline |
|---|---:|---:|---:|---:|---:|
| jobs/s | 0.3329 | 0.3564 | +7.0% | **0.3663** | **+10.0%** |
| job mean | 71.8 s | **64.2 s** | **-10.6%** | 65.6 s | -8.5% |
| job P50 | 54.7 s | 55.3 s | +1.1% | **53.6 s** | -2.1% |
| job P95 | 159.0 s | 151.6 s | -4.7% | **145.5 s** | **-8.5%** |
| end-to-end step mean | 60.2 s | 55.3 s | -8.2% | **55.1 s** | **-8.6%** |
| end-to-end step P50 | 50.1 s | 48.2 s | -3.8% | **47.4 s** | -5.3% |
| end-to-end step P95 | 150.3 s | 133.7 s | -11.0% | **133.4 s** | **-11.2%** |
| barrier-tail mean | 16.2 s | 16.0 s | -0.9% | **14.6 s** | -9.5% |
| barrier-tail P95 | **45.8 s** | 54.5 s | +19.1% | 53.1 s | +15.9% |
| engine queue mean | 3.082 s | 3.040 s | -1.4% | **2.847 s** | -7.6% |
| engine queue P95 | 19.560 s | 19.136 s | -2.2% | **17.415 s** | -11.0% |
| preemptions | 4 | 6 | +50.0% | **1** | -75.0% |
| APC hit ratio | 88.28% | **90.55%** | +2.27 pp | 89.52% | +1.24 pp |
| max in-flight | 113 | 124 | +9.7% | 124 | +9.7% |
| engine requests/s | 2.487 | 2.673 | +7.5% | **2.747** | +10.5% |
| generated tokens/s | 260.1 | 292.2 | +12.3% | **298.7** | +14.9% |

P2 的 cap=32 没有限制 32 个 worker，因此 gang 方案实际上只是 step-FIFO priority，不是 hard bundle。它相对 baseline 的 jobs/s、job P95、step mean/P95 都改善，不能称为“捆绑不好”；elastic 又略胜一筹。P2 的 barrier P95 反而上升，说明收益主要来自队列顺序、APC locality、engine 吞吐和更少 preemption，而不是将单个 step 的最后一条 rollout 更快收齐。

综合三种 profile：

- 高 fanout P0：hard step locality 对平均/P95 step 和 barrier 最有效；elastic 的吞吐略高。
- 中 fanout P1：cap 11 太紧；cap 16-22 能在消除 preemption 的同时改善端到端 tail，elastic/soft cap 是产品化方向。
- 低 fanout P2：step-FIFO 与 elastic 都优于 baseline，hard admission 已无必要，elastic 最均衡。

因此下一版不应提供一个由 `MNS/(C x R)` 静态算出的固定 cap，而应依据当前实际 ready candidate/rollout、engine runnable slots 和 barrier-critical work 动态借还容量：高 fanout 保持较强 locality，中 fanout 使用 soft cap，低 fanout 只保留 elastic priority。

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
- P1 cap sweep 原始目录：`/data/disk/wangzili/cis-step-affinity/p1-cap-sweep-20260910/{cap-16,cap-22,cap-32}`。
- P2 调度原始目录：`/data/disk/wangzili/cis-step-affinity/p2-ab-20260910/{baseline,step-gang,step-elastic}`。
- 调度摘要：`docs/experiments/data/cis_step_affinity_ab_20260910.json`，SHA256 `1b5d213a4ac3d1c78789d7babd3d586d584369cb25291d673e84fb7fec3bef3a`。
- P1 摘要：`docs/experiments/data/cis_step_affinity_p1_20260910.json`，SHA256 `8d9b577ce8a42097ca2373c6dfb1c4600566fdb0608abee9b65447556012dd37`。
- P2 摘要：`docs/experiments/data/cis_step_affinity_p2_20260910.json`，SHA256 `3d92315b61cdef9f210bcf0b81d5739dfdd51ad0e741bf8892e296ac96a292e5`。
- P1 cap sweep：`docs/experiments/data/cis_step_affinity_p1_cap_sweep_20260910.json`，SHA256 `06e97622fe848c0a73181c091ff5d46516094b9e26583cd44eb27488e73dfe1b`。
- Attention 摘要：`docs/experiments/data/cis_forest_attention_rerun_20260910.json`，SHA256 `89df83228197b513b492471b212a36ebc98cb8dec548908d4481371e4a552792`。
- 启动脚本：`experiments/swebench/run_cis_forest_ab_remote.sh`。
- 汇总脚本：`experiments/swebench/summarize_cis_step_ab.py`。

这轮为无 profiler 的性能 A/B，避免 trace 开销改变调度结论。下一次正式双卡 P0 profiling 应同时保留 baseline 与 `step-gang-6`，以算法事件、MS Service Profiler 和 Ascend PyTorch Profiler 解释 barrier、APC locality、preemption 与 NPU batch shape 的变化。
