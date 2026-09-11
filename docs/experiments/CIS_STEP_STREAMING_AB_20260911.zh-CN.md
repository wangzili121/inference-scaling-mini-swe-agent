# Conditional IS candidate-rollout 流式实验

日期：2026-09-11
分支：`cis-tree-streaming`
模型：`Qwen3-Coder-30B-A3B-Instruct` BF16
部署：vLLM-Ascend v0.18 MRV1，双卡 TP2，`MNS=256`，`MBT=32768`，memory=0.90
负载：固定 64 个公开 SWE-agent 调用快照，workers=32

## 实验问题

本实验只回答一个窄问题：candidate 请求完成后，不等本 step 的其余
candidate，立即向同一个 AsyncLLM engine 提交该 candidate 的 R 个 rollout，
是否能够通过 candidate/rollout 重叠缩短完整 Conditional IS job。

它不是 engine-native tree：candidate 和 rollout 仍是独立的 vLLM
`EngineCoreRequest`，仍分别排队、调度、申请 KV block 和处理抢占。实现只是让
rollout 更早进入 ready queue，并保留 step admission 和 step FIFO priority。

## 正式结果

下表的百分比均为 streamed 相对已有 Step 调度的变化。延迟下降为正向；吞吐
上升为正向。

| 配置 | jobs/s | Job mean / P95 | Step E2E mean / P95 | Barrier mean / P95 | candidate-rollout overlap mean / P95 | max in-flight | Preempt |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 `C15/R3`, Step | 0.1688 | 134.7 / 246.1s | 119.5 / 238.9s | 22.3 / 43.1s | 0 / 0ms | 64 | 0 |
| P0, Step-streaming | 0.1618 `-4.2%` | 134.1 `-0.5%` / 245.3s `-0.3%` | 114.2 `-4.5%` / 235.9s `-1.2%` | 21.5 `-3.6%` / 49.5s `+14.9%` | 71 / 369ms | 81 | 0 |
| P1 `C8/R3`, Step | 0.2752 | 97.5 / 173.2s | 85.2 / 168.3s | 29.7 / 75.4s | 0 / 0ms | 114 | 0 |
| P1, Step-streaming | 0.2760 `+0.3%` | 96.6 `-0.9%` / 167.6s `-3.2%` | 88.0 `+3.3%` / 165.1s `-1.9%` | 28.0 `-5.9%` / 79.7s `+5.8%` | 19 / 73ms | 125 | 0 |
| P2 `C4/R2`, same-host Step | 0.3977 | 65.2 / 144.1s | 54.7 / 133.5s | 16.2 / 59.7s | 0 / 0ms | 121 | 0 |
| P2, same-host Step-streaming | 0.2877 `-27.7%` | 78.7 `+20.7%` / 173.9s `+20.7%` | 56.4 `+3.1%` / 151.1s `+13.2%` | 16.3 `+0.6%` / 46.6s `-21.9%` | 11 / 31ms | 117 | 7 |

P0/P1 的 Step 对照来自另一台同型号 910B3 机器，适合判断大方向，但不能把
几个百分点解释成稳定收益。P2 在同一台机器、同一对 NPU 6/7 上补做了对照，
因此 P2 的退化证据更强。

P2 两次运行走过的随机轨迹不同：Step 完成 76 steps、生成 47,543 tokens；
Step-streaming 完成 89 steps、生成 72,451 tokens。即便考虑这点，工作归一化
吞吐仍从 5,794 降到 4,511 forward-token-slots/s，下降 22.2%，并出现 7 次
KV preemption，所以不能把回退只归因于生成工作量增加。

## 失败但重要的边界实验

P0 最初将同时执行的 rollout submission group 限制为 2。结果 max in-flight
只有 30，jobs/s 仅 0.1172，Job mean 205.5s，step admission mean 142.2s。
它证明对 fanout 做静态小上限会饿死 engine；后续正式实验将上限放宽到 C，
保留该失败数据用于约束下一版 admission controller。

## 结论

1. candidate 和 rollout 已经发生真实重叠，但只有毫秒量级。P0/P1/P2 的 P95
   overlap 分别为 369ms、73ms、31ms，而一个 step 通常需要 50--120s。
2. 同一批固定长度 candidate 的完成时间本来就很接近，等待最后一个 candidate
   不是当前主要 barrier；单纯“完成即提交”无法释放显著关键路径。
3. 更早、更多次提交 rollout 会打碎 batch 并增加并发状态。P2 出现 preemption
   和 22.2% 工作归一化吞吐下降，是明确反例。
4. 因而不能把算法层 streaming 当作最终 tree 方案。真正值得继续验证的是：
   一个逻辑父请求管理同胞分支、KV continuation/fork、树级 admission/preemption，
   以及 attention 对共享 trunk 的一次读取多分支复用。

## 下一层原型的验收方式

下一层命名为 `native-tree`，不覆盖本实验。先复用 vLLM v1 的
`ParentRequest` 验证 candidate/rollout sibling grouping，随后才考虑动态两层
tree。每个 P0/P1/P2 都必须有同机 control，并报告：

- 完整 Job mean/P50/P95/P99 和 jobs/s；
- Step E2E、barrier、queue、preemption；
- forward-token-slots/s、生成 token/s 和实际生成工作量；
- frontend parent 调用数与 EngineCore child request 数；
- candidate 完成到 child admit 的时间；
- KV block 复用、重新 prefill 和树级抢占量。

只有完整 job 指标与工作归一化指标同时改善，才进入 EngineCore 动态树/KV fork。

## 原始数据

- `docs/experiments/data/cis_step_streaming_full_20260911.json`
- `docs/experiments/data/cis_step_streaming_p2_same_host_20260911.json`
- `docs/experiments/data/cis_step_streaming_underfill_20260911.json`
- 服务器：`wangzili@189.1.227.167`
- 原始目录：`/data/disk/wangzili/cis-step-streaming/tree-streaming-20260911`
