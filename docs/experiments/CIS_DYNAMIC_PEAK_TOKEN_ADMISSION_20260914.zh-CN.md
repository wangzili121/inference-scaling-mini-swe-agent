# CIS 动态 Peak-Token Admission 实验报告

日期：2026-09-14

## 1. 问题与目标

此前效果最好的 `Step cap + step_fifo` 用固定数量限制同时展开的 Conditional IS
step。它能显著降低 barrier 长尾和 KV preemption，但最佳 cap 会随 `C/R/L`、prompt
长度和终止比例改变。本文验证一个更细粒度的外层控制器：不再把每个 step 都视为
等成本槽位，而按其预计的物理 token/KV 峰值做动态 admission。

目标不是重写 vLLM EngineCore scheduler，而是在算法层已知 CIS 结构、vLLM 只看见扁平
request 的边界上，验证一种可产品化的动态控制是否能匹配或超过手工调出的静态 cap。

## 2. 实现

新增 `PeakTokenStepAdmissionController`。每个 step 的初始 reservation 为：

```text
shared prompt/state（只计一次）
+ C * candidate_max_tokens
+ planned_rollout_requests * rollout_max_tokens
```

candidate 全部完成后，根据实际 terminal candidate 数量缩小 reservation；完成
reward/resample 后，当前 step 可以原子地把 claim 转交给同一 job 的下一 step，避免释放
后被其他短上下文 job 插队而引起长前缀重新 prefill。控制器还具有：

- 最近 32 个 `step_index=0` 样本的滚动预算校准，避免短 warmup 固定整轮预算；
- `max_active_steps` 安全上限；
- 结构化 claim ID、实时 active/peak token 统计和 transition 计数；
- `fifo`、`largest_fit`、`balanced_fit` 三种可消融队列策略。

默认策略仍为 `fifo`。`largest_fit` 与 `balanced_fit` 只用于实验，不作为生产默认值。

## 3. 固定环境

- 模型：`Qwen3-Coder-30B-A3B-Instruct` BF16
- Runtime：vLLM-Ascend v0.18 MRV1
- 硬件：服务器 `159.138.5.111` 的物理 NPU 0/5，TP2
- Engine：MNS 256、MBT 32768、memory 0.90、partial prefill `(1,1)`
- 请求：32 jobs、32 workers
- P1：`C8/R3/B128/L512`，ignore EOS，确保各方案工作完全一致
- P0：`C15/R3/B128/L512`，真实 EOS，同 namespace、同请求 seed

所有运行均在启动前检查所选设备无他人进程；本轮容器现已退出。

## 4. P1 固定工作量结果

各方案均完成 3328 个 engine request、720896 个 generated token、1344808 个
forward-token-slot（LPT 1 秒因调度顺序造成 1664 个额外 prefill token，另行注明）。

| 策略 | jobs/s | 相对静态 | Job mean | 相对静态 | Job P95 | FTS/s | preempt |
|---|---:|---:|---:|---:|---:|---:|---:|
| 静态 `step_fifo + cap16` | 0.053449 | baseline | 400.0 s | baseline | 596.5 s | 2246.2 | 0 |
| 动态 peak-token FIFO | 0.053506 | +0.11% | 412.0 s | +3.01% | 597.9 s | 2248.6 | 0 |
| `largest_fit`，coalesce 1 s | 0.053934 | +0.91% | 494.3 s | +23.56% | 593.2 s | 2269.4 | 0 |
| `balanced_fit`，coalesce 1 s | 0.053111 | -0.63% | 467.6 s | +16.88% | 598.8 s | 2232.0 | 0 |

结论：动态 FIFO 已能在完全相同工作量下匹配静态最优 cap16，吞吐差异只有 0.11%。
LPT 虽将 makespan 缩短 0.90%，却令平均 Job 延迟恶化 23.56%；长短交替也恶化
16.88%。将极端不同长度的上下文主动混入同一 decode batch，会破坏 attention batch
shape，说明“更满”或“最长优先”并不等于更高效。

## 5. P0 高 fanout 迁移结果

| 指标 | 静态 cap16 | 动态 peak-token FIFO | 变化 |
|---|---:|---:|---:|
| jobs/s | 0.170260 | 0.179503 | +5.43% |
| wall time | 375.90 s | 356.54 s | -5.15% |
| Job mean | 139.63 s | 137.12 s | -1.80% |
| Job P95 | 240.35 s | 242.81 s | +1.02% |
| Job P99 | 301.00 s | 264.50 s | -12.13% |
| generated tokens/s | 606.75 | 631.08 | +4.01% |
| FTS/s | 3025.60 | 3178.76 | +5.06% |
| max in-flight | 203 | 214 | +5.42% |
| preemption | 0 | 0 | unchanged |

动态策略的 7 次跨 step continuation 全部原子完成，没有 timeout。它把滚动 token
budget 从约 418K 校准到 543K，实际 active 峰值约 537K。与早期固定 5 秒 continuation
在 P0 上 `FTS/s -12.24%` 的结果相比，显式 transition 和滚动预算消除了 C/R 敏感的
grace-time 问题。

P0 真实 EOS 下，静态与动态虽然使用相同顶层 request ID 和 seed，但只有 25/64 个最终
输出完全一致，20/64 个 completion token 数一致。原生 sampler 的结果会随 batch/admission
顺序变化，因此不能只用 jobs/s 下结论。本轮动态运行的总 FTS 少 0.35%，但 FTS/s 仍提高
5.06%，说明主要收益不是靠生成更少工作得到；后续严格算法等价 A/B 仍应先修复或绕过
per-request sampler 非确定性。

## 6. 已证伪的动态方向

1. EngineCore peak-budget 直接淘汰 tree：P1 prefill 增加 116%，jobs/s 下降 13.9%。
2. 外层 peak-budget 但不保留 continuation：P1 prefill 增加 188%，jobs/s 下降 18.9%。
3. 预算由短 warmup 一次性固定：P1 虽保持正确 prefill，仍明显欠填；滚动校准才恢复。
4. `largest_fit`：略缩短 makespan，但平均延迟恶化 23.6%，不适合作为在线默认策略。
5. `balanced_fit`：吞吐与平均延迟同时回退，长短请求混排不是有效 locality 策略。

这些反例表明下一步不应继续堆通用排队启发式。CIS 的收益来自保存同一 job 的跨 step
前缀局部性，同时根据 fanout/剩余 token 放行适量独立 tree，而不是简单改变 FCFS 顺序。

## 7. 当前定位与后续方向

当前动态版本属于 **CIS-aware 外层 token-budget admission**：vLLM 提供 per-request
priority 和 continuous batching，本实现增加 CIS step 成本模型、动态容量、结构化生命周期
与原子 continuation。它比固定 cap 更能跨 P0/P1 迁移，但仍保留 `active_step_limit=16`
作为初始预算倍率，以及 `max_active_steps=32` 作为安全边界，因此还不是完全免调参。

下一版若继续提高技术含量，应按以下顺序推进：

1. 从 runtime 读取 KV capacity、MNS 和实际 block size，直接推导 token budget，取消 cap16
   这一名义容量标尺；
2. 加入基于实际 preemption、decode batch 与 KV waterline 的慢速闭环，只调整 budget，不改变
   FIFO/locality 顺序；
3. 修复 request-local sampler 确定性，使调度 A/B 在相同 seed 下生成完全相同的分支；
4. 若外层控制仍受限，再将结构化 tree state 下沉到 EngineCore，按 ready branch、剩余 token、
   barrier urgency 和 KV 所有权调度。

## 8. 原始数据与校验

原始目录：

```text
/data/disk/wangzili/cis-peak-budget-20260914
```

关键 `benchmark.json` SHA256：

```text
602bbbe969171bab70c4020f78b6ae55bb562873eec440bf059fbcf8f349baea  fixed-work/p1-static-cap16-05/benchmark.json
66940c6a9bac997af84175e4a059cf84481a470c1fc505cb5af9c811c0411fb9  fixed-work/p1-outer-rolling-05/benchmark.json
d550491fd7121b607360a2bdbddd93190062e40bb788b54305f08eded7a8d387  fixed-work/p1-outer-lpt1-05/benchmark.json
6dd347e1b9ccb7247083ce7648f1969764eb4532c7e11037bf96b900ff765055  fixed-work/p1-outer-balanced-05/benchmark.json
947be328b2a4b3a5229dddf25b28b1cabb2d55e103be1a3aec78987f2bbede82  real-eos/p0-static-confirm-05/benchmark.json
81bb36f3b15868952d86fd43dd935fd7b3d5c032a082e0e4b74c7cac6250db7e  real-eos/p0-outer-confirm-05/benchmark.json
```

## 9. Runtime-KV 跨上下文扩展

后续检查发现，滚动方案的预算是 `cap16 * 最近 step0 平均 reservation`。当整个数据集
从短上下文切换为长上下文时，分子与预算会同步变大，活跃 step 数仍接近16。因此它只能
处理同一 workload 内部的长短异构，不能真正适应整体 context shift。

当前分支已增加 `runtime_kv_budget` 模式：

- 通过一个只读 EngineCore utility 获取每 rank 的 `num_gpu_blocks`、KV `block_size` 和
  `token_capacity`；
- 使用 `token_capacity * active_step_kv_capacity_fraction` 作为固定硬件预算，默认安全
  系数为0.9；
- shared trunk、candidate suffix 和每条 rollout tail 分别按真实 KV block 向上取整；
- 单个 step 的保守估算超过预算时允许独占前进，避免永久等待；
- terminal candidate resize、FIFO locality 和同 job 原子 continuation 继续保留。

已从同一公开 workload 确定性抽取三个32请求分层，未复制 prompt：

| 分层 | 范围 | prompt 中位数 | prompt 最小/最大 | 0.9 runtime budget 预计初始 step 数 |
|---|---:|---:|---:|---:|
| short | `<4K` | 2813.5 | 1518 / 4086 | 32 |
| medium | `8K-16K` | 11530.5 | 8703 / 16055 | 20 |
| long | `16K-32K` | 20376.5 | 16500 / 31934 | 13 |

以上 step 数按既有 TP2 日志中的 508928-token KV capacity、0.9安全系数和 P1
`C8/R3/B128/L512` 计算。它显示新模式会随整体上下文变长自动从32降到20、13，而滚动
reference 模式基本仍保持16。

正式 NPU 验证矩阵为三种分层分别比较 static cap16、rolling peak-token 和
runtime-KV 0.9；再在相同 P0 条件补跑 candidate-subtree bundle。当前两台服务器都只有
单张空闲卡，没有可用 TP2 组合。单卡4B capability 尝试在模型启动前被 Ascend DCMI/
categorical 的全机设备枚举失败拦截，并非 admission 代码错误。代码、workload 和运行入口
均已准备，不能在没有同条件 NPU 数据前把上述预计并发写成性能收益。

## 10. 设备预留审计与可恢复矩阵

后续重查发现，服务器 A 的物理卡1/4虽然没有 `npu-smi` 计算进程，但分别被运行中的
`orthrus-torchprof-npu1` 和 `orthrus-v023-npu4` 容器映射。实际服务与只读
`torch_npu` 探针均无法枚举设备。因此“无 NPU PID”不足以证明共享服务器上的卡可用。

远程 runner 现同时检查：

1. `npu-smi` 中所选卡是否存在计算进程；
2. 运行中 Docker 容器是否映射所选 `/dev/davinciN`。

默认发现任一容器预留便拒绝启动，并报告卡号和容器名。只有得到容器所有者确认后才可
显式设置 `CIS_ALLOW_DOCKER_RESERVED_DEVICES=1`；本轮没有使用该覆盖选项，也没有停止
任何他人容器。

新增可断点续跑入口：

```text
experiments/swebench/run_cis_context_transfer_remote.sh
```

`context` 模式运行 `short/medium/long × static/rolling/runtime-KV` 共9组；`subtree`
模式在同一 P0 workload 上运行 static、runtime-KV 与 `step-subtree-8` 三组。已有完整
`benchmark.json` 的运行会跳过，所有运行结束后由现有 summarizer 生成统一 JSON。
