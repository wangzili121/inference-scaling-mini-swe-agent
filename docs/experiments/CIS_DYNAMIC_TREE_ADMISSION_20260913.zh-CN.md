# CIS EngineCore 动态树 Admission 实验

## 目标

本轮不再调整普通 vLLM 参数，也不修改 Conditional IS 的 C/R/B、采样或 reward。
它只验证一个问题：能否把已有的固定 Step cap 升级为 EngineCore 内的 CIS 树状态与
反馈式 admission，在保留设备利用率的同时缩短 step barrier，并避免 KV preemption。

原始 vLLM 仍负责 continuous batching、MNS/MBT、KV 分配和 preemption；新增
`CISTreeScheduler` 只决定哪些完整 CIS step 可以进入普通 waiting/running queue。

## 实现

每条底层请求携带结构化元数据：

```text
job_id / step_index / node_type
candidate_index / rollout_index
candidate_count / expected_rollouts
```

Scheduler 为每个 `job_id + step_index` 维护候选、rollout、活跃请求和延迟请求集合。
当前规则是：

1. 初始窗口为 `ceil(MNS / (C * R))`；
2. 窗口外的新 step 保留在 EngineCore 的结构化 deferred queue；
3. 已激活 step 的 rollout 总能进入 vLLM，避免 child 被自己的 admission 阻塞；
4. 只有完成过真实 step、running 低于目标、普通 waiting 为空且 KV 低于水位时，
   窗口才加一；
5. 出现 preemption 或 KV 超过高水位时，窗口减一，但不低于初始窗口；
6. 窗口内仍使用已有 `step_fifo` priority，底层 sequence 仍可跨树 continuous batch。

因此它不是把一整棵树强制塞进一个 batch，也不是把 vLLM scheduler 整体重写一遍。

## 固定条件

- 模型：`Qwen3-Coder-30B-A3B-Instruct` BF16；
- runtime：vLLM-Ascend v0.18 MRV1，双卡 TP2 单 instance；
- NPU：服务器 `159.138.5.111` 的物理卡 1/4，启动前为空卡；
- engine：MNS 256、MBT 32768、memory 0.90、partial prefill `(1,1)`；
- 已启用 APC、chunked prefill、native categorical sampler、FULL_DECODE_ONLY；
- workload：64 个固定公开 SWE-agent 调用，workers 32，一次性释放；
- P0：`C15/R3/B128/L512`；P1：`C8/R3/B128/L512`；
- sequence-logprob、`T=1/top_p=1/top_k=None`。

P0 flat/priority/cap 是 2026-09-13 同 namespace 两轮的等权平均；P1 既有对照来自
同模型、runtime、engine、64-request workload 的历史同机实验。随机采样会改变实际
生成 token 数，因此 jobs/s 必须与 generated-token/s、FTS/s 和延迟一起解释。

## P0 结果

| 策略 | jobs/s | generated tok/s | FTS/s | Job mean/P95 | Barrier mean/P95 | preempt |
|---|---:|---:|---:|---:|---:|---:|
| flat vLLM | 0.1539 | 552.7 | 3904 | 164.7/327.6 s | 69.4/173.0 s | 0.5 |
| strict step priority | 0.1824 | 602.9 | 3428 | 147.2/276.1 s | 58.6/153.6 s | 10.0 |
| hard cap6 + priority | 0.1667 | 553.4 | 2901 | 139.1/230.4 s | **21.2/46.4 s** | **0** |
| hard cap16 + priority | 0.1944 | **630.7** | 3382 | 134.7/234.0 s | 40.5/112.8 s | **0** |
| Cohort6 | 0.1677 | 606.9 | 3119 | 145.7/282.7 s | 59.0/152.4 s | **0** |
| **EngineCore dynamic 6->16** | **0.1988** | **612.1** | **3432** | **130.0/227.0 s** | 38.5/94.9 s | **0** |

相对 flat，dynamic 的 jobs/s `+29.1%`、generated-token/s `+10.8%`、Job P95
`-30.7%`、Barrier P95 `-45.2%`。但 FTS/s `-12.1%`，且本轮总生成 token 更少，
所以不能把 `+29.1% jobs/s` 全部视为 engine efficiency 收益。

相对 strict priority，dynamic 的 FTS/s 基本持平（`+0.1%`），generated-token/s
`+1.5%`，Job P95 `-17.8%`，Barrier P95 `-38.2%`，并将 preemption 从平均 10
降为 0。这是当前最可靠的 P0 结论：它主要改善完成顺序、尾延迟和 KV 压力，在保持
priority 模式单位 forward 工作吞吐的同时，形成了新的 balanced Pareto 点。

相对 hard cap6，dynamic 的 FTS/s `+18.3%`、generated-token/s `+10.6%`，Job P95
`-1.5%`；代价是 Barrier P95 从 46.4s 上升到 94.9s。它不是纯 tail-first 策略。

最关键的固定 cap16 消融表明：dynamic 相对 cap16 只有 jobs/s `+2.2%`、FTS/s
`+1.5%`、Job P95 `-3.0%`，generated-token/s 反而 `-3.0%`；较清晰的增量是
Barrier P95 `-15.9%`。因此固定 cap16 已复现 dynamic 的大部分 P0 收益，当前动态
反馈只能视为有限增量，不能宣称已经得到优于简单外层控制的成熟 scheduler。

Scheduler trace 显示：

- 窗口从 6 增长到 16；
- scheduler 内活跃 step 峰值 16；
- 延迟 step 峰值 30、延迟 candidate request 峰值 450；
- running request 峰值 217；
- KV usage 峰值 68.98%；
- 全程零 preemption。

`engine_queue P95=102.4s` 高于 flat，是因为 deferred candidate 从首次到达就记录
QUEUED。它表示等待被搬到了可控的树级队列，不表示 NPU 更空闲；判断用户体验应看
Job/Step/Burst latency。

## P1 结果

| 策略 | jobs/s | generated tok/s | FTS/s | Job mean/P95 | Barrier mean/P95 | preempt |
|---|---:|---:|---:|---:|---:|---:|
| flat vLLM | 0.2352 | 475.0 | 4604 | 110.7/223.6 s | 45.5/114.5 s | 28 |
| hard cap16 + priority | **0.2752** | **506.0** | **4150** | 97.5/**173.2 s** | **29.7/75.4 s** | **0** |
| hard cap22 + priority | 0.2687 | 468.2 | 4019 | **93.1**/186.3 s | 32.0/97.3 s | **0** |
| EngineCore dynamic 11->22 | 0.2583 | 459.6 | 4058 | 95.1/183.3 s | 32.7/88.1 s | **0** |

Dynamic 相对 flat 改善 jobs/s `+9.8%`、Job P95 `-18.0%`、Barrier P95 `-23.0%`
并消除 preemption，但 generated-token/s `-3.3%`、FTS/s `-11.9%`。它与固定 cap22
接近，却没有胜过既有 cap16：jobs/s 低 `6.1%`、Job P95 高 `5.8%`、Barrier P95
高 `16.9%`。

P1 trace 的窗口从 11 增长到 22，KV 峰值 88.28%、running 峰值 178，仍未出现
preemption。它说明只用瞬时 running 数量与 KV 水位会把 candidate 阶段的短暂欠载
误判为需要继续扩窗；scheduler 看不到尚未进入 engine 的下游 rollout 工作，也没有
用 step completion rate 判断扩窗是否真正有益。

## 语义边界

P0/P1 分别有 22/31 个 step 的全部 candidate 直接 EOS，因此算法没有提交 rollout。
当前原型通过 5 秒 idle grace 回收这类 step。trace 已确认这些 step 在回收后没有
重新出现 rollout，故本轮没有发生错误放行；但正式实现不能依赖超时猜测，必须由
算法层显式发送 `step_closed`，或在 EngineCore 用准确 finish reason 判定 terminal。

## 剩余分支预算实验

为避免第一版窗口随运行时间永久增大，又测试了更结构化的 fractional work policy。
每个活跃 step 初始占 `1.0` 预算；进入 rollout 后，其占用按
`尚未完成 rollout / (C * R)` 连续下降，累计释放一个完整预算时才放入下一棵树。
Rollout metadata 额外携带该 step 的准确实际 rollout 总数，因此 terminal candidate
不会被误计为潜在 child。

自动初始预算 6 的 16-request smoke 已通过，但 FTS/s 相对同规模 Cohort6 低 2.7%，
Barrier P95 高约 9%。64-request 运行中，running 峰值仅 134/256、KV 峰值 47.4%，
确认预算公式明显 underfill 后按 successive-halving 提前终止，只保留 partial trace。

随后使用 general 调优得到的 cap16 作为共同基础预算，做两轮正式 64-request A/B：

| 策略 | jobs/s | generated tok/s | FTS/s | Job mean/P95 | Barrier mean/P95 | preempt | KV peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed cap16 | **0.1944** | **630.7** | 3382 | **134.7/234.0 s** | **40.5/112.8 s** | **0** | - |
| fractional cap16 | 0.1884 | 611.9 | **3493** | 145.6/294.3 s | 60.3/150.2 s | 2 | 99.27% |
| fractional + queue/KV gate | 0.1793 | 595.1 | 3367 | 142.9/287.0 s | 57.1/153.0 s | **0** | 97.66% |

无门控 fractional 相对 fixed cap16：jobs/s `-3.1%`、Job P95 `+25.8%`、Barrier P95
`+33.1%`，虽有 FTS/s `+3.3%`，但以 2 次 preemption 和严重尾延迟为代价，判负。

安全门控只允许在普通 vLLM waiting queue 已空、当前 KV 低于 80%时借槽。它消除了
preemption，但 jobs/s `-7.8%`、FTS/s `-0.5%`、Job P95 `+22.7%`、Barrier P95
`+35.6%`，仍明确判负。当前 KV 低并不表示新 candidate 及其未来 rollout 的增量 KV
足够小；一次放入完整 candidate group 后，KV 仍可继续涨到 97.66%。

## 当前判断

1. 这版 EngineCore scheduler 在高 fanout P0 上形成弱 Pareto 点，但固定 cap16 已
   复现大部分收益。结构化树级 admission 值得继续，当前规则本身还不够强。
2. 它没有自动找到 P1 的既有最优 cap16，因此当前反馈规则不能称为通用最优调度器。
3. 两种 fractional policy 均已证伪。若未来再做动态 admission，必须先拥有可校准的
   增量 KV/remaining-token 预测和显式 `step_closed`，不能继续叠加 Host 水位规则。
4. P2 `C4/R2` 在 workers=32 下初始窗口就是 32，不会延迟任何 step；运行同一策略
   只会退化为已有 step priority，因此本轮不重复占用 NPU。
5. 当前产品基线仍应保留 P0 fixed cap16、P1 fixed cap16、P2 elastic。实验性的
   EngineCore policy 保留在独立 variant 和分支中，不能默认启用。
6. Host 动态调度扩展到此停止。后续更有价值的 scheduler 工作应建立在准确的树状态、
   增量 KV 模型或 EngineCore 原生 fork 上，并先提出能超过 fixed cap 的可证伪假设。

## 原始数据

```text
/data/disk/wangzili/cis-dynamic-scheduler-20260913/full/tree-adaptive-p0
/data/disk/wangzili/cis-dynamic-scheduler-20260913/full/tree-adaptive-p1
/data/disk/wangzili/cis-dynamic-scheduler-20260913/full/p0-cap16
/data/disk/wangzili/cis-dynamic-scheduler-20260913/smoke/tree-fractional-p0-v1
/data/disk/wangzili/cis-dynamic-scheduler-20260913/full/tree-fractional-p0-v1
/data/disk/wangzili/cis-dynamic-scheduler-20260913/full/tree-fractional-p0-cap16
/data/disk/wangzili/cis-dynamic-scheduler-20260913/full/tree-fractional-safe-p0-cap16
```

每个 dynamic 目录包含 `benchmark.json`、request/algorithm trace，以及原样
`cis-scheduler-trace.jsonl`。最终派生摘要将提交到：

```text
docs/experiments/data/cis_dynamic_tree_admission_20260913.json
```

摘要 SHA256：

```text
f9278b1cf1c84a37226b753736961f711be8342e04e1938934c735982c58543e
```
