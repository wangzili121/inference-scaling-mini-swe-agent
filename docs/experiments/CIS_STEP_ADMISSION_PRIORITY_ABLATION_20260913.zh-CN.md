# Conditional IS Step Admission 与 Priority 消融

## 结论先行

本轮回答的是：此前 `Step cap + step_fifo` 的收益，是否只是把外层并发从 32 降到
6，或者只是打开了 vLLM 已有功能。

结论如下：

1. 纯外层限流不能复现吞吐收益。`workers=6` 相对扁平 `workers=32` 的 jobs/s
   为 `-2.3%`，wall time 为 `+2.4%`。
2. 只开 step cap 的收益较小。jobs/s 为 `+2.7%`，preemption 从 1 降到 0，
   barrier P95 为 `-64.3%`，但 burst P95 为 `+10.4%`。
3. 只开 `step_fifo` 的两轮结果都明显优于扁平 vLLM。jobs/s 分别为
   `+13.0%/+24.1%`，generated tokens/s 为 `+7.7%/+10.4%`。代价是 engine
   queue P95 和 preemption 上升。
4. `cap6 + step_fifo` 的两轮 jobs/s 为 `+5.5%/+11.1%`；两轮均保持 0
   preemption，Job P95 为 `-24.9%/-34.1%`，barrier P95 为
   `-75.3%/-71.0%`。它是当前更稳健的尾延迟方案，但不是吞吐最高方案。
5. vLLM 自带的是 per-request priority scheduler，不自带 CIS step 分组。我们新增
   的是 `job_id + step_id` 到 priority 的映射，以及从 candidate 开始到 rollout、
   reward、resample 完成才释放的 step admission 生命周期。因此它不是“漏开了一个
   vLLM 配置”，但当前也还不是完整的 tree-aware scheduler。

## 固定条件

- 服务器：`159.138.5.111`，物理 NPU 1/4；启动前两张卡均无进程，实验结束后释放。
- 模型：`Qwen3-Coder-30B-A3B-Instruct` BF16。
- Runtime：vLLM-Ascend v0.18 MRV1，双卡 TP2，单 instance。
- 算法：ordinary Conditional IS P0，`C15/R3/B128/L512`，sequence-logprob，
  `T=1/top_p=1/top_k=None`。
- Engine：`MNS=256`、`MBT=32768`、memory `0.90`、partial prefill `(1,1)`。
- 已启用 persistent AsyncLLM、continuous batching、APC、chunked prefill、
  native categorical sampler、request-local seed 和
  `FULL_DECODE_ONLY + Npugraph_ex`。
- Workload：同一批 64 个公开 SWE-agent 调用；除 B 外均使用 32 个客户端 worker。
- 所有正式实验成功率 100%，无 transport retry。

## 实验矩阵

| ID | workers | Step cap | Priority | 要回答的问题 |
|---|---:|---:|---|---|
| A | 32 | 无 | vLLM FCFS | 扁平 baseline |
| B | 6 | 无 | vLLM FCFS | 纯外层限流能否复现收益 |
| C | 32 | 6 | vLLM FCFS | step admission 的独立贡献 |
| D | 32 | 无 | `step_fifo` | CIS priority 映射的独立贡献 |
| E | 32 | 6 | `step_fifo` | 旧完整方案 |

`step_fifo` 仍调用 vLLM 的通用 priority scheduler，但 priority 值由本仓库根据
CIS request ID 中的 `job_id + step_id` 生成。同一个 step 的 candidate 和 rollout
共享 priority，较早 step 优先。

## 第一轮完整消融

括号中的百分比均相对 A。`Burst P95` 以本轮最早 worker start 作为 64 个请求的
共同近似到达时刻，包含客户端 worker queue；它比每个线程开始发 HTTP 后才计时的
`Job P95` 更适合比较 W32 和 W6。

| 方案 | jobs/s | generated tok/s | Wall | Job P95 | Burst P95 | Barrier P95 | Engine queue P95 | Preempt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A flat W32 | 0.1553 | 555.0 | 412.2 s | 311.6 s | 357.4 s | 176.0 s | 56.8 s | 1 |
| B flat W6 | 0.1517 (-2.3%) | 546.1 (-1.6%) | 422.0 s (+2.4%) | 90.7 s* | 353.7 s (-1.0%) | 47.8 s (-72.8%) | 0.0004 s | 0 |
| C cap6 W32 | 0.1594 (+2.7%) | 546.5 (-1.5%) | 401.5 s (-2.6%) | 316.5 s (+1.6%) | 394.6 s (+10.4%) | 62.9 s (-64.3%) | 0.0005 s | 0 |
| D priority W32 | **0.1755 (+13.0%)** | **598.0 (+7.7%)** | **364.7 s (-11.5%)** | 277.4 s (-11.0%) | **343.3 s (-3.9%)** | 155.3 s (-11.8%) | 65.1 s (+14.6%) | 16 |
| E cap6+priority W32 | 0.1638 (+5.5%) | 544.8 (-1.8%) | 390.7 s (-5.2%) | **234.1 s (-24.9%)** | 345.9 s (-3.2%) | **43.5 s (-75.3%)** | **0.0005 s** | **0** |

`*` B 的 90.7 秒不包含尚未获得客户端 worker 的请求，不可直接与 W32 的 Job P95
比较。按共同 burst 起点重算后，B 的 P95 仅改善 1.0%，wall time 反而增加 2.4%。

## A/D/E 确认轮

首轮不同策略的实际生成 token 数相差约 5% 到 8%。为避免把较短输出误认为调度
收益，只复跑最关键的 A/D/E，不重复 B/C。

| 方案 | jobs/s | generated tok/s | Wall | Job P95 | Burst P95 | Barrier P95 | Engine queue P95 | Preempt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A flat W32 | 0.1526 | 550.4 | 419.4 s | 343.7 s | 376.6 s | 170.0 s | 50.8 s | 0 |
| D priority W32 | **0.1894 (+24.1%)** | **607.8 (+10.4%)** | **337.9 s (-19.4%)** | 274.8 s (-20.0%) | **324.1 s (-13.9%)** | 152.0 s (-10.6%) | 68.5 s (+35.0%) | 4 |
| E cap6+priority W32 | 0.1695 (+11.1%) | 562.1 (+2.1%) | 377.5 s (-10.0%) | **226.6 s (-34.1%)** | 356.4 s (-5.4%) | **49.3 s (-71.0%)** | **0.0005 s** | **0** |

两轮按 run 等权平均：

| 方案 | jobs/s | generated tok/s | Job P95 | Burst mean / P95 | Barrier P95 | Preempt/run |
|---|---:|---:|---:|---:|---:|---:|
| A flat W32 | 0.1539 | 552.7 | 327.6 s | 234.9 / 367.0 s | 173.0 s | 0.5 |
| D priority W32 | **0.1824 (+18.5%)** | **602.9 (+9.1%)** | 276.1 s (-15.7%) | 212.7 (-9.4%) / **333.7 s (-9.1%)** | 153.6 s (-11.2%) | 10.0 |
| E cap6+priority W32 | 0.1667 (+8.3%) | 553.4 (+0.1%) | **230.4 s (-29.7%)** | **179.5 (-23.6%)** / 351.1 s (-4.3%) | **46.4 s (-73.2%)** | **0.0** |

P95 的平均仅用于快速展示两轮趋势；正式统计若需要置信区间，应合并相同策略的原始
样本后 bootstrap，而不是把两个 P95 当成一个最终总体 P95。

## 归因

### 纯外层限流不是等价替代

B 把 active step peak 从 32 降到 6，engine queue 和 preemption 几乎消失，但设备
工作供给不足，jobs/s 和 generated tokens/s 都下降。普通 API worker pool 只能限制
整个 job 数量，不能在一个 job 完成当前 step 后，让其下一 step 与其他 job 的 ready
step 按 CIS 结构重新竞争。

C 保留 32 个外部 job，只在算法内部限制 6 个 active step。它比 B 的 jobs/s 高
5.1%，证明 step 生命周期 admission 比固定 job concurrency 更细，但单独 cap 的吞吐
收益仍然很小，且 burst P95 回退。

### 主要吞吐收益来自 CIS priority 映射

D 没有 hard cap，只给同一 `job_id + step_id` 的 candidate/rollout 相同 priority。
两轮 generated tokens/s 均提升，说明其收益不能只由输出变短解释。它更快完成旧 step，
提高 prefix locality 和 useful decode 产出，但仍允许多达 414/456 个 in-flight request，
因此 queue P95 和 preemption 较高。

### Hard cap 是尾延迟与稳定性控制器

E 把 D 的高吞吐换成更低的 queue、barrier 和 tail。两轮均为 0 preemption，barrier
P95 稳定下降约 71% 到 75%。但 generated tokens/s 两轮平均几乎与 baseline 相同，
所以不能继续把 E 描述为“吞吐一定提升 15%”。更准确的定位是：

- 吞吐模式：D，step priority only；
- 尾延迟/无 preemption 模式：E，cap6 + step priority；
- 后续动态 scheduler 的目标：在 D 的吞吐与 E 的 tail 之间自动选择，而不是固定 cap6。

## 与 vLLM 及其他框架能力的边界

vLLM v0.18 已提供 FCFS/priority、waiting/running queue、MNS/MBT、continuous
batching 和 KV preemption。这些都是本实验依赖的通用机制。它不知道 request 属于
哪个 CIS job/step，也不知道 candidate、rollout 和 barrier 关系。

本仓库新增的部分是：

- `StepAdmissionController`：以完整 CIS step 生命周期持有和释放 admission slot；
- request ID 中的 CIS 元数据；
- `job_id + step_id -> monotonic priority` 映射；
- 算法 trace 中的 step、candidate、rollout、barrier 和 queue 归因。

因此现有实现属于“基于 vLLM 通用原语的 CIS-aware policy”，创新和工程价值高于普通
参数开关，但低于真正把 tree/subtree、ready branch、剩余工作量和 KV 所有权放进
EngineCore 的新 scheduler。其他框架即使有 generic priority、router 或 concurrency
limit，也必须实现等价的 CIS 元数据和策略才能得到本实验行为，不能直接自动获得。

## 限制

- `T=1` 且 native categorical 在不同 batch shape 下没有逐 token 确定性。两轮 D
  的生成 token 数相对 A 少 4.7%/11.0%，E 少 7.0%/8.1%。因此同时报告 jobs/s、
  generated tokens/s、wall time 和延迟，不能只挑 jobs/s。
- 本轮只覆盖 P0、6 到 8K 为主的固定 workload、双卡 TP2 和单 instance。
- 本轮是无 NPU profiler 的性能消融；保留了 benchmark、algorithm trace、request
  trace 和 service log，但没有采集 Ascend kernel trace。
- B/C 只跑一轮。它们已经足以回答外层限流问题，但不用于宣称小于 3% 的稳定收益。

## 原始数据与复现

服务器原始目录：

```text
/data/disk/wangzili/cis-step-ablation-20260913/full
/data/disk/wangzili/cis-step-ablation-20260913/confirm
```

每个 variant 保留 `benchmark.json`、`artifact-manifest.json`、algorithm/request trace、
service log、环境和完整启动命令。两个目录合计约 96 MiB，未提交 Git。

Git 中的派生数据：

```text
docs/experiments/data/cis_step_admission_priority_ablation_20260913.json
docs/experiments/data/cis_step_admission_priority_ablation_20260913.SHA256SUMS
docs/experiments/data/cis_step_admission_priority_confirm_20260913.json
docs/experiments/data/cis_step_admission_priority_confirm_20260913.SHA256SUMS
```

入口脚本：

```text
experiments/swebench/run_cis_step_ablation_remote.sh
experiments/swebench/run_cis_forest_ab_remote.sh
experiments/swebench/summarize_cis_step_ab.py
```
