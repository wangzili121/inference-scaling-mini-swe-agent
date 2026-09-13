# CIS Work-Conserving Cohort Scheduler 实验

## 目标

此前消融已经证明 `job_id + step_id -> priority` 是主要吞吐收益来源，hard Step
cap 则主要降低 barrier tail 和 KV preemption。不过二者分别偏向吞吐和尾延迟，且
固定 cap 需要随 C/R 手工重调。

本轮验证一个更小、更明确的中间设计：**work-conserving step cohort**。

- 按 step 首次进入 runtime 的顺序划分 cohort；
- 同一 cohort 的 candidate 和 rollout 共享 priority；
- 老 cohort 优先；
- 不设置 host hard cap，老 cohort 暂时填不满 MNS/MBT 时，vLLM 可从下一 cohort
  自动回填，因此仍是 work-conserving；
- 默认 cohort 大小为 `ceil(MNS / (C * R))`。P0 `C15/R3/MNS256` 自动得到 6。

它与 `cap6 + step_fifo` 的区别是后者禁止第 7 个 step 展开，前者只建立优先级边界，
不禁止后续 work 填补空闲设备。

## 固定条件

- 模型：`Qwen3-Coder-30B-A3B-Instruct` BF16；
- runtime：vLLM-Ascend v0.18 MRV1，双卡 TP2，单 instance；
- engine：MNS 256、MBT 32768、memory 0.90、partial prefill `(1,1)`；
- 已启用 APC、chunked prefill、native categorical sampler 和 FULL_DECODE_ONLY；
- workload：64 个固定公开 SWE-agent 调用，workers 32，一次性释放；
- 算法：P0 `C15/R3/B128/L512`、sequence-logprob、T=1；
- 服务器：`159.138.5.111`，物理 NPU 1/4，启动前为空卡。

## P0 结果

本表中的 A/D/E 是 2026-09-13 两轮消融的 run 等权平均；Cohort6 是本轮首个正式
64-request 结果。百分比相对 A。随机采样会改变实际生成 token 数，因此 jobs/s 与
generated-token/s 必须同时看。

| 策略 | jobs/s | generated tok/s | Job P95 | Burst P95 | Barrier P95 | preempt/run |
|---|---:|---:|---:|---:|---:|---:|
| A flat vLLM | 0.1539 | 552.7 | 327.6 s | 367.0 s | 173.0 s | 0.5 |
| D strict step priority | 0.1824 | 602.9 | 276.1 s | 333.7 s | 153.6 s | 10.0 |
| E cap6 + step priority | 0.1667 | 553.4 | 230.4 s | 351.1 s | 46.4 s | 0.0 |
| Cohort6 | **0.1677 (+9.0%)** | **606.9 (+9.8%)** | 282.7 s (-13.7%) | 344.6 s (-6.1%) | 152.4 s (-11.9%) | **0.0** |

首轮结论：

1. Cohort6 的 generated-token/s 与 strict step priority 基本持平，同时把 preemption
   从平均 10 次降到 0，说明分组 priority 可以保留主要有效工作吞吐，并减少 KV
   过载。
2. Cohort6 的 Job/Burst P95 优于 flat vLLM，但 Barrier P95 没有达到 hard cap6 的
   水平。它应定位为 balanced policy，而不是替代 tail-first policy。
3. 最大 in-flight 为 418，低于 strict priority 的 414/456 区间上端，但明显高于
   hard cap。它确实在做弹性回填，不是隐式复现 hard cap。
4. 这还是利用 vLLM 通用 priority 的 CIS-aware policy，不应描述成完整 EngineCore
   tree scheduler。其价值在于给下一版动态 scheduler 提供了更好的、无需固定 cap
   的基线。

## 实现变化

- 新增 `step_cohort` priority policy；
- 新增 `request_priority_cohort_size="auto"`，按 MNS 与 C/R 自动计算；
- 新增结构化 `CISRequestMetadata`，逐步替代从 request ID 正则解析 job/step/node；
- request trace 直接记录 job、step、candidate、rollout 和 expected rollout 数；
- 保留 request-ID fallback，便于旧 workload 和旧 trace 兼容。

## 原始数据

```text
/data/disk/wangzili/cis-dynamic-scheduler-20260913/smoke/cohort6
/data/disk/wangzili/cis-dynamic-scheduler-20260913/full/cohort6
```

Git 中的 P0 派生摘要：

```text
docs/experiments/data/cis_step_cohort_p0_20260913.json
```

## P1 迁移结果

P1 `C8/R3` 的自动 cohort size 为 `ceil(256/24)=11`。正式 64-request 结果：

| 策略 | jobs/s | generated tok/s | FTS/s | Job P95 | Barrier P95 | preempt |
|---|---:|---:|---:|---:|---:|---:|
| flat vLLM | 0.2352 | 475.0 | 4604 | 223.6 s | 114.5 s | 28 |
| strict step priority | **0.2661** | 461.0 | **4582** | **206.8 s** | 121.2 s | 18 |
| Cohort11 auto | 0.2350 | 436.9 | 4244 | 230.9 s | 145.5 s | 23 |

Cohort11 在 P1 基本没有 jobs/s 收益，generated-token/s、Job P95 和 Barrier P95
均比 flat 更差，因此自动公式被否定。它重复了历史 hard cap11 的根因：按理论最大
`C * R` 估算每个 step 占用，忽略了终止分支和阶段错开，导致错误的分组尺度。

P2 `C4/R2`、workers32 下 auto cohort size 恰为 32，所有 job 落在同一 priority
cohort，等价于没有 cohort 边界，所以不重复运行。

后续 EngineCore dynamic 与 fractional admission 已继续完成；P0 有弱 Pareto 点，
但固定 cap16 复现了大部分收益，P1 与两种 fractional 规则均未形成新最佳点。详见：

```text
docs/experiments/CIS_DYNAMIC_TREE_ADMISSION_20260913.zh-CN.md
docs/experiments/data/cis_dynamic_tree_admission_20260913.json
```
