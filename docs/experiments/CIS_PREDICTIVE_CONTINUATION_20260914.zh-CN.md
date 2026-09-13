# CIS Predictive Continuation 调度实验

## 结论

本轮目标是回答一个具体问题：能否把固定 `Step cap + step_fifo` 升级为无需依赖
临界 timeout、并且稳定胜过静态最优点的动态调度器。

结论分三层：

1. 已修复第一版 EngineCore tree scheduler 的严重缺陷。准确追踪 candidate finish
   reason 和实际 rollout 数后，scheduler 不再依赖 5 秒 idle timeout 猜测 step 是否
   结束；continuation reservation 也能避免纯广度优先造成跨 step 前缀重算。
2. P1 固定工作量下，5 秒 continuation 相对静态 cap16 有小幅真实收益：jobs/s 和
   FTS/s `+0.73%`、Job mean `-4.67%`、P95 `-1.44%`，且 prefill、生成 token 和
   forward work 完全一致。
3. 该收益不能迁移到 P0。P0 中 53 次 reservation 只有 11 次 claim，42 次过期；
   相对静态 cap16，jobs/s `-15.20%`、FTS/s `-12.24%`。因此当前动态实现不应成为
   默认配置，生产基线仍采用已验证的静态 CIS step 策略。

换言之：动态 scheduler 已从“明显错误”推进到“P1 略胜静态、P0 仍失败”的可研究
原型，但尚未达到产品化标准。继续把 timeout 从 5 秒调成 10/20 秒只是在制造另一项
依赖 C/R 的静态参数，不值得做。

## 固定条件

- 模型：`Qwen3-Coder-30B-A3B-Instruct` BF16；
- runtime：vLLM-Ascend v0.18 MRV1，双卡 TP2 单 instance；
- NPU：服务器 `159.138.5.111` 的物理卡 1/4，运行前为空卡；
- engine：MNS256、MBT32768、memory0.90、partial prefill `(1,1)`；
- 已启用 APC、chunked prefill、native categorical sampler、FULL_DECODE_ONLY；
- P1：`C8/R3/B128/L512`，32 jobs/workers32，窗口16；
- P0：`C15/R3/B128/L512`，64 jobs/workers32，窗口16；
- sequence-logprob、`T=1/top_p=1/top_k=None`。

P1 使用 performance-only fixed-work 模式：忽略 EOS，但不改变 C/R/B/L，使每种策略
严格执行 3328 个 engine request、生成 720896 token。该模式只用于调度因果 A/B，
不用于算法质量结论。P0 保留真实 EOS 行为，用 FTS/s 辅助控制不同输出长度造成的
混淆。

## P1 固定工作量结果

百分比均相对 `step_fifo + outer cap16` 静态基线。

| 策略 | jobs/s | Job mean | Job P95 | FTS/s | prefill tokens | max in-flight | preempt |
|---|---:|---:|---:|---:|---:|---:|---:|
| 静态 step cap16 | 0.052508 | 420.56s | 609.40s | 2206.67 | 627240 | 288 | 0 |
| job_fifo + cap16 | 0.052392 `-0.22%` | 410.78s `-2.33%` | 606.53s `-0.47%` | 2201.79 `-0.22%` | 627240 | 288 | 0 |
| EngineCore，无 continuation | 0.047132 `-10.24%` | 648.72s `+54.25%` | 675.93s `+10.92%` | 3543.44 | 1688232 | 416 | 0 |
| continuation 1s | 0.053424 `+1.74%` | 430.42s `+2.34%` | 598.95s `-1.72%` | 2319.31 | 671656 | 416 | 0 |
| continuation 1.5s | 0.052706 `+0.38%` | 418.87s `-0.40%` | 604.10s `-0.87%` | 2221.93 | 631464 | 416 | 0 |
| work-conserving release | 0.052099 `-0.78%` | 468.50s `+11.40%` | 609.58s `+0.03%` | 2319.94 | 707368 | 416 | 0 |
| continuation 5s | **0.052893 `+0.73%`** | **400.93s `-4.67%`** | **600.64s `-1.44%`** | **2222.84 `+0.73%`** | **627240** | 416 | 0 |

`no continuation` 的 FTS/s 较高不能解释为更快：它把 prefill 从 627240 增加到
1688232，实际总 forward work 增加约 79%，wall time 反而增加 11.4%。这是自定义
scheduler 先执行所有 job 的 step0、再执行 step1 所造成的跨 step prefix 重算。

5 秒版本执行了 96 次 reservation 和 96 次 claim，零 timeout；其 prefill、APC
saved tokens、总 FTS 与静态基线完全相同，因此 `+0.73%` 不是少算 token 或缓存命中
差异。收益仍较小，单轮结果只能证明它形成了新的弱 Pareto 点，不能宣称大幅加速。

## 被证伪的两种直觉

### Job 深度优先

`job_fifo` 让同一 job 的后续 step 继承最初 priority。它确实形成前16个 job完成后再
运行后16个 job的两波执行，但总 work、prefill 与静态基线完全相同，jobs/s 为
`-0.22%`。单纯让旧 job 一路跑到底没有额外局部性收益。

### 看到短暂欠载就释放 continuation

work-conserving 版本在 `running + waiting < 128` 时释放 reservation。96 次可续接中
92 次 claim、4 次主动让位、0 次 timeout，但这4次让位额外产生 80128 prefill token。
四个被让位 job 的下一 step prefix 分别约为 18.8K、22.9K、28.6K 和 8.6K，合计与
额外 prefill 基本一致。80--120 个 runnable request 并不代表 NPU 已经需要换树；
基于请求数量的即时 underfill 判断忽略了重新 prefill 长上下文的代价。

## P0 迁移结果

| 策略 | jobs/s | generated tok/s | FTS/s | Job mean/P95 | Barrier P95 | preempt |
|---|---:|---:|---:|---:|---:|---:|
| 静态 step cap16 | **0.19443** | **630.73** | **3382.45** | **134.74/233.97s** | 112.83s | 0 |
| EngineCore，无 continuation | 0.17179 | 635.30 | 3124.53 | 140.37/249.88s | 123.99s | 0 |
| continuation 5s | 0.16488 | 611.85 | 2968.51 | 142.94/241.13s | **100.69s** | 0 |

P0 5 秒版本相对静态基线：jobs/s `-15.20%`、FTS/s `-12.24%`、Job mean `+6.09%`、
Job P95 `+3.06%`；只有 Barrier P95 改善 `10.76%`。53 次 reservation 中仅11次
claim、42次 timeout，说明高 fanout 改变了 step completion 到下一 step readmission
的时间尺度。这个结果直接否定固定 grace 的通用性。

P2 的窗口32等于 workers32，不存在被 deferred 的其他 top-level job；continuation
在该设置下不会改变 admission 集合，因此未重复占用 NPU。

## 实现变化

- EngineCore scheduler 从结构化 `CISRequestMetadata` 读取 job/step/node/C/R；
- 使用准确 candidate `RequestStatus` 推导 terminal candidate 与实际 rollout 总数；
- all-terminal step 立即关闭，不再等待 idle grace；
- 支持最大窗口、continuation reservation/claim/expire 与完整 scheduler trace；
- 增加实验性的 work-conserving release；
- 增加 `job_fifo` 作为严格消融；
- 增加 `generation.ignore_eos` fixed-work 模式，保证调度 A/B 的工作量一致；
- runner 增加相应独立 variant，均不覆盖原有策略。

## 当前采用与下一步

当前默认选择不变：

- P0：`step_fifo + cap16` 用于吞吐基线；需要最低 barrier tail 时保留 cap6 对照；
- P1：`step_fifo + cap16`；
- P2：已有 elastic/step priority 基线。

动态 continuation、job_fifo 和 work-conserving release 均只保留为实验 variant。
下一版若继续，必须去掉 wall-clock grace，改为算法层显式发送 `step_closed` 与
`next_step_ready`，再以 prefix bytes、预计剩余 rollout token 和实际 decode 饱和点
计算换树代价。验收条件应是 P0/P1 同时不低于各自静态 oracle，且在变化 workload
下展示无需重新扫 cap 的收益；否则不再投入。

## 原始数据

机器可读摘要：

```text
docs/experiments/data/cis_predictive_continuation_20260914.json
```

远程原始数据根目录：

```text
/data/disk/wangzili/cis-predictive-scheduler-20260913
```

每个 run 的 `benchmark.json` 与 `cis-scheduler-trace.jsonl` SHA256 已记录在机器可读
摘要中。所有实验只使用启动前空闲的物理 NPU 1/4，结束后两张卡均已释放。
