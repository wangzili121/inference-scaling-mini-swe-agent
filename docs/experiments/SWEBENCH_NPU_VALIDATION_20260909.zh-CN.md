# SWE-bench 真实 NPU 流程验证记录（2026-09-09）

## 目的

本轮只验证 `mini-SWE-agent -> ordinary Conditional IS -> vLLM-Ascend v0.18 MRV1 -> Docker` 的真实数据流和两卡拓扑能力，不据此选择算法质量或部署最优参数。

固定组件：

- 上游 Conditional IS：`04492fe1e227e7171f9f20c971240c2d5ac3c535`
- 模型：`Qwen3-Coder-30B-A3B-Instruct` BF16
- runtime：`quay.io/ascend/vllm-ascend:v0.18.0`
- Agent：mini-SWE-agent `2.4.6`
- 算法 smoke：`C4/R2/B128/L512`、sequence-logprob、`T=1`、`top_p=1`
- 已启用：AsyncLLM、continuous batching、APC、chunked prefill、native categorical sampler、CPU binding、`FULL_DECODE_ONLY + Npugraph_ex`

## TP2 单题闭环

任务为 `astropy__astropy-12907`，Docker ARM64 镜像为 `greynewell/swe-bench-arm64:astropy-astropy-12907`。

结果：

- exit status：`Submitted`
- assistant/model calls：51；另有一次无 action 的格式重试
- prompt token 范围：1,583 到 23,959
- 模型调用累计时间：562.2 秒
- Docker 启动时间：2026-09-09 07:21:22 CST
- 轨迹保存时间：2026-09-09 07:32:00 CST
- 无 NPU OOM、无 KV preemption，Docker 正常自动清理
- 最终 `model_patch` 不是 unified diff，因此只证明数据流闭环，不计为有效质量结果

## TP2 三题并发闭环

三个任务由同一 TP2 常驻服务承载，Agent worker 数为 3，各自使用独立 Docker 容器。

| 任务 | 终态 | calls | prompt tokens | 模型调用累计时间 | patch |
|---|---:|---:|---:|---:|---|
| `astropy__astropy-13033` | Submitted | 54 | 1,970–19,420 | 645.9 s | 有效 unified diff |
| `astropy__astropy-13236` | Submitted | 22 | 1,731–9,016 | 230.9 s | 非 unified diff |
| `astropy__astropy-13398` | RepeatedFormatError | 14 | 2,493–12,963 | 64.1 s | 空 |

并发窗口为 2026-09-09 07:35:01–07:46:18 CST。服务观察到 `maximum_in_flight_requests=24`，无 OOM、无 KV preemption。`L=512` 下存在 completion 达到长度上限后缺少完整 tool-call 的情况，Agent 的格式重试路径已被实际覆盖。

这里的 `RepeatedFormatError` 和无效 patch 属于 P2 低预算下的模型输出/提交质量，不是服务崩溃。后续 general 调优只使用具有有效 bash action 的独立模型调用，完整失败记录仍保留在原始 trace 中。

## 固定 Workload

公开轨迹使用目标 tokenizer 和当前 Qwen chat template 精确展开，不截断、不 padding：

- 文件：`public-64.jsonl`
- SHA256：`a86280eedf3c9e7b7003a226bee30215a483ca4e7e6118c4d287528282cacaa8`
- prompt tokens：最小 1,516，中位 9,825.5，P95 28,402，最大 57,401
- 所需上下文：58,169；选择 `max_model_len=65,536`

从本轮当前模型与 Agent 产生的成功调用冻结 128 条，seed 固定为 `20260908`：

- `tune-64.jsonl`：`78ff5c127ee6f9d743f3cc568a23c065559b3d2dbdab0bcc7bb1cd68b4941a33`
- `holdout-64.jsonl`：`dba1b6a401445d3db2be119bee589b4cebbd9f9aa683dd3f8b45529e4e3b8eab`
- prompt token 范围：1,583–23,959

## PP2 Capability

最初在物理 NPU 2、3 上启动失败，错误为 `aclInit 107001: Invalid device ID`。根因是容器保留了物理设备编号，而 vLLM worker 按逻辑设备 0、1 初始化。提交 `629451f` 将物理卡映射到容器内连续逻辑卡；修复后两个 rank 正确识别为 PP0/PP1。

随后在最短公开真实请求 `public:django__django-15732:call-0` 上运行一条 P2：

- prompt tokens：1,516
- wall time：1.9784 秒
- jobs/s：0.5055
- engine requests：4
- generated tokens：274
- shared prefill tokens saved：5,632
- 成功率 100%，无 scoring forward、无 preemption

一次 capability 命令曾把 `workers=1` 误当作请求总数限制，开始顺序执行完整 64 条 workload；发现后中止，相关数据不进入性能比较。提交 `cdfc157` 为 standalone benchmark 增加显式 `--limit`。

## P0 General 部署快速收敛

固定 `C15/R3/B128/L512`、sequence-logprob 和公开 workload 后，先用 64-way burst 确认饱和边界，再用 32-way burst 选择可交付配置。每轮所有请求同时释放；成功率、transport retry 和 KV preemption 均作为硬门槛。低于 3% 的吞吐差异视为噪声，不用更复杂或风险更高的配置替换基线。

服务最初使用 `ThreadingHTTPServer` 默认 listen backlog，64-way burst 出现 6--8 次 `ConnectionResetError`。提交 `0410709` 将 backlog 提升至 1024，并用相同 `request_id` 做幂等 transport retry；修复后重跑的所有有效配置均为零 retry。旧结果只用于发现入口瓶颈，不参与选参。

32-way 核心结果：

| MNS | MBT | memory | jobs/s | forward slots/s | P95 (s) | preemption | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 384 | 32K | 0.92 | 0.154041 | 2,964.1 | 197.89 | 0 | 未超过 256 的 3% 门槛 |
| 256 | 32K | 0.92 | 0.154948 | 2,949.8 | 194.20 | 0 | MNS 基线 |
| 256 | 64K | 0.92 | 0.157076 | 3,017.8 | 196.54 | 0 | 收益不足 3% |
| 256 | 128K | 0.92 | 0.142523 | 4,491.5 | 214.25 | 13 | 硬门槛淘汰 |
| 256 | 32K | 0.88 | 0.155767 | 3,003.9 | 202.60 | 0 | 慢于 0.90 |
| 256 | 32K | 0.90 | **0.162483** | **3,076.8** | **183.10** | 0 | 两卡胜出配置 |
| 256 | 32K | 0.94 | 0.161861 | 3,073.2 | 187.65 | 0 | 与 0.90 持平但尾延迟更差 |
| 256 | 32K | 0.98 | 0 | 0 | 314.38 | 0 | warmup 后无法完成 burst |

64-way 压力下，MNS 256/384 分别产生 1/43 次 preemption；MNS 512 在更早的压力轮产生 109 次 preemption。64-way 因而用于观察过载瓶颈，不作为两卡最佳在线并发。两卡当前锁定为 `TP2、MNS=256、MBT=32768、memory=0.90、partial-prefill=(1,1)、workers=32`。

vLLM-Ascend v0.18 MRV1 对 `(2,1)` 和 `(4,2)` 均在启动 capability check 明确报错 `Concurrent Partial Prefill is not supported`，因此不再扫描其余组合。提交 `f2be1ec` 将该搜索设为 capability-gated。

## 四卡双 Instance

在两份相同 TP2 instance 上使用两卡胜出参数，总并发为 64，使每个 instance 约承接 32 个 CIS job。提交 `8be6203` 允许两种路由复用同一组已初始化服务，避免重复模型加载和 graph capture。

| 路由 | jobs/s | generated tokens/s | P95 (s) | success | retry | preemption |
|---|---:|---:|---:|---:|---:|---:|
| round-robin | **0.350569** | **1,213.3** | **165.99** | 100% | 0 | 0 |
| least-outstanding | 0.298827 | 1,063.2 | 185.35 | 100% | 0 | 0 |

least-outstanding 未带来收益，反而使 jobs/s 降低约 14.8%。四卡 general 配置因此锁定为 `2xTP2 + round-robin + total workers=64`。候选/rollout 分阶段路由与 P/D 不在 general baseline 中实现，留待完整 profiling 后凭 trace 决定。

## 原始数据位置

- smoke 归档：`/data/disk/wangzili/cis-artifacts-629451f/smoke-validation`
- PP2 capability：`/data/disk/wangzili/cis-artifacts-629451f/capability/pp2`
- 自采 workload：`/data/disk/wangzili/cis-artifacts-629451f/workloads/self-128`
- 双卡 tuner：`/data/disk/wangzili/cis-artifacts-fb4bc3c/tuning/two-card`
- 干净的 32/64-way focused 结果：`/data/disk/wangzili/cis-artifacts-0410709`
- 四卡路由对照：`/data/disk/wangzili/cis-artifacts-8be6203/four-card/routing`

上述目录保存原始服务日志、算法 JSONL、Agent trajectory、prediction、exit status、部署 manifest、诊断快照与 SHA256 清单。正式性能结论只从后续相同 workload、饱和负载、独立 profiler pass 的实验产生。
