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

## 原始数据位置

- smoke 归档：`/data/disk/wangzili/cis-artifacts-629451f/smoke-validation`
- PP2 capability：`/data/disk/wangzili/cis-artifacts-629451f/capability/pp2`
- 自采 workload：`/data/disk/wangzili/cis-artifacts-629451f/workloads/self-128`
- 双卡 tuner：`/data/disk/wangzili/cis-artifacts-fb4bc3c/tuning/two-card`

上述目录保存原始服务日志、算法 JSONL、Agent trajectory、prediction、exit status、部署 manifest、诊断快照与 SHA256 清单。正式性能结论只从后续相同 workload、饱和负载、独立 profiler pass 的实验产生。
