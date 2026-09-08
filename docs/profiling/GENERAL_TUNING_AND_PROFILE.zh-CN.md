# Conditional IS General 调优与 Profiling 执行手册

## 本阶段边界

本阶段只完成三件事：验证真实 MiniAgent/Docker 数据流、把通用部署调到可证明的局部最优、在代表性算法压力下生成可复查的多层 profiling。暂不实现 candidate/rollout pipeline、定向 KV 复用、剪枝或 Megakernel。

算法固定为常博 `04492fe1` 的 ordinary `conditional_is`；模型固定为同一份 Qwen3-Coder-30B-A3B-Instruct BF16；目标 runtime 为 vLLM-Ascend v0.18 MRV1。

## 已实现

- mini-SWE-agent 2.4.6 adapter、常驻 CIS HTTP 服务、Docker/tool-call 数据流。
- generation-time sequence-logprob/Consilience statistics，正式 reward 不再重复整段前向。
- public `.traj.json` 模型调用快照提取；使用目标 Qwen tokenizer 和当前 bash tool chat template 精确计数，不截断、不 padding。
- TP2/PP2、自适应 MNS/MBT 扩边、memory、partial-prefill、worker 饱和搜索和断点续跑。
- stock/native sampler、eager/graph、HCCL AIV、async scheduling、CPU binding A/B。
- 四卡两实例 `2xTP2`/`2xPP2`、round-robin/least-outstanding 比较及局部参数复调。
- 无 profiler、MS Service Profiler、Ascend PyTorch Profiler 三次独立采集。
- 原始数据校验清单、派生 CSV/SVG、中文报告和 Perfetto/Chrome 合并时间线。

## 1. 数据准备

先下载公开轨迹仓库，再在装有目标 tokenizer 的环境中构造独立调用：

```bash
conditional-is-freeze-workload \
  --public-trajectories /data/qwen3-mini-swe-agent/trajectories \
  --model "$CIS_MODEL_PATH" \
  --output-directory artifacts/workloads/public \
  --seed 20260908 --total 64
```

输出 `public-64.jsonl` 和 `public-manifest.json`。manifest 记录最小值、中位数、P95、最大 prompt tokens，并在 16K/32K/64K 中选择能容纳最大 prompt、512 输出和 256 余量的最小 `max_model_len`。选择过程是对所有真实调用做固定 seed 的无放回随机抽样，不复制 prompt。

真实流程先跑 1 题，再连续跑 3 题。之后从本次服务产生的 `model_calls.jsonl` 中冻结至少 16 个自采调用；达到 128 个时生成正式 `tune-64`/`holdout-64`：

```bash
conditional-is-freeze-workload \
  --trace artifacts/traces/model_calls.jsonl \
  --output-directory artifacts/workloads/self-128 \
  --seed 20260908 --total 128
```

## 2. 双卡单实例调优

先生成搜索计划确认边界，再正式运行：

```bash
conditional-is-runtime-tune \
  --config configs/swebench/conditional_is_smoke.toml \
  --workload artifacts/workloads/public/public-64.jsonl \
  --holdout-workload artifacts/workloads/self-128/holdout-64.jsonl \
  --warmup-workload artifacts/workloads/warmup.jsonl \
  --devices 0,1 \
  --output-directory artifacts/tuning/two-card \
  --dry-run
```

去掉 `--dry-run` 后执行。搜索顺序不是完整笛卡尔积：

1. 在 P0 压力下分别 A/B native sampler、graph、HCCL AIV、async scheduling 和 CPU binding。
2. TP2/PP2 一起进入粗网格：MNS `{64,128,256,512}`，MBT `{8K,32K,128K}`，memory `0.92`。
3. 用 16/32/64 个请求 successive halving。
4. 只有胜者落在上界时，MNS 扩到 `{768,1024,1536,2048}`，MBT 扩到 `{256K,512K}`；每次增长低于 3%、失败、OOM 或 preemption 即停止该方向。
5. 补测 MNS `{192,384}`、MBT `{16K,64K}`，再依次搜索 memory 和 partial-prefill。
6. worker 从 `{4,8,16,32,64}` 找吞吐平台；有足够不同调用且 W64 仍是边界胜者时扩到 96/128/256。

每个 arm 启动干净 engine，避免继承上一个 arm 的 APC 状态；结果实时写入 `arms/<phase>/`，用 `--resume` 可安全续跑。硬门槛是成功率 100%、无 OOM、无 KV preemption，且 P95 不超过当轮最低值的 1.25 倍。

## 3. 四卡双实例调优

四卡不是预设 `2xTP2`。它复制双卡胜出的单实例拓扑，因此可能是 `2xTP2` 或 `2xPP2`，再比较静态均分和 least-outstanding：

```bash
conditional-is-four-card-tune \
  --config configs/swebench/conditional_is_smoke.toml \
  --two-card-result artifacts/tuning/two-card/result.json \
  --workload artifacts/workloads/public/public-256.jsonl \
  --holdout-workload artifacts/workloads/self-128/holdout-64.jsonl \
  --instance-devices 0,1 --instance-devices 2,3 \
  --output-directory artifacts/tuning/four-card
```

该阶段只局部复调总 worker、每实例 MNS/MBT 和 partial-prefill。`TP4`、`TP2xPP2` 是四卡单实例，不进入导师指定的主比较。

v0.18 的 P/D 依赖 KV connector、独立 prefiller/decoder 服务和请求路由。当前 CIS backend 是进程内 AsyncLLM，不具备把一次 generation 拆到外部 P/D 服务的无侵入接口。真机 capability smoke 会检查现有接口；若必须替换核心 backend/scheduler，则本阶段只记录“不满足最小兼容接线”，不为赶 profiling 临时开发新的执行系统。

## 4. 固定算法压力矩阵

矩阵保存在 `configs/swebench/profile_matrix.toml`：

| ID | C/R/B/L | Reward | 目的 |
|---|---|---|---|
| P0 | 15/3/128/512 | sequence-logprob | full、高压力主配置 |
| P1 | 8/3/128/512 | sequence-logprob | 中等 candidate |
| P2 | 4/2/128/512 | sequence-logprob | quick、低预算整体配置 |
| P3 | 15/3/128/512 | Consilience | reward/statistics 路径变化 |

全部使用 `T=1, top_p=1, top_k=None`。P2 同时改变 C/R，只用于观察整体低预算下瓶颈迁移，不用于单独归因 rollout。Consilience 固定 `top_k=5, window=0.2, skip=0.05, initial_penalty=3`。

## 5. 饱和与三层采集

“打满”表示增加独立请求并发后 jobs/s 增长不足 3%，同时 service profiler 的 batch/scheduler 数据仍显示 backlog；不是无限堆积队列。无 profiler 数字用于性能结论，service/torch pass 只用于归因 profiler 开销。

把双卡和四卡胜者分别物化为 deployment JSON 后执行固定矩阵：

```bash
conditional-is-deployment-manifest \
  --tuning-result artifacts/tuning/two-card/result.json \
  --id two-card-best --devices 0,1 \
  --output artifacts/deployments/two-card.json

conditional-is-profile-matrix \
  --config configs/swebench/conditional_is_smoke.toml \
  --matrix configs/swebench/profile_matrix.toml \
  --deployment artifacts/deployments/two-card.json \
  --deployment artifacts/tuning/four-card/deployment.json \
  --workload artifacts/workloads/self-128/holdout-64.jsonl \
  --warmup-workload artifacts/workloads/warmup.jsonl \
  --output-directory artifacts/profiles/formal
```

每个硬件配置和算法配置分别运行 `none`、`service`、`torch`。P0 无 profiler 重复三次；其他项只在观察到超过 5% 噪声时追加复验。MS Service 窗口默认 60 秒，torch 默认 10 秒；如果没有同时覆盖 candidate/rollout，torch 可单独延长到最多 30 秒。首轮关闭 stack/memory。

## 6. 交付物

每个 profile 目录保存：

- 原始 `*ascend_pt`、`analysis.db`、`trace_view.json`、MS Service 原始目录和分析后的 Chrome/CSV/DB。
- `benchmark.json`、每实例算法 JSONL、service log、0.5 秒 telemetry。
- `artifact-manifest.json`：Git/runtime/环境/命令/config/workload 哈希和全部文件 SHA256。
- `derived/analysis.json`、阶段/rank CSV、SVG、中文 `REPORT.zh-CN.md`。
- `derived/unified-timeline.json`：CIS stage 与导出 kernel 的 Perfetto/Chrome 视图；精确设备时序仍以未修改的 profiler trace 为准。

最终总报告必须比较八个“硬件配置 x 算法配置”组合，明确说明瓶颈是否随 C/R/reward 迁移，并把 general 部署问题、CIS 特有问题和 profiler 扰动分开。只有完成这一步后，才选择下一阶段要实现的定向优化。

原始 profile 最终从服务器归档时使用流式压缩并生成独立校验文件：

```bash
conditional-is-archive-artifacts \
  --source artifacts/profiles/formal \
  --output /data/profile-archives/conditional-is-formal.tar.gz
```
