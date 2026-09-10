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

## Profiling 依赖与有效性检查

正式采集前增加了 fail-fast 预检，模型加载和占用 HBM 之前必须同时满足：

- 配置、workload、warmup、输出目录和配置引用的环境变量有效；
- 本地模型目录含有效 `config.json/model_type`、tokenizer 文件和权重分片；
- Ascend 设备节点、驱动目录和 `npu-smi` 可用，且 `libatb.so` 可以实际加载；
- Service pass 固定 `msserviceprofiler==1.2.2`、`tzdata==2025.3`，且 analyzer CLI 和实际 vLLM hook 均可导入；
- Torch pass 能导入并调用 `torch_npu.profiler.profiler.analyse`；
- native categorical 的 sampler、Python extension、kernel library 和自定义 OPP 与编译产物逐项 SHA256 一致，算子已注册；
- 运行结束后每个实例日志都出现 native categorical 首次激活标记。

预检发现早期 P0 profiler 容器只设置了 `VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE=1`，但没有挂载 backport 资产；vLLM 日志明确将其报告为未知环境变量。因此下列早期 trace 只作为 stock-sampler 诊断数据，不作为“已有优化全部开启”的最终 baseline：

- 两卡 Service：`/data/disk/wangzili/cis-artifacts-e02af15/profiles/two-card/p0/service`
- 两卡 Torch：`/data/disk/wangzili/cis-artifacts-40da93c/profiles/two-card/p0/torch`
- 四卡 Service：`/data/disk/wangzili/cis-artifacts-b068967/profiles/four-card/p0/service`
- 四卡 Torch：`/data/disk/wangzili/cis-artifacts-869cc59/profiles/four-card/p0/torch`

正确挂载后的两卡 P0 native 无 profiler 基线使用 64 个独立请求、32 workers：成功率 100%，无 preemption，`0.156288 jobs/s`，P95 `313.91s`，forward token slots/s 为 `3706.84`。两个 TP rank 均出现 native sampler 激活标记。对应原始目录为 `/data/disk/wangzili/cis-artifacts-f655118/validation/native-p0-none`。

为排除启用 native sampler 后 MNS 最优点迁移，又只补测了相邻的 `MNS=384`，其余配置和 64 个请求保持不变。结果为 `0.138898 jobs/s`、P95 `331.32s`，成功率 100%，但出现 36 次 KV preemption；相对 `MNS=256` 吞吐下降 `11.13%`、P95 上升 `5.54%`。因此不再扩展 MNS，正式 profiling 锁定 `TP2、MNS=256、MBT=32768、memory=0.90、partial-prefill=(1,1)、workers=32`。原始目录为 `/data/disk/wangzili/cis-artifacts-53a19cf/validation/native-mns384-p0-none`。

相同环境关闭 native 后，stock 对照为 `0.151602 jobs/s`、P95 `337.42s`，同样 100% 成功且无 preemption。native 的 jobs/s 高 `3.09%`，P95 低 `6.97%`。两种 sampler 采样分布相同但随机流不 bit-exact，导致 candidate 长度、APC 和总 forward work 不同，因此这组 SWE workload 只证明正确 native 路径没有端到端回退，不能把全部差异都归因给 sampler kernel；sampler 的直接收益仍以既有同卡正反序两组 A/B 的几何平均 `1.115x` 为主要证据。stock 原始目录为 `/data/disk/wangzili/cis-artifacts-f655118/validation/stock-p0-none`。

Service Profiler 的原始 SQLite 数据和 `batch.csv`、`kvcache.csv` 均可解析，但 1.2.2 的厂商 trace exporter 无法将 vLLM 批量 request-id 列表转换为字符串，因而不生成 `chrome_tracing.json`。原始库、完整 analyzer 日志和部分成功产物均保留；项目自己的统一 Perfetto/Chrome 时间线直接使用 batch CSV 与 CIS 事件补齐该可视化，且把厂商导出失败显式标记为无效而非静默通过。

首次使用新 Service 镜像启动时，预检只验证了发行包和 analyzer CLI，却没有覆盖插件内部的 `ms_service_profiler` 运行时导入；同时容器把镜像原有 Ascend `PYTHONPATH` 覆盖成了 `/workspace/src`。服务在加载模型前退出，没有占用正式 workload，也没有形成 profile。提交 `e1396a6` 改为保留 Ascend Python 路径并追加项目源码路径，同时在 preflight 中直接导入 `msserviceprofiler.vllm_profiler.vllm_v1.batch_hookers`。后续四个正式 pass 均通过该检查。

## Native P0 正式 Profiling

正式 P0 固定 `C15/R3/B128/L512`、sequence-logprob、64 个独立请求；两卡为 `TP2 + workers=32`，四卡为 `2xTP2 + round-robin + workers=64`。所有 pass 均为 100% 成功、零 OOM，且每个实例的两个 TP rank 都出现 native sampler 激活标记。无 profiler 基线和四卡两个 profiler pass 均为零 KV preemption；两卡 Service/Torch pass 在 profiler 扰动下分别观察到 35/5 次 preemption，因此只用于瓶颈归因。

| 部署 | pass | jobs/s | P95 (s) | APC hit | NPU busy 中位数 | 暴露 HCCL/窗口 | 统一时间线事件 |
|---|---|---:|---:|---:|---:|---:|---:|
| 两卡 TP2 | none | 0.156288 | 313.91 | 96.58% | - | - | - |
| 两卡 TP2 | Service | 0.133945 | 368.53 | - | - | - | 700 |
| 两卡 TP2 | Torch | 0.149312 | 318.85 | 96.02% | 78.00% | 15.69% | 190,778 |
| 四卡 2xTP2 | none | 0.270663 | 166.85 | - | - | - | - |
| 四卡 2xTP2 | Service | 0.278633 | 188.32 | 97.65% | - | - | 1,072 |
| 四卡 2xTP2 | Torch | 0.302821 | 171.11 | 97.49% | 90.51% | 18.37% | 282,468 |

native categorical 在异步调度下与 stock sampler 的随机流不 bit-exact，各 pass 的实际生成长度和 forward work 也会变化。因此性能只引用 none pass；Service/Torch 的 jobs/s 仅用于说明采集运行本身完整，不能用它们与 none 的差值估算 profiler 开销。四卡 none 相对两卡 none 的 jobs/s 为 `1.73x`，同时 P95 降低 `46.85%`，但两次运行的总 forward work 不同，精确扩展效率还需按 token slots 归一化。

两卡 Torch 中 candidate/rollout 占阶段时间 `47.16%/52.84%`，四卡 Torch 中为 `63.53%/36.47%`；reward、weight、resample 均低于 `0.01%`。四卡虽然严格按 32/32 请求分流，Service/Torch 仍分别观察到 `31.83%/18.15%` 的实例完成时间 skew，说明请求数不能代表 CIS 剩余工作量。P0 还显示暴露 HCCL 在两卡和四卡窗口分别为 `15.69%/18.37%`。这些是下一轮代表性算法配置需要验证是否稳定迁移的候选瓶颈，不在本阶段直接实现调度或通信优化。

## Native P1 并发选择与正式 Profiling

P1 固定 `C8/R3/B128/L512` 和 sequence-logprob。其单个 CIS job 比 P0 轻，因此不能机械复用 P0 的 32 workers；在两卡同一 engine 配置和 64 个独立请求上补测饱和并发：

| workers | jobs/s | P95 (s) | P99 (s) | forward slots/s | preemption |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.242964 | **145.53** | 211.22 | 3,750.40 | 0 |
| 24 | **0.289787** | 172.26 | **190.09** | **4,322.87** | 0 |
| 32 | 0.259561 | 225.04 | 242.81 | 5,292.95 | 38 |

`workers=32` 的 preemption 不是正确性错误，也不应单独作为绝对淘汰条件；它说明该点进入 KV 过载区。这里 `workers=24` 相对 32 同时提升 jobs/s 约 `11.65%`、降低 P95 约 `23.45%`，因此没有“用 preemption 换取更高总体性能”的收益。P1 两卡正式并发锁定为 24。

四卡仍采用两份相同 TP2 instance 和 round-robin，总并发按每实例 24 设置为 48。所有正式 pass 均为 100% 成功、零 OOM、零 KV preemption，每个 TP rank 均确认 native categorical 激活：

| 部署 | pass | jobs/s | P95 (s) | APC hit | NPU busy 中位数 | 暴露 HCCL/窗口 | 统一时间线事件 |
|---|---|---:|---:|---:|---:|---:|---:|
| 两卡 TP2 | none | 0.289787 | 172.26 | - | - | - | - |
| 两卡 TP2 | Service | 0.254479 | 188.25 | 95.19% | - | - | 788 |
| 两卡 TP2 | Torch | 0.2433 | 168.81 | 95.95% | 94.46% | 13.36% | 383,716 |
| 四卡 2xTP2 | none | 0.356120 | 135.47 | - | - | - | - |
| 四卡 2xTP2 | Service | 0.455568 | 135.30 | 95.02% | - | - | 1,561 |
| 四卡 2xTP2 | Torch | 0.411659 | 120.34 | 95.96% | 95.34% | 15.25% | 256,438 |

性能仍只引用 none pass，Service/Torch 用于归因。P1 四卡 none 相对两卡 none 的 jobs/s 仅为 `1.23x`，按 forward slots/s 归一化也只有约 `1.28x`；这比 P0 的表观 `1.73x` 扩展差。四卡虽然按 32/32 job 静态均分，none pass 两个 endpoint 的 forward work 相差约 `35.9%`，Service 进一步报告 `86.14%` 的 instance latency skew。

阶段也随拓扑明显迁移：两卡 Torch 的 candidate/rollout 为 `55.25%/44.75%`，四卡 Torch 变为 `66.99%/33.00%`；reward、weight、resample 仍低于 `0.01%`。暴露 HCCL 从两卡 `13.36%` 升到四卡 `15.25%`。因此截至 P1，稳定出现的候选瓶颈是 TP 通信暴露和按 job 数静态路由造成的 work skew，而不是 CPU reward；是否足以形成最终优化建议仍需 P2/P3 代表性配置验证。

本轮正式原始目录：

- P1 两卡 Service：`/data/disk/wangzili/cis-artifacts-1f16022/profiles/two-card/p1/service-w24`
- P1 两卡 Torch：`/data/disk/wangzili/cis-artifacts-1f16022/profiles/two-card/p1/torch-w24`
- P1 四卡 none/Service/Torch：`/data/disk/wangzili/cis-artifacts-e1396a6/profiles/four-card/p1/`

Torch 两卡每个 rank 的原始 `trace_view.json` 约 2.05 GB，并各自保留 `analysis.db`、`kernel_details.csv` 和 `operator_details.csv`。Service exporter 仍受已记录的 request-id 列表转换问题影响，不生成厂商 `chrome_tracing.json`，但原始数据库、batch/KV CSV 和项目统一时间线完整。

## Native P2 正式 Profiling

P2 固定 `C4/R2/B128/L512` 和 sequence-logprob。两卡使用 32 workers，四卡使用 64 workers。无 profiler pass 均为 100% 成功、零 OOM、零 KV preemption：

| 部署 | pass | jobs/s | P95 (s) | APC hit | NPU busy 中位数 | 暴露 HCCL/窗口 |
|---|---|---:|---:|---:|---:|---:|
| 两卡 TP2 | none | 0.405948 | 136.31 | - | - | - |
| 两卡 TP2 | Service | 0.378150 | 144.40 | 89.34% | - | - |
| 两卡 TP2 | Torch | 0.329767 | 167.35 | 88.64% | 95.88% | 18.10% |
| 四卡 2xTP2 | none | 0.538301 | 98.22 | - | - | - |
| 四卡 2xTP2 | Service | 0.644522 | 82.74 | 90.59% | - | - |
| 四卡 2xTP2 | Torch | 0.642923 | 81.02 | 90.01% | 98.43% | 23.49% |

P2 四卡 none 相对两卡 none 的 jobs/s 为 `1.33x`，forward-token-slots/s 也为 `1.31x`。Torch 两卡受 profiler 扰动出现 5 次 preemption，不用于性能结论。P2 的 candidate 占比在两卡/四卡 Torch 中为 `65.68%/84.30%`，rollout 为 `34.32%/15.70%`；四卡 endpoint mean latency skew 为 `16.92%`。即使低预算配置已把设备喂到 98%，通信暴露和 instance skew 仍然存在。

原始目录：

- 两卡 none：`/data/disk/wangzili/cis-artifacts-5e895a9/validation/p2-w32-none`
- 两卡 Service/Torch：`/data/disk/wangzili/cis-artifacts-5e895a9/profiles/two-card/p2/`
- 四卡三 pass：`/data/disk/wangzili/cis-artifacts-5e895a9-first/profiles/four-card/p2/`

## Native P3 正式 Profiling

P3 与 P0 使用相同 `C15/R3/B128/L512`，只把 reward 切换为 Consilience。generation-time top-5 statistics 已启用，全部 pass 的 `score_calls=0`，因此该对照不包含重复 scoring forward。

| 部署 | pass | jobs/s | P95 (s) | APC hit | NPU busy 中位数 | 暴露 HCCL/窗口 |
|---|---|---:|---:|---:|---:|---:|
| 两卡 TP2 | none | 0.160988 | 344.65 | - | - | - |
| 两卡 TP2 | Service | 0.146586 | 344.66 | 96.06% | - | - |
| 两卡 TP2 | Torch | 0.152845 | 337.07 | 96.10% | 87.17% | 14.23% |
| 四卡 2xTP2 | none | 0.363746 | 167.18 | - | - | - |
| 四卡 2xTP2 | Service | 0.271396 | 198.30 | 97.37% | - | - |
| 四卡 2xTP2 | Torch | 0.348749 | 176.90 | 96.97% | 82.41% | 13.99% |

P3 四卡 none 的表观 jobs/s 扩展为 `2.26x`，但 forward-token-slots/s 只扩展 `1.48x`；不能把不同随机生成工作量带来的表观差异算作部署收益。两卡 candidate/rollout 接近 `49.07%/50.92%`，四卡变为 `63.30%/36.69%`。最新四卡 Torch 在严格 32/32 job 分配下仍有 `30.19%` endpoint mean latency skew，四个 rank 的 NPU busy 为 `75.13%-86.58%`，而 reward、weight、resample 合计远低于 `0.01%`。

原始目录：

- 两卡 none/Service：`/data/disk/wangzili/cis-artifacts-5e895a9/{validation/p3-w32-none,profiles/two-card/p3/service-w32}`
- 两卡 Torch：`/data/disk/wangzili/cis-artifacts-5e895a9-first/profiles/two-card/p3/torch-w32`
- 四卡三 pass：`/data/disk/wangzili/cis-artifacts-5e895a9-first/profiles/four-card/p3/`

## P0 Mixed-stage 延迟采集

原 P0-P3 Torch 的 10 秒窗口几乎只覆盖 candidate。为避免将 candidate-only
行为误写成完整 CIS block 的瓶颈，新增 P0 四卡延迟 60 秒、持续 15 秒的 Torch
pass。64 请求全部成功、零 KV preemption，原始目录为
`/data/disk/wangzili/cis-mixed-profile/p0-four-card-torch`。

- 396,158 条窗口内 kernel 记录与算法 `start_unix_us` 直接对齐，无估算偏移。
- profile-0 有 11.13 秒 candidate+rollout；profile-1 有 7.02 秒 mixed，随后
  7.98 秒 rollout-only。
- profile-1 两 rank 的 mixed busy 平均 `71.8%`，rollout-only 为 `82.3%`，
  mixed batch shape 对应 `10.5pp` 的设备利用率下降。
- `FusedInferAttentionScore` 四 rank 累计 `41.03 device-seconds`，相当于
  device-busy union 的 `75.4%`；同时 APC token hit 为 `97.69%`。
- 暴露 HCCL/profile-window 从 candidate-only 的 `18.37%` 降到 mixed-stage
  的 `6.10%`，通信压力随算法阶段迁移。

可视化导出器将完整算法 Gantt、1ms NPU/HCCL 覆盖、阶段归一化统计、120ms
逐 kernel 下钻和 Service batch 事件保留在同一查看器中，并标注 trace 支持的
优化空间。

## 跨配置结论

- P0-P3 全部没有正式 scoring forward，CPU reward 不是剩余瓶颈。
- APC 已保存大量重复 prefill，但 `FusedInferAttentionScore` 在代表窗口中仍占设备 busy 时间约 38%-75%；APC 不会合并不同 decode request 对共享 KV 的读取。
- 四卡的 work-normalized 扩展只有约 `1.28x-1.48x`，静态 job 数均分在四组配置中都出现实际工作偏斜。
- 下一阶段优先验证 engine 内 branch-on-token、利用固定 `C -> R` 两层树的
  Ascend Forest Attention，以及执行中更新的 remaining-work routing；host
  streaming/bounded/frontier 已被后续 A/B 排除，不继续把普通 MNS/MBT 扩边当作
  算法创新。

## 算法专属 Scheduler 初轮 A/B

完成正式 profiling 后，在同一 P0、TP2、64-request burst 下比较了 host 侧 exact
调度变体。它们保持 C/R/B/L、reward、seed 生成规则不变，只改变 rollout 的提交
时机。该轮为同机并发筛选，作用是快速淘汰，不用来宣称小幅正收益。

| 变体 | jobs/s | forward slots/s | P95 (s) | preemption | 相对 baseline |
|---|---:|---:|---:|---:|---:|
| all-at-once baseline | 0.149947 | 3,804.6 | 335.40 | 4 | - |
| candidate completion streaming | 0.135283 | 3,326.4 | 367.35 | 39 | -9.78% jobs/s |
| per-job bounded submission=15 | 0.130760 | 3,204.7 | 399.17 | 30 | -12.79% jobs/s |

streaming 的 candidate-to-rollout 中位提前量仅为 `5.85ms`，P95 为 `2.527s`；
而 candidate 阶段累计为 `4810.8s`。固定 B 的 candidate 本来就接近同步结束，
host 回调几乎没有可利用的阶段重叠，反而把 rollout 拆成小批并增加 KV 过载。
因此这两个方案已明确淘汰，不能再写成后续默认优化。

静态 whole-job work routing 也不是稳定答案：P1 的 work-normalized throughput
提升 `22.65%`，P3 却下降 `11.39%`。随机 EOS 使实际工作量不可预知，下一轮改为
全 engine 共享的 rollout frontier 和执行中更新的 remaining-work routing。完整
机器可读数据在 `docs/experiments/data/cis_scheduler_ab_20260910.json`。

全局 rollout frontier 随后也完成 P0 筛选：capacity `128/192/240` 的
work-normalized throughput 相对 paired baseline 分别为 `-4.08%/+0.78%/-3.38%`。
192 产生 47 次 KV preemption，P95 为 `425.42s`；240 产生 52 次 preemption；
128 虽把 preemption 降至 9 次，却累计等待 `577.21s` 并使 P95 增至 `400.45s`。
三组均不满足 5% 收益与尾延迟门槛。结合前述 streaming/bounded 负结果，host
层 admission 顺序已充分排除，下一步不再扩 capacity，而是验证 engine 内
branch-on-token 与共享 KV decode attention。

## P/D Capability

v0.18 源码包含 `MooncakeConnectorV1` 和单机 P/D 文档，但当前镜像中的 `mooncake.engine.TransferEngine` 无法导入，缺少 `ascend_transport.so`。挂载主机驱动后结果不变；官方文档要求另行以 `USE_ASCEND_DIRECT=ON` 编译安装 Mooncake。P/D 因此记录为依赖不完整的 capability failure，不进入本轮 general baseline，也不为赶 profiling 临时修改核心 scheduler。

## 原始数据位置

- smoke 归档：`/data/disk/wangzili/cis-artifacts-629451f/smoke-validation`
- PP2 capability：`/data/disk/wangzili/cis-artifacts-629451f/capability/pp2`
- 自采 workload：`/data/disk/wangzili/cis-artifacts-629451f/workloads/self-128`
- 双卡 tuner：`/data/disk/wangzili/cis-artifacts-fb4bc3c/tuning/two-card`
- 干净的 32/64-way focused 结果：`/data/disk/wangzili/cis-artifacts-0410709`
- 四卡路由对照：`/data/disk/wangzili/cis-artifacts-8be6203/four-card/routing`
- 正式 native P0：`/data/disk/wangzili/cis-artifacts-e1396a6/profiles`
- 四卡 native P0 无 profiler：`/data/disk/wangzili/cis-artifacts-e1396a6/validation/four-card-p0-none`
- 四卡 P0 mixed-stage Torch：`/data/disk/wangzili/cis-mixed-profile/p0-four-card-torch`
- 双卡 P0 request-aware Service：`/data/disk/wangzili/cis-request-profile/p0-two-card-service`
- 双卡 P0 request-aware Torch：`/data/disk/wangzili/cis-request-profile/p0-two-card-torch`

最后两组补采给每条 engine request 保留了结构化
`job/block/candidate/rollout/parent` 身份和 submit/scheduled/first-token/finish
生命周期。Service pass 可精确回放每条请求参加的 batch；Torch pass 使用相同
双卡 TP2 配置和 30 秒窗口，用于对齐 rank 0/1 原始 NPU/HCCL 事件。Service
代表 block 的 51 条请求中有 50 条落入窗口，同 candidate sibling 的 batch
同批率为 `99.68%`，相关 batch 的 `93.26%` 为 Prefill+Decode 混合且全部达到
size 256；rollout 完成跨度为 `180.65s`。这些数据把“普通并发不足”排除，并把
后续重点收窄为 barrier 长尾、满批阶段干扰和同批 shared-trunk attention。

同配置双卡 Torch 补采含 4,556 条原始 kernel 事件。完整 profiler span 的
rank 0/1 busy 为 `82.48%/84.40%`，暴露 HCCL/profile 为 `8.78%/11.07%`；
与算法 trace 对齐的 30 秒 mixed 窗口 busy 为 `73.46%/73.09%`。
`FusedInferAttentionScore` 双 rank 累计 `33.83 device-seconds`，占 device-busy
union 的 `67.3%`，`hcom_allReduce` 累计 `5.83 device-seconds`。因此新的逐请求
查看器把 36-job 交错、193-request queue peak、逐 batch 成员、TP rank 0/1 的
1ms compute/HCCL、FIA Pareto 和代表 block 的 180.65 秒 barrier tail 放在同一
时间方向中；Service 与 Torch 是同配置独立 pass，绝不伪造成同一次采集。

上述目录保存原始服务日志、算法 JSONL、Agent trajectory、prediction、exit status、部署 manifest、诊断快照与 SHA256 清单。正式性能结论只从后续相同 workload、饱和负载、独立 profiler pass 的实验产生。
