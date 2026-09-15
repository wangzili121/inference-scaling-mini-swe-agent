# CIS Scheduler Plugin 产品设计

日期：2026-09-14

## 1. 产品判断

最终产品不应把整个 `candidate x rollout` 物理地合成一条 vLLM request，也不应强迫
一棵 tree 独占一个 attention batch。采用四种不同粒度：

- admission/accounting 单位：完整 CIS step tree；
- prefix-locality 单位：同一 candidate 的 sibling rollouts；
- 实际执行与 continuous-batching 单位：ready branch；
- 用户可见完成单位：完整 CIS job。

这样 scheduler 能理解 tree 的 KV 成本和 barrier，但仍允许不同 tree 的 ready branch
共同填满 MNS/MBT。它避免两个已验证的问题：固定 step cap 随 context/C/R 变化失效，以及
candidate-subtree 小批量提交导致设备欠填。

## 2. 已有实验证据

以下路径不作为产品默认值：

- candidate completion streaming 会造成碎片化 fanout；
- candidate-subtree K8 在代表性 P0 上相对 Step control 的 jobs/s 下降约17.8%；
- direct KV fork 在 APC 已约96%时相对最佳 Step 调度只有约1%的边际变化；
- Forest Attention Python/FIA 兼容层破坏原生 paged attention 与 full graph，端到端回退；
- 无显式 continuation 的 EngineCore tree scheduler 在 P1 将 prefill 从627240增加到
  1688232，wall time 增加约11.4%；
- wall-clock continuation 在 P1略有收益，但迁移到 P0 后 FTS/s下降12.24%。

因此 fork 只保留为可选 fast path。只有 profile 证明 request lifecycle 或未命中的 prefix
tail 成为瓶颈时才开启；它不能代替 tree-aware admission，也不能独占 batching。

## 3. 插件结构

```text
Conditional IS adapter
  -> schema-versioned cis_request metadata
  -> explicit step acquire / resize / transition / release
  -> CISTreeScheduler via vLLM scheduler_cls
  -> native vLLM priority, token scheduler, continuous batching and KV allocator
```

算法层拥有最准确信息，因此负责：

- 在 candidate 提交前声明 prompt、C/R/B/L 和 worst-case reservation；
- candidate 结束后用实际 terminal 数缩小 reservation；
- reward/resample 后显式将当前 claim 原子转移到下一 step；
- job 完成或异常时释放 claim。

EngineCore 插件负责：

- 解析带版本的 `cis_request` 元数据；
- 从 `num_gpu_blocks * block_size` 获取真实每-rank KV capacity；
- 对 step tree 做 block-rounded capacity guard；
- 将已 admission 的 branch 留给 vLLM 原生 continuous batching；
- 输出 tree、queue、KV、preemption 和 barrier trace。

当前算法层 `runtime_kv_budget` 已具有显式 transition。此前 P1匹配静态最优、P0的
FTS/s提升5.06%的结果属于 rolling peak-token，不是尚未运行 NPU A/B 的 runtime-KV。
新增 `step-tree-runtime-kv` 是纯 EngineCore 容量消融，它没有猜测固定 cap，但仍缺显式
跨进程 transition，不能在 NPU A/B 前取代外层控制器。

## 4. 严格 A/B

`run_cis_scheduler_plugin_ab_remote.sh` 固定同一模型、workload、MNS/MBT、sampling 和 seed，
分别在 P0 `C15/R3` 与 P1 `C8/R3 fixed-work` 比较：

1. P0 静态 cap6 尾延迟 oracle 与 cap16 吞吐 oracle，P1 静态 cap16；
2. 已取得 P0 `+5.43%` jobs/s、P1 `+0.11%` jobs/s 的 rolling peak-token；
3. 算法层 `runtime_kv_budget`；
4. EngineCore `step-tree-runtime-kv`。

这里必须区分 rolling 与 runtime-KV：已有 `+5.43%/+0.11%` 是 rolling peak-token 相对
静态 cap16 的结果，硬件容量推导的 runtime-KV 尚无 NPU 结论。

smoke 只运行两个尚未验证的新 runtime-KV 架构，不重复静态与 rolling。只有 EngineCore
版本成功率100%、无 OOM，且 jobs/s/FTS/s 不比算法层版本低10%，才保留为 full 候选；
否则直接选择算法层版本。full 通过 `CIS_PLUGIN_RUNTIME_VARIANT=outer|engine` 只运行胜出的
runtime-KV 架构以及静态、rolling 基线，避免重新运行已经淘汰的架构。保留 `both` 仅用于
结果冲突时复验。正式结果同时比较 jobs/s、FTS/s、generated tokens/s、Job/Barrier P95、
prefill、APC 和 preemption，不能只用 jobs/s。

完整矩阵先决定 outer 与 EngineCore 架构。只有胜出且无回退的架构进入 `tune`，再扫描
runtime KV 安全系数 `{0.80,0.90,0.95}`；若最优点位于0.95且没有 preemption，才补0.97，
若0.80仍有明显欠填则补0.70。不能同时穷举两种架构。胜出配置随后迁移到 P2 与
8K-128K workload，必须接近各 workload 的静态 oracle 才能作为默认插件策略。

若纯 EngineCore 版本因 continuation 重算再次失败，产品采用两层插件，不再为形式上的
“纯 scheduler”牺牲性能：算法 adapter 持有显式 admission 生命周期，vLLM scheduler
只负责结构化 priority 与 telemetry。

## 5. 产品化验收

- 安装 wheel 后只需设置 `scheduler_cls` 和 policy 配置，不复制仓库源码；
- 非 CIS request 完全回退到原生 vLLM 行为；
- 未识别的 metadata schema fail closed，不静默误调度；
- 不依赖 request ID 正则解析；
- runtime capacity 来自实际 EngineCore，不把 cap16固化为默认；
- `8K-128K` 四档均接近各档静态 oracle，且无 OOM/KV preemption；
- P0/P1/P2、双卡和四卡均保留收益；
- direct fork、Forest Attention 和算法剪枝均为独立可选模块，不污染默认 scheduler。

## 6. 当前实现与验证状态

已实现 `step-tree-runtime-kv`：`CISTreeScheduler` 直接读取 EngineCore 的
`cache_config.num_gpu_blocks` 与 `block_size`，按 `VLLM_CIS_KV_CAPACITY_FRACTION`
生成容量预算，并按真实 KV block 对 prompt、candidate suffix 和 rollout tail 向上取整。
单棵树大于软预算时允许独占前进，避免 admission 死锁；vLLM 仍是最终 KV allocator。

`cis_request` 增加 `schema_version=1`；插件拒绝未知版本，旧的无版本 payload 暂按 v1
兼容。非 CIS 请求继续执行 `Scheduler.add_request()` 原始路径。

在真实 vLLM-Ascend v0.18 镜像中完成了无计算卡构造验证：3976个 KV block、每 block
128 token 得到508928-token capacity；0.9安全系数得到458035-token budget。相关
`cis_scheduler` 与 backend 测试为 `37 passed`。两台服务器检查时仍无未被运行中容器
映射的安全 TP2，因此尚无本模式的 NPU 性能结论。

## 7. 下一策略的准入门槛

已增加两阶段安全准入容量模拟器，但不直接加入默认运行时。它把一个 step 拆成 candidate
当前 allocation 和 rollout maximum claim，并用 Banker-style 安全序列避免所有 tree
同时等待 expansion 的死锁。

在458035-token 软预算、MNS 256下，固定 trace 顺序的 P0 纯 KV frontier 从16扩大到35，
但加入 candidate branch 执行边界后只从16变成17；P1则都被32棵 tree、256条 branch限制。
500次 shuffle 中，P0/P1 的 MNS-bound 中位数分别从14/23提高到17/30，说明它更可能解决
异构上下文 head-of-line blocking，而不是稳定长度 workload 的首要瓶颈。

新增模拟器及原有 scheduler/backend 相关用例已在真实 vLLM-Ascend v0.18 镜像中运行，
结果为 `43 passed`；该验证不占用 NPU，也不代表两阶段策略已经取得性能收益。

当前仓库已成功构建 `inference_scaling-0.1.0-py3-none-any.whl`，并在未挂载源码的干净
vLLM-Ascend v0.18 容器中以 `--no-deps` 安装后导入
`inference_scaling.arllm.backends.cis_scheduler.CISTreeScheduler`。wheel SHA256 为
`c7558333abcb6811868e90369f62016d8c28cc169428fe81fa423466133c613c`。这证明现有
`scheduler_cls` 接口可按插件方式装载；待 NPU A/B 确认默认策略后，再拆分轻量独立 wheel，
不提前固化实验接口。

因此产品路线为：

1. 先完成 static、算法层 runtime-KV、EngineCore runtime-KV 的 NPU A/B；
2. 只有 profile 证明 full reservation 导致欠填或异构 head-of-line blocking，才实现
   `candidate_closed -> rollout_granted` 两阶段运行时；
3. 运行时版本必须有 completion lane、expansion 优先、KV residency 监控和 full-reservation
   fallback，不能采用固定预测折扣。

完整模型、结果和相关工作见
`docs/experiments/CIS_TWO_PHASE_SAFE_ADMISSION_20260914.zh-CN.md`。

## 8. 与近期工作的关系

- MISA-T 将 mixed-rollout admission 建模为 KV commitment，并保持 continuation；它支持
  gateway 插件形态，但没有 CIS 的 candidate/rollout/barrier 两阶段树语义。
- TOPAS 面向 multi-agent workflow DAG 的关键路径与 prefix state；本插件限定在一次 CIS
  调用内部，不引入多轮 Agent 执行感知。
- BatchLLM 联合 prefix grouping、token batching 和 attention；它说明 grouping 不能脱离
  batch shape，符合我们 candidate-subtree 小批量路径的负结果。
- vLLM priority 与 scheduler_cls 是底层通用原语，不提供 CIS claim、barrier 或原子跨 step
  transition。通用 scheduler plugin RFC 若落地，可成为标准装载接口，而非替代策略本身。

## 9. Muyuan 提交形态

Muyuan 当前要求插件能独立构建、测试和使用，Benchmark 只通过稳定 CLI/API 调用插件。
因此最终不应把整个 inference-scaling 仓库复制到 `plugins/`，而应拆成一个轻量 wheel：

```text
plugins/cis-scheduler/
├── pyproject.toml
├── README.md
├── src/muyuan_cis_scheduler/
│   ├── metadata.py       # 版本化 job/step/candidate/rollout 元数据
│   ├── lifecycle.py      # acquire/resize/transition/release 协议
│   ├── policy.py         # runtime-KV admission 与可选动态反馈
│   ├── vllm.py           # scheduler_cls 适配器
│   └── telemetry.py      # queue/KV/barrier/preemption 事件
├── adapters/
│   └── conditional_is.py # 常博实现所需的薄 adapter/参考补丁
└── tests/
```

默认采用“两层插件”而不是强求零侵入：

1. Conditional IS adapter 在 candidate 开始、candidate 结束、resample 和异常退出处分别
   发出 acquire、resize、transition、release。它不改变 C/R/B、采样顺序、reward 或输出；
2. 独立 runtime wheel 实现容量控制、priority 和 telemetry，通过 vLLM `scheduler_cls`
   装载；非 CIS 请求保持原生行为；
3. 纯 EngineCore 推断模式保留为可选的零侵入兼容模式和消融，不因接口形式牺牲性能；
4. SWE-bench workload、部署脚本和性能报告放到 `benchmarks/swebench/`，不进入插件核心。

只有 runtime-KV NPU A/B 和跨上下文迁移完成后，才把当前实验代码拆成上述目录。默认策略、
安全系数和 telemetry schema 都必须由实测结果冻结，避免提前产品化一个尚未胜出的策略。

## 10. 首轮真实 NPU A/B（2026-09-15）

服务器 A 的物理 NPU 2/3 与 6/7 在启动前均无计算进程。本轮只覆盖长期交互容器的设备
映射，没有停止或抢占其他容器；每次运行结束后均释放本项目容器。统一配置为 TP2、
MNS 256、MBT 32768、memory 0.90、APC、chunked prefill、native categorical sampler 和
`FULL_DECODE_ONLY + Npugraph_ex`。

### 10.1 outer 与 EngineCore 架构选择

P1 16-job fixed-work smoke 中，两种实现的 engine requests、generated tokens、prefill 和
FTS 完全相同，均为100%成功、0 preemption：

| 实现 | jobs/s | FTS/s | Job mean | Job P95 |
|---|---:|---:|---:|---:|
| outer runtime-KV | 0.04718 | 2121.52 | 322.02 s | 336.08 s |
| EngineCore runtime-KV | 0.04537 | 2040.29 | 332.82 s | 348.16 s |

EngineCore 相对 outer 的 jobs/s/FTS/s 均下降3.83%，mean 增加3.35%，P95 增加3.60%。P0
非 fixed-work smoke 曾出现 EngineCore jobs/s 较高，但生成工作量不同，不能推翻 P1 的
严格结果。因此产品默认选 outer 生命周期控制器；EngineCore 版本只保留作零侵入兼容和
消融。

### 10.2 P0 full

| 实现 | jobs/s | FTS/s | Job mean | Job P95 | preemption |
|---|---:|---:|---:|---:|---:|
| static cap16 | 0.16474 | 2975.27 | 133.71 s | 246.19 s | 0 |
| outer runtime-KV | 0.17165 | 3045.65 | 131.21 s | 229.29 s | 0 |

outer 相对 static cap16：jobs/s +4.19%，FTS/s +2.37%，mean -1.88%，P95 -6.86%。两者
均为100%成功。P0 使用真实 EOS，生成量不同，因此优先采用 FTS/s 与延迟共同判断。

### 10.3 P1 full 静态安全边界

P1 使用 fixed work，三组成功运行的 engine requests=3328、generated tokens=720896、
prefill tokens=627240、FTS=1344808，因而可以直接比较：

| 实现 | jobs/s | FTS/s | Job mean | P50 | P95 | max in-flight |
|---|---:|---:|---:|---:|---:|---:|
| static cap8 | 0.04804 | 2018.90 | 386.90 s | 379.85 s | 665.83 s | 96 |
| static cap12 | 0.05113 | 2148.87 | 382.75 s | 424.26 s | 622.05 s | 192 |
| outer runtime-KV | 0.05221 | 2194.14 | 447.05 s | 382.47 s | 612.84 s | 520 |

static cap16 在同一32-job burst 首批 forward 中额外申请798 MiB时 OOM，不能再作为该
workload 的安全静态配置。outer 相对最佳安全 static cap12：jobs/s/FTS/s +2.11%，P50
-9.85%，P95 -1.48%，P99 -1.88%，但 mean +16.80%。动态策略已经略胜人工静态点的吞吐
和中尾延迟，并自动避开 cap16 的 OOM；mean 回退说明其跨 job 公平性仍需继续优化，当前
不能宣称已经完成产品化。

### 10.4 runtime-KV 安全系数与公平性

P1 继续固定同一份 32-job fixed-work，比较 `step_fifo`、`job_fifo` 和 KV 安全系数。
除 `job_fifo f0.80` 的 prefill 增加7168 token（1.14%）外，generated tokens、engine
requests 和算法参数完全一致；所有运行均100%成功、0 preemption。

| 实现 | jobs/s | FTS/s | Job mean | P50 | P95 | max in-flight |
|---|---:|---:|---:|---:|---:|---:|
| static cap12 + step FIFO | 0.05113 | 2148.87 | 382.75 s | 424.26 s | 622.05 s | 192 |
| runtime-KV f0.80 + step FIFO | 0.05355 | 2250.50 | 429.16 s | 365.56 s | 594.83 s | 392 |
| runtime-KV f0.80 + job FIFO | **0.05363** | **2265.75** | **404.28 s** | **332.32 s** | **591.94 s** | 384 |
| runtime-KV f0.70 + job FIFO | 0.05121 | 2152.30 | 419.35 s | 396.46 s | 605.58 s | 328 |

`job_fifo f0.80` 相对 `step_fifo f0.80` 的 jobs/s基本不变（+0.14%），但 mean/P50/P95
分别下降5.80%/9.10%/0.49%。这说明跨 step 保持 job 连续性可以改善公平性，而不需要
牺牲吞吐。相对最佳安全静态 `cap12`，它的 jobs/s/FTS/s +4.88%/+5.44%，P50/P95/P99
-21.67%/-4.84%/-4.71%，但 mean仍高5.63%。静态策略让最早一批 job 在约131秒完成，
显著拉低 mean；动态策略改善了中位数、尾部和总 makespan，但仍有更多 job 同时竞争。

将比例从0.80降到0.70后，jobs/s/FTS/s相对 f0.80下降4.50%/5.01%，mean/P50/P95/P99
分别增加3.73%/19.30%/2.30%/3.99%。因此0.80是当前安全系数的明确局部最优，停止继续
向下扫描。后续若继续改善 mean，应引入由 `max_num_seqs` 推导的第二种 sequence/branch
资源约束，而不是继续手调 KV 比例。

### 10.5 P0 priority 与随机工作量消融

P0 的64-job真实 EOS 运行中，`job_fifo f0.80` 相对 `step_fifo f0.80` 的 jobs/s/FTS/s
+15.01%/+10.56%，mean/P50/P95/P99 -5.67%/-4.90%/-5.61%/-1.52%；相对静态 cap16 的
jobs/s/FTS/s +16.03%/+12.08%，mean/P95/P99 -2.27%/-11.65%/-6.48%。但它同时少生成
12.62%的 engine requests，说明调度改变了 native categorical 的实际随机路径，不能把
全部收益归因于执行效率。

因此又用16个 job、固定512输出做严格 A/B：

| 实现 | jobs/s | FTS/s | Job mean | P50 | P95 | prefill | engine requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| runtime-KV f0.80 + step FIFO | **0.02606** | 1826.61 | 457.54 s | **434.04 s** | **586.36 s** | **448661** | 3120 |
| runtime-KV f0.80 + job FIFO | 0.02602 | 1905.58 | **422.79 s** | 443.32 s | 608.31 s | 499221 | 3120 |

两边 generated tokens均为675840、engine requests均为3120、0 preemption。`job_fifo` 的
jobs/s -0.18%、mean -7.60%，但 P50/P95/P99 +2.14%/+3.74%/+0.87%，并增加11.27%
prefill。更高 FTS/s来自额外 prefill，不是更高 jobs/s。因此 P0 中 `job_fifo` 是降低平均
JCT的目标模式，`step_fifo` 是更好的吞吐/尾延迟模式；真实 EOS 的大幅结果只作为端到端
现象，不作为固定工作量的 priority 因果结论。

### 10.6 P2 同机 baseline

P2 必须使用同机同卡 baseline；此前 Host B 的0.40595 jobs/s不能直接与 Host A 比较。
服务器 A 的严格同机结果为：

| 实现 | jobs/s | FTS/s | Job mean | P50 | P95 | P99 | preemption |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始 vLLM | 0.33876 | 5480.85 | **67.16 s** | 57.73 s | 146.64 s | **164.34 s** | 8 |
| runtime-KV f0.80 + job FIFO | **0.35840** | 4855.56 | 67.83 s | **56.50 s** | **141.40 s** | 173.06 s | **0** |

新策略的 jobs/s +5.80%、P50/P95 -2.13%/-3.58%，mean/P99 +1.01%/+5.31%，并消除8次
preemption。FTS/s下降11.41%是因为 baseline 多执行了17.50%的 prefill，其中包含
preemption重算；这里 jobs/s、延迟和重算量比“处理了多少冗余 FTS”更能反映效率。

### 10.7 产品策略结论

当前没有一个 priority 同时支配所有目标：P1 的 `job_fifo f0.80` 改善吞吐、中位数和
尾延迟，P0 fixed-work 的 `job_fifo` 改善 mean但轻微损害吞吐与尾部，P2 则改善吞吐/P95
但轻微损害 mean/P99。产品层应提供：

- `throughput_tail`：以 step locality 为主；
- `mean_jct`：以 job continuation 为主；
- `auto`：根据 runtime KV、preemption、活跃 step 的服务时间膨胀和完成速率选择，而不是
  固定按 C/R 或数据集写死。

`auto` 尚未完成 NPU 验证；在此之前，不能把 `job_fifo f0.80` 宣称为所有场景的唯一默认。

### 10.8 环境结论与原始数据

服务器 B 的5/6/7虽无计算进程，但容器内驱动预检报
`Can't get ascend_hal device count`，因此没有把该次启动计入性能数据，也没有在 B 上
盲目重试。

原始结果位于服务器 A：

```text
/data/disk/wangzili/cis-scheduler-plugin-ab-20260915
```

下一步实现 objective-aware `auto` 控制，并在8K-128K上下文迁移验证；若 P1 mean仍是
产品目标，再加入 runtime sequence/branch budget。除非结果冲突，不再对 EngineCore
架构、低于0.70的安全系数或大量固定 cap 做重复长跑。
