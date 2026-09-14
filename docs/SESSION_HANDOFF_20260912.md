# Conditional IS × mini-SWE-agent Infra 交接文档

更新时间：2026-09-12 17:15（Asia/Shanghai）

本文档用于在新的 Codex 对话中快速恢复完整背景。阅读者应先区分三类内容：

- **已经由真实实验确认的事实**：可以直接作为后续判断依据；
- **已经实现但效果为负的原型**：应保留记录，不应不加分析地重复；
- **尚待验证的设计**：不能写成已经取得的收益。

## 1. 项目目标与导师意图

### 1.1 当前研究对象

当前只研究常博仓库中的 ordinary `conditional_is` 路径，不使用
`small_proposal`。Conditional IS 的一个 block/step 可以抽象为：

```text
selected trunk
  -> 生成 C 个 candidate，每个 candidate 生成 B token
  -> 每个 candidate 派生 R 条 rollout，每条最多生成 L token
  -> 对 C x R 条完整路径计算 reward
  -> reduce/resample 选择一个 candidate
  -> 将胜出 candidate 的 B token 接入 selected trunk
  -> 进入下一个 block，直到完成最终回答
```

这整个过程发生在 MiniAgent 的**一次模型调用内部**。Docker 只执行该次调用最终
选中的 assistant/tool-call；Docker observation 进入下一轮 Agent 上下文。不要把
本项目扩展成昂贵的多轮 Agent tree search，也不要把 Docker 执行结果放入单次
Conditional IS 的内部 reward。

### 1.2 导师希望本阶段回答的问题

导师并未指定 TP/PP/P/D 的最终答案，而是指定两类硬件场景，让我们自己找出合理
部署并做类似 Nsight Systems/MindStudio Insight 的 profiling：

1. 双卡、单 instance；
2. 四卡、双 instance。

真正目标不是先把算法 accuracy 参数搜索到唯一最优，而是：

1. 先启用已经验证有效的通用部署优化，得到强 general baseline；
2. 用少量代表性算法配置观察 efficiency 特征和瓶颈是否迁移；
3. 取得原始、可复查、足够细的 profiling 数据；
4. 基于证据提出 Conditional IS 特有的 infra 优化；
5. 后续再实现最有希望的优化并严格 A/B。

因此，P0/P1/P2/P3 不是 accuracy sweep，而是 profiling probes。SWE-bench 完整
accuracy 评测和 500 题最终质量结论属于后续阶段。

### 1.3 什么算有价值的工作

仅继续扫描通用 `max_num_seqs`、`max_num_batched_tokens`、TP/PP 或普通 prefix
cache 不足以构成最终创新。这些参数仍需调好，因为算法专属 profiling 必须建立在
强 baseline 上，但最终工作应利用 Conditional IS 独有的结构，例如：

- candidate -> rollout -> reduce 的依赖图；
- 同一 candidate 下 R 条 rollout 的共享 KV；
- block barrier 和最后少数 rollout 的关键路径；
- loser/winner branch 的确定性生命周期；
- 一棵 CIS tree 在两个 instance 间按 candidate subtree 拆分时的极低 reduce 通信量。

## 2. 代码、仓库与版本

### 2.1 本地项目

主项目：

```text
/Users/li/Desktop/swe-agent/inference-scaling-mini-swe-agent
```

Git 远程：

```text
git@github.com:wangzili121/inference-scaling-mini-swe-agent.git
```

当前实验分支：

```text
cis-forest-attention
```

注意：用户明确要求分支名不要带 `codex`。当前最新已提交基础为：

```text
460e9e9 Prototype EngineCore-native CIS tree forks
```

截至本文档更新时间，Forest Attention、Forest window 和最新实验说明仍有未提交
改动。完成验证、更新文档和测试后只做**本地提交**；用户此前说暂时不要联网 push，
除非之后再次明确要求。

### 2.2 上游来源

常博最新源码快照来源：

```text
origin/main@04492fe
```

源码以快照形式导入 private 仓库，不保留上游 Git 历史，但必须保留来源 SHA。
常博原始本地仓库：

```text
/Users/li/Desktop/xchang_original
```

更早的总交接文档：

```text
/Users/li/Documents/Codex/2026-08-20/ni-q/inference-scaling-main-readme/docs/SESSION_HANDOFF_20260903.md
```

### 2.3 Runtime 与模型

当前固定：

```text
vLLM-Ascend v0.18 MRV1
Qwen3-Coder-30B-A3B-Instruct BF16
Ascend 910B3
```

模型服务器路径：

```text
/data/disk/models/Qwen3-Coder-30B-A3B-Instruct
```

Docker image：

```text
wangzili/vllm-ascend:v0.18.0-msprof1.2.2-tzdata2025.3-pandas2.2.3
```

原生 categorical runtime：

```text
/data/disk/wangzili/vllm-categorical-runtime
```

## 3. 远程实验环境

### 3.1 服务器 A

```text
SSH: wangzili@159.138.5.111
Workspace: /data/disk/wangzili/inference-scaling-mini-swe-agent-tree-test-20260912
当前实验通常使用物理卡 2,3
```

### 3.2 服务器 B

```text
SSH: wangzili@189.1.227.167
Workspace: /data/disk/wangzili/inference-scaling-mini-swe-agent-forest-test-20260912
当前实验通常使用物理卡 5,6
```

不要删除或停止其他人的容器、进程、模型或目录。只清理本项目明确创建的容器和
输出。两台机器的卡型号相同，卡间个体差异不是当前主要变量；仍需记录设备和环境，
但不应为此做大量交叉复验。

### 3.3 Workload

公开轨迹 workload：

```text
/data/disk/wangzili/cis-artifacts-fc11386/workloads/public-256
```

自采轨迹 workload：

```text
/data/disk/wangzili/cis-artifacts-629451f/workloads/self-128
```

当前正式 A/B 使用相同请求顺序、相同算法参数和固定 `run_namespace`。必须注意：
仅固定 `--seed` 不够，因为 request-local seed 还由 request id 派生；历史上自动生成
的 namespace 进入 request id，曾导致 A/B 实际采样长度不同。现在 `profile.py`
增加了 `--run-namespace`，同一 A/B 必须使用完全相同的 namespace。

## 4. mini-SWE-agent 接入状态

已经完成的主干能力：

- 导入 ordinary Conditional IS；
- 接入 mini-SWE-agent 2.4.6；
- 提供常驻 Conditional IS 推理服务和 `ConditionalISModel.query(messages)`；
- Docker 只执行最终 tool-call；
- 支持 workload freeze、burst benchmark、runtime tuner 和 topology 脚手架；
- 支持算法 trace、request trace、MS Service Profiler 和 Ascend Torch Profiler 控制；
- generation-time 统计 sequence logprob/Consilience 所需的小型信息，正式路径不再
  对完整 sequence 做重复 scoring forward；
- request-local seed 与 native categorical sampler 已接入。

仍需在最终交付前补齐或重新确认：

- 当前源码和 Docker 的完整 1 题/3 题回归；
- bash tool-call、observation 累积、EOS、超时、错误恢复和轨迹落盘；
- 最终 MiniAgent 版本、chat template 和 tokenizer manifest；
- 完整 SWE-bench Verified 质量评测。

## 5. 代表性算法配置

本阶段统一：

```text
temperature = 1
top_p = 1
top_k = None
B = 128
L = 512
```

P0/P1/P2/P3：

| ID | 配置 | 用途 |
|---|---|---|
| P0 | `C15/R3/B128/L512` + sequence-logprob | 高扇出、高压力主配置 |
| P1 | `C8/R3/B128/L512` + sequence-logprob | 中等 candidate |
| P2 | `C4/R2/B128/L512` + sequence-logprob | 低预算整体配置 |
| P3 | `C15/R3/B128/L512` + Consilience | reward/statistics 路径对照 |

Consilience 仓库参数：

```text
top_k = 5
window = 0.2
skip = 0.05
initial_penalty = 3
```

这里 Consilience 的 `top_k=5` 是 reward statistic 的内部参数，不是生成时的
sampling top-k；两者不能混淆。

这些配置当前用于效率和 profiling 对照，不代表已经完成 accuracy 选择。P2 同时
改变 C 和 R，所以只能解释为低预算整体配置，不能单独归因 rollout 数量。

## 6. 当前 general baseline

### 6.1 已启用或已接入的有效能力

当前强 baseline 包括：

- persistent AsyncLLM / AsyncLLMEngine；
- continuous batching；
- Automatic Prefix Caching（APC）；
- chunked prefill；
- request-local seed；
- generation-time reward statistics，`score_calls=0`；
- vLLM-Ascend MRV1 native categorical sampler；
- `FULL_DECODE_ONLY + Npugraph_ex`；
- Conditional IS Step cap + `step_fifo`；
- 双卡单 instance 使用 TP2 的当前胜出配置。

当前 Forest A/B 固定的 engine 参数为：

```text
TP = 2
PP = 1
max_model_len = 65536
max_num_seqs = 256
max_num_batched_tokens = 32768
gpu_memory_utilization = 0.90
max_num_partial_prefills = 1
max_long_partial_prefills = 1
```

P0/P1/P2 的 Step cap 分别为：

```text
P0: 6
P1: 16
P2: 32
```

这些是当前严格算法专属 A/B 的固定环境，不表示整个 general 参数空间已在所有
上下文长度上得到全局最优。最终报告必须把“用于当前 A/B 的固定强基线”和“已穷尽
所有部署选择”区分开。

### 6.2 Step cap 的准确含义

Step cap 不是把整个 step 变成一个 vLLM request。它做的是：

1. 一个完整 CIS step 开始前获取全局 step 槽位；
2. 同一 step 的 candidate 和 rollout 使用相同 step FIFO 优先级；
3. rollout、reward 和 resample 完成后释放槽位；
4. 底层仍是 C 个 candidate request 和 C×R 个 rollout request。

所以它更准确的名字是：

> Step 级并发窗口 + Step FIFO locality。

它能减少不同 step 的过度交错和瞬时 fanout，但不是完整 tree scheduler。

### 6.3 早期 Step/Elastic 结果

下表来自较早的一轮同环境比较，用于说明调度趋势；由于后续修复了 namespace，
不能把这些绝对值和最新 Forest A/B 跨轮直接相减。

| 配置 | 策略 | jobs/s | Job mean/P95 | Step mean/P95 | Barrier mean/P95 | Preempt |
|---|---|---:|---:|---:|---:|---:|
| P0 C15/R3 | Step cap6 | 0.1688，较扁平 baseline `+15.3%` | 134.7/246.1s | 119.5/238.9s | 22.3/43.1s | 0 |
| P0 C15/R3 | Elastic | 0.1707，`+16.6%` | 147.5/279.5s | 130.8/243.9s | 60.3/153.6s | 0 |
| P1 C8/R3 | Step cap16 | 0.2752，`+17.0%` | 97.5/173.2s | 85.2/168.3s | 29.7/75.4s | 0 |
| P1 C8/R3 | Elastic | 0.2502，`+6.4%` | 100.4/205.1s | 86.6/201.7s | 39.0/130.0s | 17 |
| P2 C4/R2 | Step cap32 | 0.3564，`+7.0%` | 64.2/151.6s | 55.3/133.7s | 16.0/54.5s | 6 |
| P2 C4/R2 | Elastic | 0.3663，`+10.0%` | 65.6/145.5s | 55.1/133.4s | 14.6/53.1s | 1 |

`Elastic` 的含义是：不固定只允许若干完整 step 展开，而是让所有已就绪 rollout
全局优先于新 candidate，空闲容量可被任意 job 回填。它在部分配置提高吞吐，但会
重新引入跨 step 交错，P0/P1 的 barrier tail 通常不如 Step cap 稳定。

`Job latency` 是一次完整模型调用从进入到最终回答完成的时间；一个 job 内含多个
step。`Step E2E` 是一个 block 从 candidate 开始到 rollout/reduce/resample 完成
的时间。`Barrier` 是主要工作完成后，被最后少数 rollout 拖住的等待时间。mean
反映平均体验，P95 反映最慢 5% 的尾部，因此三者都同时记录 mean/P95。

当前表只能证明 `Step cap + step_fifo` 整体优于扁平 baseline，尚未隔离收益有多少
只是普通外层限流。vLLM 自身已有 waiting/running queue、continuous batching、
FCFS/priority、MNS/MBT 和 KV preemption，但不知道 request 的 CIS job/step/stage/
barrier 关系。我们新增的是 CIS step semaphore 和 step 到 priority 的映射；priority
原语本身来自 vLLM。

在把 Step cap 宣称为算法专属调度收益前，需在 P0 同机器、同 namespace 补最小消融：

| ID | 外部 workers | Step cap | priority | 作用 |
|---|---:|---:|---|---|
| A | 32 | 无 | 原始 | 扁平 baseline |
| B | 6 | 无 | 原始 | 纯外层限流 |
| C | 32 | 6 | 原始 | 仅 CIS step admission |
| D | 32 | 无 | step_fifo | 仅 CIS priority 映射 |
| E | 32 | 6 | step_fifo | 当前完整方案 |

如果 E 只与 B 相当，主要收益是把 queue 从 engine 前移，工程上仍有价值但创新程度
一般；如果 E 明显优于 B/C/D，才能证明 step locality/priority 带来额外收益。比较时
必须同时看 jobs/s、FTS/s、Job/Barrier P95、engine backlog 和 preemption。

### 6.4 FULL graph 是当前真正的强基线

Forest 早期对照曾使用 `FULL_AND_PIECEWISE`，方便 Forest forward 退出图；后来确认
普通路径使用纯 `FULL_DECODE_ONLY + Npugraph_ex` 更快。严格相同 namespace 的结果：

| 配置 | 图模式 | jobs/s | FTS/s | Job mean/P95 | Preempt |
|---|---|---:|---:|---:|---:|
| P0 | FULL | 0.166006 | 2896.83 | 151.51/292.47s | 0 |
| P0 | FULL_AND_PIECEWISE control | 0.146693 | 2602.53 | 134.85/274.83s | 0 |
| P1 | FULL | 0.243368 | 3708.91 | 92.89/171.75s | 0 |
| P1 | FULL_AND_PIECEWISE control | 0.214017 | 3307.84 | 105.49/183.94s | 0 |
| P2 | FULL | 0.319405 | 5364.44 | 97.59/166.00s | 2 |

P0/P1 中 FULL 的 jobs/s 分别比 dual graph control 高约 13.2%/13.7%，FTS/s 高
约 11.3%/12.1%。因此任何 Forest Attention 最终实现都必须保留或重新 capture
FULL graph；只和 dual graph control 打平不能算最终成功。

FTS/s 指 generation forward token slots/s，适合在生成长度和重采样路径略有变化
时衡量单位模型工作的执行能力。jobs/s 是最终端到端吞吐，两者都要报告。

## 7. Profiling 已确认的核心瓶颈

### 7.1 设备已基本被喂满

P0 decode batch 长时间接近 256；P1 的 P50/P90 也曾达到 255/256。继续无条件增加
MNS 或外部请求并发不是主要创新方向。打满的定义是：增加并发后 jobs/s 提升不足
3%，同时 scheduler 持续存在 backlog，而不是制造无限深的等待队列。

### 7.2 高扇出改变执行形状和单位工作效率

代表 profile 中：

- P0 C15/R3 约 3237 forward-token-slots/s；
- P1 C8/R3 约 4396 forward-token-slots/s；
- P0 单位 forward work 效率低约 26%；
- P0 Prefill+Decode 混合批占比约 72%，P1 约 21%；
- 64 个 P0 CIS job 曾展开约 2175 个 engine request，最大 in-flight 约 474。

具体绝对值会随版本和 fixed namespace 变化，但“高 fanout -> 更深 request 展开 ->
更多混合 batch 和 barrier 干扰”这一趋势稳定存在。

### 7.3 普通 prefix cache 和 reward 已不是主瓶颈

- APC token hit ratio 通常约 95%-97%；
- P0 某次统计复用约 3813 万 token，实际 prefill 约 143 万；
- 当前 reward 路径 `score_calls=0`；
- reward/weight/resample 的累计 Host 阶段占比低于 0.01%。

APC 只避免重复 prefill 计算。它不会让多条活跃 rollout 在每个 decode token 上只读
一次共享 KV；每条 query 的 attention 仍会读取自己的逻辑 block table。因此高
APC 命中与 Forest Attention 机会并不矛盾。

### 7.4 Attention 是最大的设备热点

Ascend Torch trace 中 `FusedInferAttentionScore` 通常占 NPU busy 时间约
38%-75%，最近代表 mixed 窗口约 67.3%。HCCL 在部分四卡窗口暴露约 10%-19%，
但它更 general，当前优先级低于共享 KV attention。

### 7.5 Sibling 已经高度同批，问题不只是“没分组”

逐请求 Service trace 发现：

- 同一 candidate 的 R 条 rollout 约 99.68% 在同一个 forward batch 中；
- 但 93.26% 的相关 batch 是 Prefill+Decode 混合批；
- 代表 block 的 rollout 输出长度 P50/P95 约 148.5/384 token；
- 同一 block rollout 完成跨度约 180.65s。

这排除了“只把 sibling 排到一起就会获得主要收益”的简单解释。真正机会是：

1. attention kernel 识别同批 sibling 的共享 KV；
2. scheduler 在确有完整 sibling group 时，适度减少新 prefill 对 decode 形状的干扰；
3. 根据在线剩余工作缩短最后少数 rollout 的 barrier critical tail。

### 7.6 KV preemption 的正确解释

KV preemption 表示正在运行的 sequence 因 KV 空间压力被移出，之后需要重算或恢复。
它通常增加额外工作和尾延迟，是重要告警，但不能机械地规定“出现一次就一定更差”。
如果更高 MNS 带来的并行收益超过 recompute 成本，端到端吞吐仍可能更高。判断顺序
应是 jobs/s、FTS/s、P95 和 recompute/preemption 成本一起看；preemption 为零是
优选条件，不是脱离端到端结果的唯一目标。

## 8. 已尝试的 Conditional IS 专属调度/KV 方向

### 8.1 Candidate completion streaming：负结果

做法：candidate 完成回调后立刻提交它的 R 条 rollout，不再等待所有 C 个 candidate。

结果：P0 jobs/s 下降约 9.78%，P95 从约 335.40s 增至 367.35s。固定 B 的
candidate 本来就几乎同时完成，candidate-to-rollout 中位提前量只有约 5.85ms；
小批、碎片化提交和 KV 压力大于 barrier 提前收益。

### 8.2 Per-job bounded rollout / sibling subtree batching：负结果

做法：同一 candidate 的 R 条 rollout 作为小 bundle，限制每棵树同时活跃的 bundle。

结果：代表 P0 K8 相对 Step control 的 jobs/s 下降约 17.8%；另一轮 per-job
bounded jobs/s 下降约 12.79%，P95 增至约 399.17s。原因是每棵树独立限流会产生
小 batch，不能观察整个 engine 的可运行工作。

### 8.3 Global frontier：无稳定收益

全局 rollout credit 取 128/192/240 的 P0 A/B，work-normalized throughput 分别
约为 `-4.08%/+0.78%/-3.38%`。192 的微小正值伴随 47 次 preemption 和更差 P95，
不保留。

### 8.4 Whole-job work-aware routing：仅部分配置有效

静态依据 prompt、C/R/B/L 估计整棵树工作并做 LPT 路由，在 P1 的
work-normalized throughput 曾提升 22.65%，但 P3 下降 11.39%。随机 EOS 和
resample 会改变真实剩余工作，静态估计不能成为默认路由。若继续，应改成 online
remaining-work routing，而不是继续调静态公式。

### 8.5 Direct KV fork 与 EngineCore-native fork

Direct KV fork 让新 rollout request 直接继承 candidate 的物理 block table。它
证明物理 KV 交接可行，但在 APC 已经 95%-97% 时，相对最佳 Step 调度只有约 1%
波动，未形成稳定收益。

之后实现了更深入的 EngineCore branch-on-token 原型：rollout waiter 预注册，parent
到达 B 后在 scheduler 内激活并 fork block table；compact waiter 让 3150 个 child
只发送单 token sentinel，避免约 4300 万 prompt token 的 IPC/占位 payload。

严格 P0 A/B 中，它相对 Step control：

- jobs/s 下降约 3.6%；
- FTS/s 下降约 5.9%；
- Job P95 恶化约 37.0%；
- Step P95 恶化约 24.9%。

结论：APC 已消除大部分 readmit prefill，Host request lifecycle 不是当前最大瓶颈。
不要继续仅围绕 placeholder identity 做复杂 persistent tree-root，除非新 trace 显示
其单独占比超过 5%。

### 8.6 Branch-aware KV 生命周期：负结果

vLLM 请求完成时释放 refcount；完整 block 可继续作为 APC 条目留在 free queue，
内存压力时再按 LRU 淘汰。CIS 的 loser branch 的确有“永不再用”的确定语义，但
只有 refcount 为零的 block 才能调整淘汰优先级，不能强制删除活跃共享 trunk。

P2 压力 A/B 中，rollout-tail demotion 实际移动 108 个请求的 175 个纯独占 block，
但 jobs/s 下降 3.7%，Job P95 增加 8.5%，preemption 从 1 增至 4，不保留。candidate
suffix 300 秒租约把 direct fork 命中率从 77.9% 提到 83.1%，jobs/s 仍无提升。

## 9. Forest Attention：当前最有技术价值的方向

### 9.1 问题定义

同一 candidate 下的 R 条 rollout 具有：

```text
[shared selected trunk + candidate suffix][rollout unique tail]
```

APC 可以让后代 admission 时复用已经存在的物理 KV block，但普通 decode paged
attention 仍为每条 rollout query 分别读取共享的长前缀。Forest Attention 希望在
一个 attention op 中让 sibling query 共享这些 KV 读取，并在数学上保持 exact。

### 9.2 重要纠正：真实 baseline 是私有 paged attention

早期 microbenchmark 使用通用 `npu_fused_infer_attention_score`（FIA）作为
baseline。后来检查 vLLM-Ascend v0.18 代码和 trace，确认正常 pure decode 使用更快
的私有：

```text
torch_npu._npu_paged_attention
```

因此，只有和 `_npu_paged_attention` 比较的结果才是可信的端到端方向判断。早期
相对通用 FIA 的巨大 speedup 不能直接写成产品收益。

### 9.3 当前 packed bridge

每个 candidate 的 sibling group 构造逻辑 KV 行：

```text
[shared candidate prefix][tail 0][tail 1]...[tail R-1]
```

将 R 个 query 作为连续 BSH query，mask 使 query r 只看共享 prefix 和 tail r。
它用一次 FIA 计算 sibling group，不需要三段 FIA 和 Host LSE merge。普通请求仍走
原生 paged attention；不完整 sibling、PCP/DCP/spec decode/sliding window 等情况
自动回退。

### 9.4 与 native paged attention 的 microbenchmark

不同完整 sibling group 数下，packed BSH 相对 native PA 的代表结果：

| 场景 | 组数/请求数 | speedup |
|---|---:|---:|
| P0, prefix 9728, unique 128 | 22 groups / 66 queries | 1.915x |
| P0, prefix 9728, unique 640 | 22/66 | 1.745x |
| P1-like, prefix 16384, unique 640 | 11/40 | 1.037x |
| P0, prefix 24576, unique 512 | 15/45 | 1.954x |

更细的规律：

- sibling group 少时 Forest 往往更慢；
- 同时存在较多完整 group、共享 prefix 足够长时可达约 1.1x-1.9x；
- P2 R2 在 16K/32K 的许多形状仍低于 1x；
- 必须依据 `shared_tokens × (R-1) × complete_groups` 动态门控。

当前门控阈值大致为：

```text
shared prefix >= 4096 tokens
estimated saved KV reads >= 196608 token-reads
packed query fraction >= 0.75
padding ratio <= 1.25
masked/saved ratio <= 1
```

### 9.5 v2 端到端：明确负结果

严格相同 namespace 的 64-request 结果：

| 配置 | 路径 | jobs/s | FTS/s | Job mean/P95 |
|---|---|---:|---:|---:|
| P0 | dual graph control | 0.146693 | 2602.53 | 134.85/274.83s |
| P0 | Forest v2 | 0.138433 | 2429.69 | 170.56/314.20s |
| P1 | dual graph control | 0.214017 | 3307.84 | 105.49/183.94s |
| P1 | Forest v2 | 0.182402 | 2814.62 | 104.13/214.42s |

P0 jobs/s `-5.6%`、FTS/s `-6.6%`；P1 jobs/s/FTS/s 都约 `-14.8%/-14.9%`。

定位到的主要额外开销：每个 decode step 在 Python 中构造巨大的 nested bool mask，
再 H2D 传输约 50 万到 100 万以上 bool；mixed batch 还要拆成 native PA 和 FIA 两条
路径，并做 gather/reshape/scatter。

### 9.6 v3：去掉 Host mask 后追回大量损失

v3 不再在 Python materialize 巨型 mask，只传 shared/tail lengths，在 NPU 上使用
缓存的 `arange` 向量化生成 mask；all-Forest batch 使用 direct-return fast path。

同 namespace 4-job P0 smoke：

| 版本 | jobs/s | FTS/s | Job mean/P95 |
|---|---:|---:|---:|
| v2 | 0.123168 | 1846.01 | 23.03/32.17s |
| v3 | 0.137397 | 2041.37 | 16.67/27.02s |

v3 相对 v2：jobs/s `+11.6%`、FTS/s `+10.6%`、mean `-27.6%`、P95 `-16.0%`。

最新 P0 64-request v3：

```text
wall_seconds = 447.085
jobs/s = 0.143149
FTS/s = 2540.65
Job mean/P95 = 161.22/249.15s
preemption = 0
Forest activations > 500
```

对 dual graph control：jobs/s `-2.4%`、FTS/s `-2.4%`、mean `+19.6%`、P95
`-9.3%`。对真正 FULL baseline：jobs/s `-13.8%`、FTS/s `-12.3%`、mean `+6.4%`、
P95 `-14.8%`。

结论：去掉 Host mask 是真实改进，Forest attention 的 kernel 机会存在；但当前
bridge 仍输给 FULL graph，尚不能作为默认优化。

最新 P2 v3 64-request：

```text
wall_seconds = 177.600
jobs/s = 0.360361
FTS/s = 4865.12
Job mean/P95 = 88.96/152.68s
preemption = 0
```

P2 的 R=2 和 saved-read 阈值使 Forest 基本未激活，结果主要体现 dual graph 路径。
它的 jobs/s 高于 P2 FULL 不能直接解释为 Forest 收益，因为实际 forward work 更少；
FTS/s 比 FULL 低约 9.3%，符合“未命中 Forest 时丢掉 FULL graph”的判断。

P1 v3 已在服务器 A 补跑 64-request，但严格门控下没有一次 Forest activation，
因此它只能说明当前阈值会保护不够密集的 P1 batch，不能作为 Forest 加速结果。该轮
与 P1 FULL/dual control 还位于不同服务器，不应做精确百分比比较。若开发出可
graph-capture 的 CANN op，必须在同一机器重跑 P0/P1/P2 严格 A/B。

## 10. 为什么原生 attention 不影响图，而当前 Forest 会影响图

原生 `_npu_paged_attention` 在 vLLM-Ascend 中已经是模型 forward 的稳定算子：

- 输入协议固定；
- graph capture buckets 已知；
- block table、slot mapping 等张量由框架按固定结构提供；
- forward 不需要 Python 根据请求语义改变算子拓扑；
- 同一个 graph 可以按已 capture 的 batch shape replay。

当前 Forest bridge 在每个 decode step 需要：

1. 读取 request metadata，判断哪些 branch 属于同一 candidate；
2. 检查完整 sibling、真实物理共享 block 和动态收益阈值；
3. 根据当步 group 数、R、prefix/tail 长度构造 packed block table 和 mask；
4. 在普通 `_npu_paged_attention` 与通用 FIA 之间做 Python 控制流分支；
5. mixed batch 时拆分 native/Forest 子集，再 gather/reshape/scatter 合并。

这些动态结构和不同算子路径不属于原有 FULL graph 的稳定签名，所以命中 Forest 的
forward 当前必须退出 FULL graph，走 `FULL_AND_PIECEWISE` 中的 piecewise/eager
路径。即使 Forest attention 本身省了 KV 读取，丢图和桥接开销也会吃掉收益。

正确的正式实现不是继续优化 Python mask，而是做一个可 capture 的融合 CANN op：

- 固定上界、bucket 化的输入形状；
- 输入紧凑 segment metadata，而非 materialized bool mask；
- 在 op 内处理 shared prefix、candidate suffix 和 unique tail；
- 在 kernel 内完成 exact online softmax/LSE merge；
- normal/Forest 选择尽量在 op 内或 graph bucket 层完成；
- 普通 batch 继续复用原生 PA fast path；
- 只在 branch topology/EOS 变化时更新 metadata，不在每 token 重建 Python tree。

只有这样才有机会同时保留 FULL graph 的约 12% 优势和 Forest 的 KV-read 节省。

## 11. Step 分组是否“没做好”的判断

用户的直觉部分正确：现有 Step cap 并没有把完整 step 或 tree 变成一等调度对象。
但已有 trace 又表明 sibling 约 99.68% 已同批，所以“更严格地把整组绑死”不是当前
主要缺口。更完整的 scheduler 应区分：

- **tree state**：依赖、parent/child、remaining children、barrier、KV ownership；
- **admission unit**：完整 candidate 的 sibling rollout bundle；
- **execution unit**：实际进入 attention batch 的 ready branch；
- **global batching**：从多棵树合批，不能让一棵树独占设备。

不应简单把一个完整 step 做成不可打散的巨型 request。这样虽然语义清晰，但容易：

- 一个 P0 step 瞬间要求 15+45 条 branch；
- KV 空间不足时更难调度；
- rollout EOS 不同导致组内大量空洞；
- 同时只有少数 tree 时 decode batch 很快欠载；
- 牺牲 vLLM continuous batching 的全局填充能力。

当前更窄的 scheduler 假设是 **Forest decode window**：只有当当前 running decode 中
已经存在足够多完整 sibling group，预计 Forest saved reads 过阈值时，短暂阻止新
waiting prefill 进入 1-N 个 scheduler tick，形成更纯的 sibling decode batch；达到
最大连续窗口后必须放行一次 waiting work，避免饥饿。

该方案不是“完整 tree scheduler”，而是一个针对已观察到 `93.26% mixed batch` 的
小型因果实验。代码已写入：

```text
infra/vllm_ascend/cis_forest_attention/vllm-0.18-cis-forest-window.patch
```

环境变量：

```text
VLLM_CIS_FOREST_WINDOW_MAX_STEPS
VLLM_CIS_FOREST_WINDOW_MIN_SAVED_READS
VLLM_CIS_FOREST_WINDOW_MIN_PACKED_FRACTION
```

第一轮 P0 smoke 在 2026-09-12 16:43 启动后因启动命令漏传 `CIS_WORKSPACE`，runner
回落到旧 workspace，旧 `profile.py` 不含 `--run-namespace`，因此退出且未产生性能
结果。该问题与 Forest scheduler 性能无关。重跑时必须显式传入对应服务器的当前
workspace；runner 也已增加实际 import path 输出。不要把该失败写成性能负结论。

修正启动路径后的同 namespace 结果：

| workload | 路径 | jobs/s | FTS/s | Job mean/P95 |
|---|---|---:|---:|---:|
| P0 4-job smoke | Forest v3 | 0.137397 | 2041.37 | 16.67/27.02s |
| P0 4-job smoke | Forest v3 + window4 | 0.203629 | 2962.55 | 14.39/19.02s |
| P0 64-request | Forest v3 | 0.143149 | 2540.65 | 161.22/249.15s |
| P0 64-request | Forest v3 + window4 | 0.142763 | 2537.63 | 150.65/265.25s |

小 smoke 曾表现为 jobs/s/FTS/s 约 `+48%/+45%`，但饱和 64-request 中 jobs/s/FTS/s
为 `-0.27%/-0.12%`，基本完全持平；mean 降约 6.6%，P95 反而升约 6.5%。整轮
64-request 只记录到一次 scheduler window activation，不能稳定改变整体 batch
结构。小 smoke 的巨大差异来自短运行的 batch/完成顺序和生成长度敏感性，不能外推。

因此 Forest window 当前判定为**无吞吐收益的负结果**，停止扩展和网格搜索。保留
patch、日志和设计说明作为证据；不要在融合 CANN Forest op 出现前继续开发完整 tree
scheduler。只有融合 op 已经在 FULL graph 内产生稳定收益，且 trace 仍显示 mixed
prefill/decode 或 barrier tail 单独损失至少 5%，才重新开启 tree-aware scheduler。

## 12. 推荐的最终产品形态

实验代码可以复杂，最终产品应收敛为三个窄组件：

```text
Conditional IS adapter
  -> 暴露 job/step/candidate/rollout/parent/segment metadata

CIS-aware scheduler hook
  -> bounded tree state
  -> optional Forest decode window
  -> online barrier/remaining-work signal

Ascend Forest Attention CANN op
  -> compact segment metadata
  -> exact shared-prefix attention
  -> graph-capturable dynamic gate
```

长期四卡方向可增加 candidate-subtree parallel：一棵 P0 tree 的 candidate 0-7 放到
instance A，8-14 放到 instance B；candidate 和自己的 rollout 不迁移，reduce 时只
交换 C 个 log-weight，winner 后只同步 B 个 token。该方向能让单个重型 CIS job
使用四卡，且通信量很小，但应排在双卡 kernel/scheduler 证据之后。

## 13. 当前文件与实现位置

核心源码：

```text
src/inference_scaling/arllm/algorithms/conditional_is.py
src/inference_scaling/arllm/backends/vllm_backend.py
src/inference_scaling/arllm/backends/packed_forest_attention.py
src/inference_scaling/arllm/types.py
src/inference_scaling/swe_agent/profile.py
```

实验与 runtime patch：

```text
experiments/swebench/profile_prefix_forest_attention.py
experiments/swebench/run_cis_forest_ab_remote.sh
infra/vllm_ascend/cis_forest_attention/vllm-ascend-0.18-packed-forest-attention.patch
infra/vllm_ascend/cis_forest_attention/vllm-0.18-cis-forest-window.patch
infra/vllm_ascend/cis_forest_attention/README.md
```

测试：

```text
tests/test_packed_forest_attention.py
tests/test_vllm_backend.py
```

设计与实验记录：

```text
docs/design/CIS_FOREST_RUNTIME.zh-CN.md
docs/experiments/CIS_PACKED_FOREST_ATTENTION_20260912.zh-CN.md
```

已有本地交互式 profiling 可视化：

```text
/Users/li/.codex/visualizations/2026/09/03/01a066ff-41e4-7ba1-b0ee-4b46d3e58ffc/cis-profile/cis-profile-dashboard.standalone.html
/Users/li/.codex/visualizations/2026/09/03/01a066ff-41e4-7ba1-b0ee-4b46d3e58ffc/cis-profile/cis-request-flow.standalone.html
```

用户对可视化的要求是：不能只有聚合柱状图；应像 MindStudio Insight 一样从左到右
展示时序，能看到 job/step/candidate/rollout、worker/instance、waiting/running、
prefill/decode、attention/HCCL、batch 组成和设备 busy/idle，并在图上直接高亮问题。

## 14. 最新实验 artifact 路径

总根目录：

```text
/data/disk/wangzili/cis-forest-attention-20260912
```

关键结果：

```text
P0 FULL baseline:
  p0-fixed-full-control-64

P0 dual graph control:
  p0-fixed-v2-control-64

P0 Forest v2:
  p0-fixed-v2-forest-64

P0 Forest v3:
  p0-fixed-v3-mask-forest-64

P1 FULL baseline:
  p1-fixed-full-control-v2-64

P1 dual graph control:
  p1-fixed-dual-control-64

P1 Forest v2:
  p1-fixed-v2-forest-64

P1 Forest v3 gate-off（服务器 A，不与 B 跨机精确比较）:
  p1-fixed-v3-mask-forest-64-v2

P2 FULL baseline:
  p2-fixed-full-control-64

P2 Forest v3 / dual graph gated path:
  p2-fixed-v3-mask-forest-64

P0 v2/v3 same-namespace smoke:
  fastpath-p0-smoke4
  v3-mask-p0-smoke4

Forest window failed launch artifact:
  p0-forest-window4-smoke4

Forest window fixed smoke/full:
  p0-forest-window4-smoke4-v3
  p0-fixed-forest-window4-64
```

每个完整结果通常包含：

```text
benchmark.json
artifact-manifest.json
algorithm-traces/profile-0.jsonl
request-traces/profile-0.jsonl
logs/profile-0.log
run-environment.txt
launch-command.txt
warmup.json
```

大模型、Docker image 和原始大 profile 不提交 Git。Git 只保存报告、派生 CSV/JSON、
图表、配置、SHA256、服务器路径和复现命令。

## 15. 当前测试状态与工作区注意事项

最近一次完整 targeted test：

```text
135 passed, 2 warnings
```

同时已通过：

- `git diff --check`；
- runner bash syntax；
- Python `py_compile`；
- vLLM v0.18 scheduler patch dry-run/apply/compile；
- vLLM-Ascend v0.18 Forest patch dry-run/apply/compile。

曾发生一次 rsync 路径错误，把若干源码 basename 放到远程 workspace 根目录，其中
`types.py` 会遮蔽 Python 标准库。已经只删除本项目误传的这些根目录文件，并改用
`rsync --relative`。后续同步必须继续保留相对路径，不能再把 basename 平铺到根。

当前 Git 工作区有用户任务相关的未提交改动，不要 reset/revert。提交前必须：

1. 等待所有本项目实验容器结束；
2. 更新 Forest README 和中文实验记录中的过时 BNSD/旧 microbenchmark 描述；
3. 把 v3 和 FULL graph 结果写入文档；
4. 跑完整测试、`git diff --check` 和 patch dry-run；
5. 检查没有模型、原始巨型 profile 或 accidental root files 被 add；
6. 做本地 commit，除非用户明确要求，否则不 push。

## 16. 下一对话的建议执行顺序

### P0：补 Step cap 的外层限流消融

Forest window smoke 和 64-request 已完成；饱和负载无吞吐收益，不再扩展。下一项必要
的调度实验不是新策略，而是第 6.3 节的 A-E 消融，用于回答 Step cap 收益是否只是
普通 workers 限流。先用 16-request 检查实现，再在同机器、同 namespace 的饱和
P0 上完成 A/B/C/D/E；不要跨服务器混算百分比。

### P1：不要误读 gate-off 结果

P0 v3 已接近 dual graph control但仍输 FULL；P1/P2 在当前严格门控下基本未激活。
若目标只是判定 Python bridge，现有证据已经足够。只有 Forest window 或新的
graph-capture 改造出现正收益，才在同一机器补 P1 FULL/control/Forest；否则把算力
投入 CANN op 更合理。

### P2：文档收敛

更新：

```text
docs/experiments/CIS_PACKED_FOREST_ATTENTION_20260912.zh-CN.md
infra/vllm_ascend/cis_forest_attention/README.md
docs/design/CIS_FOREST_RUNTIME.zh-CN.md
```

明确写出：native PA baseline、v2 负结果、v3 recovery、FULL graph gap、P2 gate-off，
不要把通用 FIA microbenchmark 写成端到端收益。

### P3：正式 CANN Forest Attention 立项门槛

只有在以下条件满足时进入正式 kernel：

1. trace 中完整 sibling group 和长共享 prefix 占比足够；
2. native PA microbenchmark 在真实 batch shape 上有稳定潜力；
3. v3 已证明 Host mask 是主要可消除开销；
4. 设计能在 FULL graph 内 capture；
5. 数值逐 token 对齐，采样/seed/rollout 集合不变。

第一版 kernel 目标不应宣称 1.9x 端到端。更合理的验收是：相对所有已有优化开启的
原始 Conditional IS，在 P0 真实饱和 workload 上 jobs/s 或 P95 至少改善 5%，且
FTS/s 同向、无质量回退；最终项目目标仍是至少 10%。

### P4：之后才考虑更完整 tree scheduler

若 fused Forest op 已成立，再实现：

- compact tree metadata；
- sibling bundle admission；
- Forest-benefit-aware decode window；
- online remaining-work estimate；
- barrier-critical priority；
- 跨 tree batching；
- 四卡 candidate-subtree parallel。

不要再次先实现一个庞大的全功能 tree scheduler，再寻找它能解决的问题。

## 17. 当前最短结论

1. Step cap 是有效强 baseline，但只是 step 并发窗口和 FIFO locality，不是真正 tree
   scheduler。
2. 单纯 candidate streaming、per-job bundle、global frontier、EngineCore fork 和
   loser KV demotion 都没有稳定端到端收益，不能重复包装成创新。
3. APC 已解决绝大部分重复 prefill；剩余最有价值的结构性问题是 sibling rollout
   在 decode attention 中反复读取同一长 candidate KV。
4. Forest Attention 在真实 native PA microbenchmark 的密集 sibling shape 上有
   1.1x-1.9x kernel 潜力。
5. Python/FIA bridge v2 明显回退；v3 去掉 Host mask 后接近 dual graph control，
   证明优化方向有内核价值，但仍因退出 FULL graph 比真正 baseline 慢约 12%-14%。
6. Forest decode window 的饱和 P0 jobs/s/FTS/s 为 `-0.27%/-0.12%`，没有吞吐
   收益；小 smoke 的约 45% 假象未复现，停止扩展完整 tree scheduler。
7. 下一核心工作应是 graph-capturable CANN Forest Attention，而不是继续堆 Host
   调度规则；Step cap 的外层 workers 消融已完成，结果见第 18 节。

## 18. 2026-09-13 Step 外层限流与 Priority 消融已完成

第 16 节 P0 所列 A-E 已在服务器 `159.138.5.111` 的空闲 NPU 1/4 上完成。没有停止、
覆盖或共用其他人的卡。固定 P0、TP2、64-request workload、v0.18 和同一组 engine
参数，测试：

```text
A flat W32
B flat W6
C cap6 W32
D step_fifo priority W32
E cap6 + step_fifo W32
```

并额外复跑 A/D/E。最关键结果：

- B 相对 A：jobs/s `-2.3%`，证明纯外层 workers 限流不能复现收益。
- C 相对 A：jobs/s `+2.7%`，0 preemption，barrier P95 `-64.3%`，但 burst P95
  `+10.4%`。cap-only 是稳定性控制，不是主要吞吐来源。
- D 两轮相对 A：jobs/s `+13.0%/+24.1%`，generated tokens/s
  `+7.7%/+10.4%`。主要吞吐收益来自 CIS `job+step -> priority` 映射，而不是 cap；
  代价是 preemption 16/4，engine queue P95 上升。
- E 两轮相对 A：jobs/s `+5.5%/+11.1%`，Job P95 `-24.9%/-34.1%`，barrier
  P95 `-75.3%/-71.0%`，两轮 0 preemption。它是尾延迟/稳定性方案，不是最高吞吐
  方案。
- A/D/E 两轮等权平均：D jobs/s `+18.5%`、generated tokens/s `+9.1%`；E
  jobs/s `+8.3%`、generated tokens/s `+0.1%`、Job P95 `-29.7%`、barrier P95
  `-73.2%`。

统计注意：benchmark 原 `Job latency` 从客户端线程实际发 HTTP 时开始，W6 会隐藏
尚未获得 worker 的请求。汇总器已增加以最早 worker start 为共同近似到达时刻的
`burst_latency_seconds`，用于公平比较 W32/W6。以后 benchmark 应直接保存精确
`released_at`。

当前准确边界：vLLM 自带的是 per-request priority 原语；本仓库新增的是 CIS step
元数据、step 生命周期 admission 和 `job_id + step_id` priority 映射。因此不是
漏开的框架开关，但也还不是完整 tree scheduler。产品化应提供 throughput priority-only
模式和 tail-safe cap+priority 模式，后续再用动态 policy 统一二者。

详细报告：

```text
docs/experiments/CIS_STEP_ADMISSION_PRIORITY_ABLATION_20260913.zh-CN.md
```

原始数据：

```text
/data/disk/wangzili/cis-step-ablation-20260913/full
/data/disk/wangzili/cis-step-ablation-20260913/confirm
```

下一步不应再重复固定 cap sweep。针对调度的最小后续是用统一 trace 解释 D 的吞吐
提升与 preemption 上升，以及 E 的 barrier 收益和设备利用率损失；更高技术含量的
主线仍是能进入 FULL graph 的 CANN Forest Attention。

## 19. 2026-09-13 Dynamic Tree Scheduler 实验已完成

当前本地分支为 `cis-dynamic-scheduler`。本轮先新增结构化 `CISRequestMetadata`，将
job、step、candidate、rollout、C/R 和准确的 step rollout 总数传入 vLLM
`SamplingParams.extra_args`；`step_fifo` 优先读取结构化元数据，旧 request-ID regex
只作为兼容 fallback。实验源码入口：

```text
src/inference_scaling/arllm/backends/cis_scheduler.py
experiments/swebench/run_cis_forest_ab_remote.sh
```

完成三类策略：

1. `step_cohort`：相邻 N 个 step 共用 priority，不 hard cap；
2. `step-tree-adaptive`：EngineCore 保留 deferred tree，依据 running/KV/preemption
   将窗口从保守起点逐步扩张；
3. `step-tree-fractional`：按剩余 rollout 分支比例归还 step budget，并测试普通
   queue/KV 安全门控。

固定 P0、TP2、MNS256、MBT32768、memory0.90、64 requests/workers32 的核心结果：

| 策略 | jobs/s | generated tok/s | FTS/s | Job P95 | Barrier P95 | preempt |
|---|---:|---:|---:|---:|---:|---:|
| flat 两轮均值 | 0.1539 | 552.7 | 3904 | 327.6 s | 173.0 s | 0.5 |
| priority 两轮均值 | 0.1824 | 602.9 | 3428 | 276.1 s | 153.6 s | 10 |
| fixed cap6 | 0.1667 | 553.4 | 2901 | 230.4 s | 46.4 s | 0 |
| fixed cap16 | 0.1944 | **630.7** | 3382 | 234.0 s | 112.8 s | 0 |
| EngineCore adaptive 6->16 | **0.1988** | 612.1 | **3432** | **227.0 s** | **94.9 s** | 0 |
| fractional cap16 | 0.1884 | 611.9 | 3493 | 294.3 s | 150.2 s | 2 |
| fractional + safe gate | 0.1793 | 595.1 | 3367 | 287.0 s | 153.0 s | 0 |

解释边界：adaptive 相对 fixed cap16 只有 jobs/s `+2.2%`、FTS/s `+1.5%`、Job P95
`-3.0%`，generated-token/s `-3.0%`；较明确的增量是 Barrier P95 `-15.9%`。
因此它是弱 Pareto 点，不足以证明当前动态规则有明显产品价值。fractional 两版均使
Job/Barrier tail 明显恶化；无门控版 KV 99.27%并有 2 次 preemption，安全门控版
虽零 preemption，KV 仍达 97.66%，说明当前 KV 水位无法预测新 tree 后续增量。

P1 `C8/R3`：Cohort11 无收益；EngineCore adaptive 11->22 为 0.2583 jobs/s、
FTS4058、Job P95 183.3s、Barrier P95 88.1s、零 preemption，优于 flat 但不如已测
fixed cap16 的 0.2752 jobs/s、FTS4150、Job P95 173.2s、Barrier P95 75.4s。
P2 在 workers32 下自动窗口/cohort 均为 32，策略退化为已有 priority，未重复占卡。

另一个语义缺口：P0/P1 中分别有 22/31 个 all-terminal candidate step 不产生 rollout。
当前 scheduler 用 5 秒 idle grace 回收；trace 证实本轮所有 expired step 后续均未
出现 rollout，但正式版本应由算法显式发 `step_closed` 或使用准确 finish reason，
不能依赖超时。

准确结论：

- vLLM priority 是现成原语；CIS 元数据、step 映射和 EngineCore tree admission 是
  本仓库新增，不是漏开的框架选项。
- 固定 cap16 能复现 dynamic 的大部分 P0 收益；不要把当前实现描述成成熟的新
  scheduler。
- Host 侧 cohort、瞬时 running/KV 窗口、按分支归还预算都已得到反例，不继续调
  水位或堆规则。
- 当前可用基线仍是 P0 fixed cap16、P1 fixed cap16、P2 elastic；实验 scheduler
  只保留在 variant 中，不默认启用。
- 若重启动态调度研究，前置条件是显式 tree lifecycle、可校准的增量 KV/remaining
  token 模型，并用同 workload 证明超过 fixed cap16。

完整报告与派生数据：

```text
docs/experiments/CIS_WORK_CONSERVING_COHORT_SCHEDULER_20260913.zh-CN.md
docs/experiments/CIS_DYNAMIC_TREE_ADMISSION_20260913.zh-CN.md
docs/experiments/data/cis_dynamic_tree_admission_20260913.json
```

派生数据 SHA256：

```text
f9278b1cf1c84a37226b753736961f711be8342e04e1938934c735982c58543e
```

原始数据根目录：

```text
/data/disk/wangzili/cis-dynamic-scheduler-20260913
```

服务器 `159.138.5.111` 的实验均只使用启动前空闲的物理 NPU 1/4；结束后容器已退出，
卡已释放。第二台服务器检查时只有单张 7 号卡空闲，未占用。

## 20. 2026-09-14 Predictive Continuation 实验

当前本地分支为 `cis-predictive-scheduler`。在上一节 EngineCore dynamic 基础上完成：

- 用 candidate `RequestStatus` 准确识别 terminal candidate，推导实际 rollout 数；
- all-terminal step 立即关闭，不再依赖 idle timeout；
- 实现 continuation reservation/claim/expire 和 work-conserving release；
- 增加 `job_fifo` 消融与 performance-only fixed-work 模式；
- P1 fixed-work 保证各方案均为3328个 engine request、720896个 generated token。

P1 相对静态 `step_fifo + cap16` 的关键结果：

| 策略 | jobs/s 变化 | Job mean 变化 | Job P95 变化 | prefill |
|---|---:|---:|---:|---:|
| job_fifo + cap16 | -0.22% | -2.33% | -0.47% | 627240 |
| EngineCore，无 continuation | -10.24% | +54.25% | +10.92% | 1688232 |
| work-conserving release | -0.78% | +11.40% | +0.03% | 707368 |
| continuation 5s | **+0.73%** | **-4.67%** | **-1.44%** | **627240** |

无 continuation 会先处理所有 job 的 step0，再处理 step1，导致跨 step prefix 重算；
5秒 continuation 的96次 reservation 全部 claim，恢复与静态基线完全相同的 prefill
和总 FTS，并形成小幅 Pareto 改善。work-conserving 版本释放4个长上下文 reservation
后恰好多出约80K prefill，证明只看 running request 数量会低估换树成本。

P0 `C15/R3` 迁移失败：5秒 continuation 相对静态 cap16 的 jobs/s `-15.20%`、
FTS/s `-12.24%`、Job P95 `+3.06%`；53次 reservation 只有11次 claim、42次
timeout。固定 grace 随 C/R 改变，不具备通用性。P2 的窗口32等于 workers32，
continuation 不改变 admission 集合，未重复占卡。

最终决策：当前生产 baseline 仍是静态 CIS `step_fifo/cap`；新动态代码只作为研究
variant，不默认启用。若继续，必须由算法层显式发送 `step_closed/next_step_ready`，
并将 prefix bytes、剩余 rollout token 和 decode 饱和点纳入换树代价，而不是继续扫
timeout。

完整报告和机器可读数据：

```text
docs/experiments/CIS_PREDICTIVE_CONTINUATION_20260914.zh-CN.md
docs/experiments/data/cis_predictive_continuation_20260914.json
```

原始数据：

```text
/data/disk/wangzili/cis-predictive-scheduler-20260913
```

本轮只使用服务器 `159.138.5.111` 启动前空闲的物理 NPU 1/4；实验结束后已释放。

## 21. 2026-09-14 Dynamic Peak-Token Admission 已完成

在 `cis-predictive-scheduler` 分支继续实现并验证了算法外层的动态 admission。新
`PeakTokenStepAdmissionController` 不再假设每个 step 等成本，而按 shared prefix、
`C * candidate tokens` 和实际/预计 rollout tail 计算 token/KV 峰值；candidate 完成后
缩小 reservation，同一 job 下一 step 使用原子 transition 保存跨 step 局部性。预算由最近
32 个 step0 滚动校准，并有独立 max-active 安全上限。

P1 `C8/R3/B128/L512` 的严格 fixed-work A/B 中，各方案均为 3328 engine request、
720896 generated token、1344808 FTS。动态 FIFO 相对静态 `step_fifo + cap16`：jobs/s
`+0.11%`、FTS/s `+0.11%`、Job P95 `+0.23%`、0 preemption，说明它已经匹配静态最优，
但没有明显超越。

P0 `C15/R3/B128/L512` 的同卡真实 EOS A/B 中，动态 FIFO 相对静态 cap16：jobs/s
`+5.43%`、FTS/s `+5.06%`、Job mean `-1.80%`、Job P99 `-12.13%`、Job P95
`+1.02%`，0 preemption。7/7 个跨 step transition 全部成功。它修复了上一版固定 5 秒
continuation 在 P0 上 FTS/s `-12.24%` 的迁移失败。

另做了队列消融。`largest_fit + 1s coalesce` 虽使 P1 makespan 缩短 0.9%，却使 Job mean
恶化 23.6%；`balanced_fit` 使吞吐下降 0.63%、Job mean 恶化 16.9%。这说明长短 context
主动混排会破坏 attention batch shape，下一步不再堆 LPT/交替等通用启发式，默认保留
FIFO/locality。

准确边界：该方案是新增的 CIS-aware 外层 token-budget admission，不是 vLLM 自带开关，
也还不是完整 EngineCore tree scheduler。它仍以 `active_step_limit=16` 作为初始预算倍率、
以32作为 safety bound。下一步应从 runtime KV capacity/MNS 直接推导预算，再用
preemption/KV/batch telemetry 做慢速闭环，以逐步取消固定 cap 标尺。

P0 同 namespace 的64个顶层请求 seed 完全相同，但只有25个最终输出一致，证明当前 native
sampler 仍受 batch/admission 顺序影响。P0 动态 FTS 总量只少0.35%，FTS/s仍提高5.06%，
正向性能结论不依赖 jobs/s；不过正式 exact A/B 前仍需解决 sampler 确定性。

详细报告与机器数据：

```text
docs/experiments/CIS_DYNAMIC_PEAK_TOKEN_ADMISSION_20260914.zh-CN.md
docs/experiments/data/cis_dynamic_peak_token_admission_20260914.json
```

原始数据：

```text
/data/disk/wangzili/cis-peak-budget-20260914
```

本轮只使用服务器 `159.138.5.111` 启动前空闲的物理 NPU 0/5；所有本轮容器均已退出。

## 22. Runtime-KV Context Transfer 扩展已实现，待双卡验证

继续审计发现第21节 rolling peak-token 仍以 `cap16 * workload 平均 step 成本` 为预算，
所以数据集整体变长时并发不会自动下降。当前工作区已新增 `runtime_kv_budget`：用最小
vLLM v0.18 EngineCore 只读补丁返回每 rank KV block 数、block size 和 token capacity，
外层 controller 以容量的0.9为硬件预算，并按真实 KV block 粒度估算 CIS step。

新增固定 workload 工具：

```text
experiments/swebench/stratify_cis_workload.py
```

它从原 public-256 无放回抽取32条 short `<4K`、medium `8K-16K`、long
`16K-32K` 请求，中位 prompt 分别为2813.5、11530.5、20376.5。分层文件已放在服务器：

```text
/data/disk/wangzili/cis-context-strata/{short,medium,long}
```

按现有 TP2 的508928-token KV capacity和P1配置，0.9 runtime budget预计首次分别放行
32/20/13个 step；rolling reference 仍接近16。这只是容量模型结果，尚不是性能结论。

待卡矩阵：三个分层分别跑 static cap16、rolling peak-token、runtime-KV 0.9；另以同一
P0设置补跑 `step-subtree-K`，直接回答 runtime dynamic 与“每 candidate 的 sibling
rollout bundle”谁更好。旧数据中 subtree K8 相对当时 Step control 的 jobs/s 下降约
17.8%，而新动态在另一组严格同卡 P0 中相对静态 cap16 的 FTS/s提高5.06%；由于不是同一
轮 A/B，当前只能说新方案更有希望，不能把两组百分比直接相减。

本地验证：相关 pytest `140 passed, 1 deselected`；被排除的 Sobol 用例只因本机没有
torch。vLLM v0.18 capacity patch 已在官方镜像源码上通过 `git apply --check`。两台服务器
检查时都只剩一张空闲卡，未抢占他人设备；单卡4B smoke 在模型加载前被 Ascend 的全机
DCMI/categorical 枚举问题拦截。

## 23. 共享服务器设备预留检查与跨 context 矩阵

服务器 A 的卡1/4一度被 `npu-smi` 显示为无进程，但运行中的
`orthrus-torchprof-npu1`、`orthrus-v023-npu4` 已分别映射这两个设备节点；服务因此在
`torch_npu` 初始化时得到0个设备。没有终止或复用这些容器。服务器 B 的各卡也均被运行中
容器映射，目前没有经过双重检查的空闲 TP2。

`run_cis_forest_ab_remote.sh` 已增加 Docker reservation preflight：除 NPU PID 外，还会
检查运行容器的 `.HostConfig.Devices`。默认遇到预留即失败；只有所有者明确确认后才允许用
`CIS_ALLOW_DOCKER_RESERVED_DEVICES=1` 覆盖。

新增 `run_cis_context_transfer_remote.sh`，支持断点续跑：

- `context`：P1 的 short/medium/long 分层分别运行 static cap16、rolling peak-token、
  runtime-KV 0.9；
- `subtree`：同一 P0 workload 分别运行 static cap16、runtime-KV 0.9、subtree-K8；
- `all`：顺序完成以上12组并生成 summary JSON。

真实 NPU 结果仍待完整空闲双卡出现，不能用容量模型预计值替代性能结论。
