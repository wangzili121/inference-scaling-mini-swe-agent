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
版本成功率100%、无 OOM，且 jobs/s/FTS/s 不比算法层版本低10%，才运行 full。正式结果
同时比较 jobs/s、FTS/s、generated tokens/s、Job/Barrier P95、prefill、APC 和
preemption，不能只用 jobs/s。

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
