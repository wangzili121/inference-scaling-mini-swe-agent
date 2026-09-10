# CIS-Forest：Conditional IS 专属 Branch-Reduce-Continue Runtime

## 1. 问题边界

本设计只优化一次 ordinary Conditional IS 模型调用内部的执行，不改变
candidate、rollout、reward、重采样概率或最终输出分布。它不是多轮 Agent
编排，也不依赖剪枝、提前停止或减少 rollout 数量来获得性能收益。

每个 block 的真实依赖图为：

```text
selected trunk
  -> C 个 candidate，每个生成 B token
  -> 每个 candidate 派生 R 个 rollout
  -> 对 C x R 个完整分支计算 reward
  -> reduce + resample 选中一个 candidate
  -> 仅把选中 candidate 的 B token 接到下一 block
```

通用 vLLM 把这些节点摊平成普通请求。APC 可以避免重复 prefill，但 decode
attention 仍会为每条活跃分支重复读取共享 trunk KV；scheduler 也不知道哪组
请求属于同一棵树、哪条完成后能够解除 block barrier。

## 2. Profiling 证据

固定 Qwen3-Coder-30B-A3B-Instruct、vLLM-Ascend v0.18 MRV1、真实
SWE-bench 调用和已调优 general baseline 后，P0-P3 的共同结论为：

- sequence-logprob 与 Consilience 均没有额外 scoring forward，reward、weight、
  resample 的累计阶段占比低于 0.01%。
- APC token hit ratio 通常为 95%-97%；它已解决大部分重复 prefill，但没有解决
  decode 时的共享 KV 重读。
- Ascend Torch trace 中 `FusedInferAttentionScore` 始终是最大或最主要 kernel，
  典型占设备 busy 时间约 38%-61%。
- P0/P1/P2/P3 的四卡双 instance 均出现不同程度的实际工作偏斜；最新 P3
  Torch pass 在严格 32/32 job 分配下仍有 30.19% endpoint mean latency skew。
- P3 四卡 Torch pass 中四个 rank 的 NPU busy ratio 为 75.13%-86.58%，暴露
  HCCL 占各自采集窗口约 10.57%-19.07%。
- P0 `C15/R3` 的单位 forward-work 吞吐显著低于较小的 P1，且高扇出时
  candidate/prefill 与 rollout/decode 形状更容易相互干扰。

已经完成的算法专属 A/B 也给出了两个重要的负结论：

- host 侧 candidate completion streaming 相对同机 baseline 的 jobs/s 下降
  `9.78%`，P95 从 `335.40s` 增至 `367.35s`；固定 B 的 candidate 本来就几乎
  同时完成，candidate-to-rollout 的中位提前量只有 `5.85ms`，不足以覆盖新引入
  的小批提交和 KV 压力。
- 单 job rollout 分批提交相对 baseline 的 jobs/s 下降 `12.79%`，P95 增至
  `399.17s`。这说明背压必须观察整个 engine 的 candidate/rollout 构成，不能
  每棵树独立限流。
- 静态 whole-job work routing 在 P1 的 work-normalized throughput 提升
  `22.65%`，但在 P3 下降 `11.39%`。仅用请求开始前可见的 prompt/C/R/B/L
  无法稳定预测随机 EOS 后的实际剩余工作，不能直接成为默认路由。

因此，继续只调 MNS、MBT 或普通请求并发不会触及主要剩余问题。下一阶段需要
让 runtime 显式理解 Conditional IS 的树结构。

## 3. 两层 Forest Attention

### 3.1 固定几何

Conditional IS 的树不是任意 radix tree，而是运行前已知的两层规则结构：

1. 全部 `C x R` rollout 共享 selected trunk；
2. 同一 candidate 下的 `R` 个 rollout 还共享该 candidate 的 B-token suffix；
3. 每条 rollout 只有尾部 KV 唯一。

对每个 decode query，将 attention 分为 trunk、candidate suffix 和 unique tail
三段，分别得到 partial output 与 softmax LSE，再用 online-softmax 精确合并。
数学结果与普通全序列 attention 相同。

理论 KV 读取量由：

```text
C * R * (trunk + candidate_suffix + unique_tail)
```

降为：

```text
trunk + C * candidate_suffix + C * R * unique_tail
```

### 3.2 Ascend 机会探针

`experiments/swebench/profile_prefix_forest_attention.py` 已用
`npu_fused_infer_attention_score(..., softmax_lse_flag=True)` 验证 exact 分解。
固定 `R=3`、candidate/unique tail 均为 128 token 时：

| trunk | C | branch | 理论 KV 减少 | 三次 FIA 相对基线 |
|---:|---:|---:|---:|---:|
| 8K | 15 | 45 | 23.95x | 0.434x |
| 16K | 15 | 45 | 31.12x | 0.701x |
| 32K | 15 | 45 | 36.74x | 1.333x |
| 32K | 8 | 24 | 21.50x | 0.759x |
| 32K | 4 | 12 | 11.38x | 0.447x |

最大 output/LSE 误差分别不超过 `4.88e-4/1.91e-6`。这证明共享 IO 的上界
很大，但三个现成 FIA 的 launch、partial output 和 merge 开销会吞掉 8K/16K
收益，不能直接接入正式 runtime。

进一步把三段 FIA 放到独立 NPU stream 并行也没有奏效：所有形状均变慢，
`C15/R3/32K` 仅为 0.886x，且一个形状出现不可接受的数值偏差。该路径被记录为
负结果，不进入产品实现。

### 3.3 正式算子要求

正式实现需要单个 prefix-aware Ascend kernel，或由一个 op 内部管理 multi-tile：

- 输入 paged block table、`tree_id/candidate_id` 和每段长度；
- trunk KV 在片上按 sibling query group 复用一次；
- candidate suffix 在同一 `R` 子组内复用；
- 在 kernel 内完成 partial softmax/LSE merge，不落完整中间张量；
- 依据 `shared_tokens x active_siblings`、head 数和尾长动态选择普通 FIA 或
  Forest Attention；
- schedule 仅在 branch/EOS 结构改变时更新，不在每 token 重建任意 radix tree。

这与 Hydragen、DeFT、FastTree、FlashForge 和 PAT 的 shared-prefix attention
原则一致，但利用了 CIS 固定的 `C -> R -> reduce` 结构，并以 Ascend FIA/CANN
为实现目标；现有公开实现主要面向 CUDA，vLLM-Ascend v0.18 没有等价插件。

## 4. Forest Scheduler

kernel 与调度必须共同设计。只做 candidate completion streaming 会降低 barrier，
但也可能把 sibling 拆散，损失 Forest Attention 的共享组；只等待完整 sibling
则会增加尾延迟。因此 scheduler 的调度单位应是可调大小的 subtree bundle。

前两种直观方案已被真实 P0 饱和 A/B 排除：candidate streaming 和 per-job
bounded submission 都会降低吞吐。静态 whole-job work balancing 也只在部分
profile 生效。实验收敛与后续 scheduler 顺序据此收窄为：

1. `global stage-aware frontier`：在同一个 engine 的所有 CIS job 之间共享 rollout
   credit，根据全局 candidate/rollout 在途量做 admission；每次按完整 candidate
   sibling bundle 获取 credit，完成一条释放一条。capacity `128/192/240` 已完成
   P0 A/B，分别为 `-4.08%/+0.78%/-3.38%` work-normalized throughput；192 的
   微小正值伴随 47 次 preemption 和更差 P95，不保留。至此 host admission 调整
   已有三类负结果，不再继续扩网格。
2. `in-engine branch-on-token`：candidate 生成到 B token 时，在 engine 内直接 fork
   R 个 block-table child，避免 host 完成回调、重新 tokenize/hash/readmit 后代。
   vLLM 的 `SamplingParams(n=R)` 在 admission 时展开为 R 个独立 EngineCoreRequest，
   不是这种到达 branch point 后的 continuation fork。
3. `online remaining-work routing`：路由代价由已完成 block、真实生成 token、当前
   branch 数和剩余 barrier 数持续更新，不再只用请求开始前的静态长度估计。
4. `barrier-critical priority`：不再作为独立 host admission 实验；只有 engine 内
   fork 后仍观察到显著 barrier tail，才为即将解除 reduce 的最后一个 sibling
   bundle 提高优先级。

每项必须记录 candidate 完成到 child admission 的间隔、活跃 sibling 数、混合
prefill/decode batch 比例、最后一个 rollout 的 barrier tail、jobs/s、P95 和
forward-token-slots/s。优化若仅减少 queue wait 却降低单位 forward-work 效率，
不予保留。

## 5. 跨 Instance Branch Parallel

whole-job routing 只能改善多 job 负载均衡，不能缩短单个重型 CIS job。后续四卡
模式应把一棵树按 candidate subtree 分给两个 TP2 instance：

- 两端各持有 selected trunk；
- candidate 及其 R 个 rollout 保持同 instance，保留二级 KV locality；
- reduce 时只汇总 C 个标量 log-weight；
- 选中后先传 B 个 token，由另一端按需重算短 candidate delta；只有 profiling
  证明重算成本显著时才实现 selected KV transfer；
- work stealing 只移动尚未生成 candidate 的 subtree，避免搬运活跃 KV。

该方案让单个 `C15/R3` job 同时使用四卡，同时通信量与 C 个权重和一个 B-token
delta 成正比，而不是传输所有 rollout KV。它比通用 P/D 更贴合 Conditional IS
的 branch-reduce-continue 语义。

## 6. Persistent Branch-Reduce-Continue

最终 runtime 应把一次 MiniAgent 模型调用作为一个持久 CIS job，而不是每个
candidate/rollout 都重新进入普通请求生命周期：

- selected trunk 的 block table 跨 block 保留；
- candidate 完成后 fork R 个只读引用，loser 在 reduce 后统一释放；
- request id、seed、reward statistics 和 branch metadata 保持设备/host 一致；
- 下一 block 直接从 winner branch continuation，不重新 hash 和 readmit 完整前缀。

是否开发该层取决于后续 trace 中 request lifecycle、block-table 操作与 Host bubble
的占比。当前数据已经排除了 CPU reward，却还不足以声称 lifecycle 是主瓶颈。

## 7. 验收顺序

1. 保留 whole-job routing、candidate streaming、per-job bounded 和 global
   frontier 的真实负结果；停止继续微调 host admission。
2. 实现 in-engine branch-on-token 原型，直接量化 host readmission、prefix hash 和
   block-table fork 的开销；有至少 5% 端到端收益后再做干净复验。
3. 实现 Forest Attention CANN microkernel；先过逐 token reference，再用真实
   branch shape 验证 attention latency，最后接 vLLM-Ascend attention backend。
4. 只有 kernel 和 scheduler 均通过后，开发四卡单 job 的 candidate-subtree
   parallel；先 token transfer，后按证据决定 KV transfer。
5. 最终在固定 holdout workload 上要求 100% 成功、无额外 preemption，并相对
   已启用全部 general 优化的原始 Conditional IS 获得至少 10% 端到端收益。

## 8. 参考工作

- [Hydragen](https://arxiv.org/abs/2402.05099)：exact shared-prefix attention 与 LSE merge。
- [DeFT, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a6df53f082619d02b9fad64a022e5de3-Abstract-Conference.html)：KV-guided grouping 与 flattened tree KV splitting。
- [FastTree, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/hash/96894468eb44631a32d7ebd56f9892c7-Abstract-Conference.html)：tree-adaptive kernel/runtime partition。
- [FlashForge](https://arxiv.org/abs/2505.17694)：prefix-aware kernel 与 cost-based task division。
- [PAT, ASPLOS 2026](https://arxiv.org/abs/2511.22333)：pack-forward-merge、lazy scheduling 与 multi-tile kernel。
- [Preble](https://arxiv.org/abs/2407.00023)：prefix-aware distributed load cost。
- [FlashTTS](https://arxiv.org/abs/2509.00195)：test-time scaling 的动态 prefix-aware scheduling。
- [Competitive Non-Clairvoyant KV-Cache Scheduling](https://arxiv.org/abs/2601.22996)：
  在未知输出长度下用 staggered pipeline 平滑 KV 内存增长；它启发了已经完成的
  global frontier A/B，但真实 CIS 结果为负，不能直接照搬其结论。
- [Regime-Aware Routing](https://arxiv.org/abs/2607.09248)：请求执行时逐步暴露行为，
  再动态路由到不同子调度器；与本实验中 P1/P3 静态路由结论相反的现象一致。

原始微基准结果保存在：

- `docs/experiments/data/cis_forest_two_level_sequential_20260910.json`
- `docs/experiments/data/cis_forest_two_level_parallel_20260910.json`
