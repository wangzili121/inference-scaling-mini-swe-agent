# Conditional IS KV Fork 与 Branch 生命周期 A/B

日期：2026-09-11
分支：`cis-tree-streaming`
模型：`Qwen3-Coder-30B-A3B-Instruct` BF16
部署：vLLM-Ascend v0.18 MRV1，双卡 TP2，`MNS=256`，`MBT=32768`，memory=0.90
负载：固定 64 个公开 SWE-agent 调用快照，workers=32

## 实验问题

本轮验证三个 Conditional IS 专属假设：

1. candidate 完成后，rollout 能否直接继承其物理 KV block table，省去普通
   APC 的重新查找并避免 candidate tail 被提前覆盖；
2. candidate 到 rollout 排队较久时，只短暂保护 candidate 独有整块是否足够；
3. 已完成 rollout 的独有 tail 永远不会复用，把它提前放到全局驱逐队首是否能
   缓解 KV pressure。

所有功能均在显式 feature flag 下运行，不改变 C/R/B、reward、seed 推导和采样
分布。失败时 direct fork 自动退回原生 APC。

## 实现边界

### Phase-1 direct block-table fork

- candidate 完成时记录完整物理 block 引用、block hash、可复用 token 数和预期
  rollout 子节点数；
- rollout 携带稳定的 parent handle；只有 token hash 与物理 block hash 全部仍
  匹配时，才把 parent blocks 直接交给 child allocation；
- 每个 child 接管后减少计数，全部 R 个 child 接管后释放 hint；stale/miss 自动
  回退 APC；
- 这已经是真实物理 block-table handoff，但 child 仍是新的 EngineCore request，
  尚不是 candidate 到达 B token 后在 scheduler 内原地 fork 的持久树节点。

### Candidate suffix lease

- 仅增加 candidate 相对 selected trunk 新产生的完整 block 的引用；不 pin 6--60K
  的共享长前缀；
- 所有 rollout 接管后立即释放，异常兜底最长 300 秒，hint 总量也有上限；
- 60 秒版本保留为反例，因为饱和 P2 中 child 排队可以超过一分钟。

### Dead rollout-tail demotion

- 只处理 rollout prompt 边界之后的纯生成完整 block；跨越 prefix 边界的混合块
  不动；
- 只移动 `ref_cnt == 0` 的 free cached block，活跃块和共享块不动；
- 不立即销毁 hash/cache entry，只把这些确定无复用价值的块移动到全局 free queue
  的最先驱逐端，因此空间充足时不制造额外操作。

## 相关工作与差异

- [vLLM Automatic Prefix Caching](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)
  已在请求完成时释放引用，并逆序释放 block，使单请求 tail 比 prefix 更早淘汰。
  本实验的 demotion 更强：把 CIS 明确标记的 dead tail 放到全局队首，而不是只在
  该请求刚释放的一组块中保持 tail-first。
- [KVFlow](https://arxiv.org/abs/2507.07400) 把工作流抽象为 Step Graph，按
  steps-to-execution 给 KV tree 节点赋优先级，并让动态 suffix 最先淘汰；共享节点
  采用所有使用者中最保守的优先级。本实验的 `dead/imminent/winner` 生命周期是
  这一原则在 CIS 固定 `candidate -> rollout -> reduce` 图上的确定版本。
- [ArborKV](https://arxiv.org/abs/2605.22106) 保护 active path/ancestors，压缩或
  淘汰 inactive subtrees，并为回溯提供 lazy rehydration。CIS reduce 后不回溯，
  loser 可以直接判定为 dead，不需要价值预测和 rehydration。
- [TensorRT-LLM KV retention policy](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md)
  提供 token-range、decode priority 和 duration；它支持了“有限期软保护”这一
  设计选择，而不是永久 pin。
- [SGLang session-aware radix cache](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/session_radix_cache.mdx)
  同样把 session reference 作为软保护，关闭 session 后只恢复普通驱逐顺序，
  不立即删除 KV。
- [vLLM context-aware retention RFC](https://github.com/vllm-project/vllm/issues/37003)
  也指出 agent workload 在暂停期间会失去可复用前缀，提出带时限的优先驱逐 API。

## 正式结果

> 2026-09-11 校正：下表 P0 的 `+8.7%` 是相对本轮同机但偏慢的
> `0.1562 jobs/s` control。与此前最佳 Step cap6（`0.1688 jobs/s`、约
> `2931 FTS/s`、Job mean/P95 `134.7/246.1s`）相比，direct fork 只有
> jobs/s `+0.5%`、FTS/s 约 `+0.8%`，Job mean/P95 反而约 `+7.0%/+2.6%`。
> 所以表中百分比不得作为 direct fork 的最终增益结论。

百分比均相对同配置 Step baseline；延迟下降和吞吐上升是正向。不同运行的随机
生成 token 数仍有小幅差异，因此同时报告 jobs/s 与 forward-token-slots/s。

| 配置 | 方案 | jobs/s | Job mean / P95 | Step mean / P95 | Barrier mean / P95 | FTS/s | Preempt | Fork hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P0 `C15/R3` | baseline | 0.1562 | 138.8 / 313.7s | 31.8 / 73.5s | 24.3 / 52.0s | 2785 | 0 | - |
| P0 | direct fork | **0.1697 `+8.7%`** | 144.1 `+3.8%` / **252.5s `-19.5%`** | **30.4 `-4.4%` / 72.8s `-1.0%`** | **23.6 `-2.8%` / 52.3s `+0.6%`** | **2954 `+6.1%`** | 0 | **906/906** |
| P0 | rollout-first + fork | 0.1650 `+5.7%` | 154.2 `+11.1%` / 303.3s `-3.3%` | 31.5 `-0.9%` / 82.0s `+11.6%` | 21.5 `-11.5%` / 57.1s `+10.0%` | 2875 `+3.2%` | 0 | 948/948 |
| P1 `C8/R3` | baseline | 0.2830 | 91.1 / 169.1s | 46.7 / 111.0s | 25.7 / 74.6s | 4218 | 0 | - |
| P1 | direct fork | **0.2894 `+2.2%`** | **90.1 `-1.2%` / 162.8s `-3.8%`** | **46.4 `-0.6%` / 104.9s `-5.5%`** | 28.6 `+11.3%` / **66.4s `-10.9%`** | **4307 `+2.1%`** | 0 | **471/471** |
| P2 `C4/R2` | baseline | **0.3814** | **62.9 / 143.7s** | **54.2 / 132.0s** | **16.7 / 51.5s** | 5138 | **1** | - |
| P2 | direct fork | 0.3410 `-10.6%` | 72.3 `+14.9%` / 152.6s `+6.2%` | 60.6 `+11.9%` / 147.6s `+11.8%` | 23.3 `+39.8%` / 77.0s `+49.7%` | 5203 `+1.3%` | 4 | 162/208 |
| P2 | 300s suffix lease | 0.3796 `-0.5%` | 64.7 `+2.9%` / **139.6s `-2.8%`** | 54.3 `+0.2%` / 136.4s `+3.3%` | 18.9 `+13.3%` / 57.8s `+12.4%` | **5510 `+7.3%`** | 2 | 148/178 |
| P2 | dead-tail demotion | 0.3674 `-3.7%` | 66.7 `+6.0%` / 155.9s `+8.5%` | 57.5 `+6.1%` / 138.9s `+5.2%` | **13.7 `-18.0%`** / 57.4s `+11.5%` | 5347 `+4.1%` | 4 | - |
| P2 | 60s lease + demotion | 0.3024 `-20.7%` | 76.1 `+20.9%` / 168.8s `+17.5%` | 59.9 `+10.6%` / 154.8s `+17.2%` | 25.6 `+53.4%` / 67.5s `+31.2%` | 5257 `+2.3%` | 9 | 84/190 |

300 秒租约运行在另一台同型号 910B3 主机上，用于快速判断机制上界；它未显示
端到端收益，因此没有继续占卡做同机复验。60 秒租约发生 203 次 lease expiry，
仅 38.5% child 命中，明确说明固定一分钟不覆盖饱和队列。

dead-tail demotion 实际处理 108 条 rollout、175 个独有完整 block，说明标记和
页级回收确实执行了；但它没有减少 preemption，端到端指标反而退化，故不作为
默认优化。P0 的 rollout-first 与 fork 组合也弱于 fork 单独使用，表明调度策略
与 KV handoff 存在负交互，不能机械叠加。

## 结论

1. Phase-1 direct fork 的 906 个 child 全部直接命中，证明物理 KV handoff
   功能成立；但相对最佳 Step cap6 的性能增量只有约 1%，且尾延迟没有改善，
   不能把相对弱 control 的 `+8.7%` 当作最终收益。
2. P1 收益只有约 2%，P2 没有端到端收益。P2 即使把 suffix lease 延长到 300 秒，
   仍有 30 个 unique child miss；没有过期事件，说明被覆盖的是未受保护的共享
   parent path，而不是最后一个 candidate suffix block。
3. 若为 phase-1 hint 硬 pin 整条 6--60K parent path，会把并发树的 KV footprint
   放大到不可接受，方向不成立。后续已经实现 scheduler 内 branch-on-token waiter：
   candidate block table 仍有引用时原地给 R 个 child 增加引用，然后再释放 parent。
4. “知道 loser 后删除 KV”可以实现，但单独提前驱逐 rollout tail 已被证伪。
   下一版应在 reduce transition 上同时表达 winner retain 和 loser demotion，并
   继续保持 soft priority；只有 pressure 需要空间时才真正覆盖 loser。
5. direct fork 暂时仅作为低 stale-risk 的实验开关，不默认启用；ParentRequest
   前端 grouping 仅带来约 `+0.7%` generated-token throughput，也不继续扩矩阵。

## 2026-09-12 后续验证与收敛

1. EngineCore fork-on-token waiter 已完成 P0/P1/P2 正式 A/B：物理 KV 命中率 100%，
   但 immediate/barrier/tail-2 均未超过各自最佳 Step control 的 FTS/s。
2. compact waiter 又在 P0 省掉 4300 万个 prompt token 的 IPC/占位 payload，
   相对同机 Step control 的 jobs/s/FTS/s 仍下降 3.6%/5.9%。因此不继续实现动态
   tree-root，也不扩大 release-policy 网格。
3. reduce 后的 loser-tail demotion 已在压力测试中回退；没有 preemption 的正式
   baseline 不继续开发 page ownership。只有未来长上下文出现 winner miss 或真实
   KV pressure 时，才恢复 soft retain/demotion。
4. 当前主线转入两层 Forest Attention CANN kernel；已有 trace 和 FIA 微基准分别
   给出了端到端占比、共享几何和 exact LSE merge 的依据。

## 原始数据

- `docs/experiments/data/cis_tree_fork_ab_host111.json`
- `docs/experiments/data/cis_tree_lease_host167.json`
- `docs/experiments/data/cis_engine_fork_compact_p0_full_20260912.json`
- Host A：`/data/disk/wangzili/cis-tree-ab-20260911`
- Host B：`/data/disk/wangzili/cis-tree-ab-20260911`
