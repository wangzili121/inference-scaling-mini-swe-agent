# Conditional IS Engine Tree Handoff 与生命周期实验

日期：2026-09-11
分支：`cis-engine-tree`
模型：`Qwen3-Coder-30B-A3B-Instruct` BF16
部署：vLLM-Ascend v0.18 MRV1，双卡 TP2，`MNS=256`，`MBT=32768`，
memory=0.90
负载：固定 64 个公开 SWE-agent 调用快照，workers=32

## 目的

这一轮不再把 candidate 和 rollout 仅仅当作字符串前缀相同的普通请求，而是验证
三个更接近 CIS 树语义的问题：

1. candidate 完成时能否把物理 block table 直接交给其 R 个 rollout；
2. 为跨越 frontend readmission 间隔而持有多少 parent KV 才不会破坏全局调度；
3. candidate 完成即提交 rollout 与物理 KV handoff 组合后，能否接近真正的
   branch-on-token。

所有方案保持 C/R/B、reward、seed 派生与重采样逻辑不变。由于 vLLM 的原生
categorical sampler 在不同 batch shape 下不保证同 seed token-bit-exact，性能同时
报告 jobs/s、forward-token-slots/s、生成 token 数与尾延迟，不从本轮推导质量结论。

## 实现

### Direct fork

candidate 在 EngineCore teardown 前记录完整物理 block table 与 block hash。rollout
到达时只有在每个 block ID 的当前 hash 仍一致时才直接采用；否则自动退回 APC。
它不额外持有引用，因此不会改变 KV 容量，但排队较久时可能 stale。

### Full-parent handoff

candidate teardown 前给可复用 parent blocks 增加临时引用，直到 R 个 child 全部
adopt 后再释放。实现额外识别 EOS/stop candidate：此类 candidate 在算法中没有
rollout，不能建立 lease。可选 credit 按全局唯一物理 block 数限制 handoff；超出
预算时退化为 candidate-suffix lease，再不足时退化为 direct fork。

### Streaming + handoff

算法层使用 candidate completion callback，一条 candidate 完成后立即准备其 R 个
rollout；EngineCore handoff 保证这段短间隔内 parent block 不回到可覆盖状态。这是
现有接口下最接近 branch-on-token 的原型，但 rollout 仍是新的 EngineCore request，
尚未消除 frontend round-trip 和 child request lifecycle。

## Handoff 结果

> 2026-09-11 校正：下表的 P0 `baseline=0.1562 jobs/s` 是后续同机重跑的较弱
> control，只适合这组运行内 A/B；它不是此前已经得到的最佳 Step cap6 结果。
> 与最佳 Step cap6（`0.1688 jobs/s`、约 `2931 FTS/s`、Job mean/P95
> `134.7/246.1s`）比较，direct fork 的增量仅为 jobs/s `+0.5%`、FTS/s
> 约 `+0.8%`，且 Job mean/P95 分别恶化约 `7.0%/2.6%`。因此 direct fork
> 只能判定为“功能成立、增量未被稳定证明”，不能表述为独立的 `+8.7%` 优化。

百分比相对同算法配置的 Step baseline；延迟下降为正向。

| 配置 | 方案 | jobs/s | Job mean / P95 | Step mean / P95 | Barrier mean / P95 | FTS/s | Preempt | Fork hit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P0 `C15/R3` | baseline | 0.1562 | 138.8 / 313.7s | 31.8 / 73.5s | 24.3 / 52.0s | 2785 | 0 | - |
| P0 | direct fork | **0.1697 `+8.7%`** | 144.1 `+3.8%` / 252.5s `-19.5%` | 30.4 `-4.4%` / 72.8s `-1.0%` | 23.6 `-2.8%` / 52.3s `+0.6%` | **2954 `+6.1%`** | 0 | 906/906 |
| P0 | full-parent handoff | 0.1580 `+1.2%` | 147.9 `+6.5%` / **248.3s `-20.9%`** | 31.0 `-2.6%` / 76.4s `+3.9%` | **20.6 `-14.9%`** / 52.8s `+1.5%` | 2792 `+0.3%` | 0 | 975/975 |
| P0 | streaming + handoff | 0.1467 `-6.0%` | 151.9 `+9.4%` / 279.6s `-10.9%` | 31.1 `-2.0%` / **69.7s `-5.2%`** | 22.0 `-9.2%` / **45.6s `-12.2%`** | 2606 `-6.4%` | 0 | 1068/1068 |
| P1 `C8/R3` | baseline | 0.2830 | 91.1 / 169.1s | 46.7 / 111.0s | 25.7 / 74.6s | 4218 | 0 | - |
| P1 | direct fork | **0.2894 `+2.2%`** | **90.1 `-1.2%`** / **162.8s `-3.8%`** | **46.4 `-0.6%`** / **104.9s `-5.5%`** | 28.6 `+11.3%` / **66.4s `-10.9%`** | **4307 `+2.1%`** | 0 | 471/471 |
| P1 | streaming + handoff | 0.2870 `+1.4%` | 92.7 `+1.7%` / 166.4s `-1.6%` | 48.9 `+4.8%` / 122.3s `+10.1%` | 29.6 `+15.0%` / 79.4s `+6.4%` | 4260 `+1.0%` | 0 | 444/444 |
| P2 `C4/R2` | baseline | **0.3814** | 62.9 / 143.7s | 54.2 / 132.0s | 16.7 / 51.5s | **5138** | 1 | - |
| P2 | direct fork | 0.3410 `-10.6%` | 72.3 `+14.9%` / 152.6s `+6.2%` | 60.6 `+11.9%` / 147.6s `+11.8%` | 23.3 `+39.8%` / 77.0s `+49.7%` | 5203 `+1.3%` | 4 | 162/208 |
| P2 | terminal-aware full handoff, 20% credit | 0.3662 `-4.0%` | **62.9 `-0.0%`** / 141.3s `-1.7%` | 53.4 `-1.4%` / 132.1s `+0.0%` | 18.1 `+8.4%` / 56.0s `+8.8%` | 4928 `-4.1%` | 2 | 172/172 |
| P2 | terminal-aware full handoff, 10% credit | 0.3252 `-14.7%` | 68.2 `+8.4%` / 151.7s `+5.6%` | 53.7 `-1.0%` / 139.2s `+5.4%` | 16.6 `-0.6%` / 51.5s `+0.0%` | 4501 `-12.4%` | 1 | 224/224 |
| P2 | streaming + handoff | 0.3670 `-3.8%` | **61.7 `-1.9%`** / **133.1s `-7.4%`** | **51.7 `-4.5%`** / **125.8s `-4.7%`** | **14.2 `-14.6%`** / **48.3s `-6.1%`** | 4832 `-5.9%` | 4 | 182/182 |

P0 streaming + handoff 在同型号的第二台主机运行，适合判断方向，但不把约 6% 的
吞吐差直接解释为严格同机回退。P2 为同机对照。

20% credit 的峰值为 606 个唯一 block，低于 795 的预算，未触发 fallback，因而
等价于该负载下的无预算 terminal-aware handoff。10% credit 的预算为 397，峰值
380；116 个 parent 中 22 个退化为 candidate-suffix lease。预算并没有带来收益。

## Step grouping 与 rollout-first 组合

此前提出的“限制活跃 step，并在窗口内让 rollout 全局优先”已在三组配置上运行，
但没有在 step-only 之上继续增益：P0 从 `0.1688` 降至 `0.1608 jobs/s`，P1 从
`0.2752` 降至 `0.2471 jobs/s`，P2 从 `0.3564` 变为 `0.3545 jobs/s`。P0 再叠加
direct fork 为 `0.1650 jobs/s`，仍低于 step-FIFO + direct fork 的 `0.1697`。
全局 rollout-first 会让不同 step 的 rollout 穿过当前局部组，从而部分抵消
step admission 获得的 prefix/batch locality；后续不继续扩大这一组合的参数矩阵。

## 失败证据

最初的 P2 full-parent run 尚未识别 terminal candidate，错误地为全部 320 个
candidate 建立 lease，其中大量节点永远没有 child。结果出现 145 次 300 秒 lease
expiry，engine queue P95 达 290.4 秒，jobs/s 从 0.3814 跌到 0.1534。这个结果不
纳入公平性能表，但它证明树 runtime 必须知道“该节点是否会产生后代”，普通 APC
或 LRU 无法自行推断。

第一次 resample-GC 运行还暴露了 ID 边界：EngineCore 的实际 request ID 带随机
后缀，而算法层只知道稳定的 CIS 节点 ID，因此 71 条 branch record 全部无法在
reduce 时匹配。修正方案是在 sampling metadata 中显式传递稳定
`cis_branch_handle`，不能从框架内部 request ID 反推算法树。第二次运行已实现
逐 transition 的精确匹配。

## Resample GC 结果

| 配置 | 对照 | jobs/s | FTS/s | Job mean / P95 | Step mean / P95 | Preempt | GC matched / blocks | GC mean / P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P0 `C15/R3` | direct fork | **0.1697** | **2954** | **144.1 / 252.5s** | **30.4 / 72.8s** | 0 | - | - |
| P0 | direct fork + exact GC | 0.1611 `-5.1%` | 2852 `-3.4%` | 154.1 `+7.0%` / 303.9s `+20.3%` | 31.9 `+5.1%` / 89.8s `+23.4%` | 0 | 313 / 431 | 77 / 142ms |
| P2 `C4/R2` | record-only、同机 | 0.3203 | **4996** | 69.61 / **151.00s** | 54.13 / **141.39s** | 1 | 0 / 0 | - |
| P2 | exact GC、同机 | 0.3367 `+5.1%` | 4981 `-0.3%` | 69.60 `-0.0%` / 151.09s `+0.1%` | 54.12 `-0.0%` / 142.49s `+0.8%` | 10 | 72 / 89 | 293 / 1401ms |

P2 的 jobs/s 变化来自本轮生成工作量和结束时刻差异；工作归一化 FTS/s、Job/Step
延迟均无改善，而且 preemption 增多。P0 是同主机、相同 direct-fork 基线上的严格
比较，也明确回退。因此不扩到 P1。

结束 request 的 block 本来就已是 `ref_cnt=0` 并位于 free queue；保留 hash 只是
允许 APC 在被覆盖前复用。GC 做的是提前删除 hash 并把已知死亡页搬到队首，并不会
凭空增加 free-block 数量。当前 P0 没有 preemption，逐 step 同步 EngineCore RPC
及缓存扰动没有可抵消的收益。另一个细节是 vLLM-Ascend 在 APC/chunked-prefill 下
使用 128-token block；不少 candidate/rollout 在分支边界后没有形成完整独占 block，
因此安全的整页记录主要来自较长 rollout。若以后需要更细粒度回收，应做压力触发的
边界页/子页策略，而不是在无压力时无条件清空。

## 当前结论

1. P0 direct fork 相对本轮较弱 control 为正，但相对历史最佳 Step cap6 只有
   jobs/s `+0.5%`、FTS/s 约 `+0.8%`，尾延迟还略差，尚不能作为独立性能优化。
   它的可靠收获是证明 906/906 个 child 可以安全 adopt 物理 block table。
   额外 full-parent retention 只增加压力，反而抹去大部分收益。
2. P2 的 streaming + handoff 明显修复了单独 streaming 的灾难性回退，并把 Job、
   Step、Barrier 尾延迟全部降低，但 FTS/s 与 jobs/s 仍下降。说明“接着 fork”能
   消除 free-queue race，却不能抵消更碎的 prefill/decode batch shape。
3. full-parent lease 不是最终 tree request。真正需要的是在 parent 仍活跃时把引用
   原子转交给已知 children，并由全局 KV credit 控制 branch admission；不能把大量
   parent 路径作为异步 hint 长期 pin 住。
4. reduce-transition GC 已证明可以用稳定 branch handle 精确表达 winner/dead
   生命周期，但无条件同步回收在 P0/P2 都没有性能收益。机制保留为诊断和未来
   pressure-gated policy 的基础，不进入当前默认快路径。

## 相关工作定位

- [vLLM Automatic Prefix Caching](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)
  管理普通请求的 hash block 与 LRU，但不知道 CIS 的 terminal、winner 或 barrier。
- [ArborKV](https://arxiv.org/abs/2605.22106) 为树搜索建立 active/inactive subtree
  价值与 lazy rehydration；CIS reduce 后 loser 不会回溯，因此可以使用无需预测的
  exact dead-branch transition。
- [Preble](https://arxiv.org/abs/2407.00023) 按 prefix tree 与公平性安排普通请求，
  但不表达 candidate -> rollout -> reduce 的依赖。
- [PAT](https://arxiv.org/abs/2511.22333) 证明 decode attention 需要在 kernel 内
  复用 sibling 共享前缀，而且已经提供 NVIDIA/vLLM 插件；因此不能把泛化
  prefix-aware attention 本身当作我们的创新，后续若做必须落在 Ascend CIS
  两层固定 forest、算法元数据直达 kernel 与实测动态门控上。
- [FlashForge/CoDec](https://arxiv.org/abs/2505.17694) 与
  [DeFT](https://arxiv.org/abs/2404.00242) 同样说明普通 paged attention 会重复读取
  共享 KV；它们是 kernel 设计参考，不是本轮 exact branch GC 的替代。
- [Feather](https://arxiv.org/abs/2605.06046) 已在论文原型中为 vLLM/SGLang 实现
  prefix-homogeneous batching，并显示“小而同质”的 batch 可能优于更大的混合
  batch。我们的 step-window 结果支持这个现象，但 CIS 已直接给出树/step 标签，
  无需其 CHT 动态发现；差异点应是利用已知 barrier 和分支生命周期，而非复现其
  通用 RL scheduler。

## 下一步门槛

1. P1 streaming + handoff 已确认只有约 `+1%` 工作吞吐，并恶化 Step/Barrier，
   不继续扩大 streaming 参数矩阵。
2. 停止逐 step 同步 GC 和普通 free-queue 微调。若后续长上下文 profile 出现真实
   KV 压力，只在 pressure event 批量降低 dead branch 保留级，避免每 step RPC。
3. direct fork 仅保留为实验机制，不进入默认配置。主要研发转向
   sibling-prefix-aware Forest Attention，因为 APC 只省 prefill，并不减少 45 条
   rollout 在 decode 中重复读取同一长 trunk。

> 2026-09-12 进展：后续 EngineCore fork-on-token waiter 已做到 candidate 只计算
> 一次、child 物理 KV 命中 100%；compact waiter 又删除了 4300 万个 prompt token
> 的 placeholder payload。P0 相对同机 Step control 的 jobs/s/FTS/s 仍下降
> 3.6%/5.9%，所以动态 tree-root 和更多 release-policy 调参均停止，Forest
> Attention 成为下一条主线。详见 `CIS_ENGINECORE_FORK_AB_20260912.zh-CN.md`。

## 原始数据

- 既有汇总：`docs/experiments/data/cis_tree_fork_ab_host111.json`
- 新增汇总：`docs/experiments/data/cis_engine_tree_handoff_ab_20260911.json`
- Host A：`/data/disk/wangzili/cis-tree-ab-20260911`
- Host B：`/data/disk/wangzili/cis-tree-ab-20260911`
