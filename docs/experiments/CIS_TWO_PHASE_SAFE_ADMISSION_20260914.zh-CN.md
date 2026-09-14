# CIS 两阶段安全准入容量验证

日期：2026-09-14

## 1. 要回答的问题

`runtime_kv_budget` 在 candidate 开始前就为整个 CIS step 预留最坏情况：

```text
共享 prompt + C 个 candidate suffix + C*R 个 rollout tail
```

这种做法 exact 且不容易 OOM，但 P0 的真实轨迹中 70.7% candidate 会提前终止，保守
reservation 平均是 realized footprint 的 2.41 倍。一个自然改进是只先预留 candidate
阶段，candidate 完成后再申请 rollout 扩张。

不能简单只按 candidate 占用放行。若所有 tree 都做完 candidate、却没有任何一个 tree
能获得 rollout 空间，系统会发生 expansion deadlock，并可能逐出本应复用的 candidate KV。
本实验验证一种 Banker-style 安全规则：只有加入新 tree 后仍存在一条最坏情况下的完整
tree 完成顺序，才允许它进入 candidate 阶段。

## 2. 容量模型

每个 step 有两个声明值：

- 当前 allocation：共享 prompt 加 C 个 candidate suffix；
- maximum claim：当前 allocation 加最坏的 C*R 个 rollout tail。

设可用 KV 容量为 `available`。如果至少有一棵 tree 的
`maximum - allocation <= available`，它可以扩张、完成并释放全部占用。继续重复这一过程，
直到所有已准入 tree 都能完成，当前状态才是安全状态。

模拟器同时给出三条边界：

1. `full_reservation`：当前 exact 基线，所有 tree 一开始预留 maximum；
2. `two_phase_safe`：只占 candidate allocation，但保持最坏情况安全完成序列；
3. `candidate_only`：完全忽略 rollout 扩张的非安全上界，仅用于判断理论极限。

为了不把排队深度误当成可执行并行度，还额外要求首批 candidate branch 总数不超过
`max_num_seqs=256`。这不是 vLLM 的强制 admission 规则，而是“这些 candidate 能否同时成为
running sequence”的执行 headroom 模型。

## 3. 输入和参数

- P0：`C15/R3/B128/L512`，真实 EOS，71 个 step；
- P1：`C8/R3/B128/L512`，fixed work，128 个 step；
- KV block size：128 token；
- 实测每 rank capacity：508928 token；
- 插件 0.90 安全系数后的预算：458035 token；
- MNS：256；
- 固定 trace 顺序，并做 500 次 `seed=20260914` 的确定性 shuffle 检查顺序敏感性。

## 4. 结果

### 4.1 固定 trace 顺序

| workload | policy | KV-only 首批 step | 加 MNS 执行边界 | post-candidate 压力/预算 |
|---|---|---:|---:|---:|
| P0 | full reservation | 16 | 16 | 42.4% |
| P0 | two-phase safe | 35 | 17 | 46.4% |
| P0 | candidate-only upper bound | 37 | 17 | 46.4% |
| P1 | full reservation | 40 | 32 | 80.1% |
| P1 | two-phase safe | 51 | 32 | 80.1% |
| P1 | candidate-only upper bound | 51 | 32 | 80.1% |

P0 中，安全算法在纯 KV 模型上能把首批准入从16扩大到35，而且非常接近 candidate-only
上界37，说明安全性本身没有吃掉主要容量。但 C15 很快耗尽 MNS，能够立即执行的 candidate
tree 只从16变成17。P1 的两个策略都被32棵 tree、256条 candidate branch 卡住。

### 4.2 顺序敏感性

500 次 shuffle 的 MNS-bound 首批 step 中位数：

| workload | full reservation | two-phase safe |
|---|---:|---:|
| P0 | 14 | 17 |
| P1 | 23 | 30 |

异构 prompt 顺序下，两阶段规则仍能减少保守 maximum claim 造成的 head-of-line blocking。
它最可能在长短上下文混合、candidate 终止率高、MNS 尚有余量时有价值，而不是在固定长度、
candidate batch 已经打满 MNS 时无条件获益。

## 5. 工程结论

暂不把两阶段策略直接设为产品默认，也不据此宣称端到端收益。容量模型没有覆盖：

- candidate 与 rollout 的实际持续时间和混合 batch shape；
- 已完成 candidate KV 在 vLLM APC 中是否会被其他工作逐出；
- 多棵 tree 等待 expansion 时的 queue/admission 开销；
- 跨 step continuation 与下一轮 candidate 的竞争。

若直接放行35棵 P0 tree，额外 tree 大部分只能在 vLLM waiting queue 中等待，可能重现此前
“队列更深但设备不更快”的失败模式。第一优先级仍是完成已有 runtime-KV 插件和静态最优的
NPU A/B。只有出现以下任一证据，才实现运行时两阶段版本：

1. runtime-KV 因过保守导致 scheduler backlog 低或 decode batch 欠填；
2. 8K-128K 混合 workload 中 full reservation 出现明显 head-of-line blocking；
3. profile 证明保留 candidate phase KV 后仍有可用 MNS 和 KV expansion 空间。

实现时必须同时加入：显式 `candidate_closed`/`rollout_granted` 状态、保留一条 completion
lane、rollout expansion 优先于新 candidate、候选 KV residency/APC miss 监控，以及无法
满足安全序列时立即回退 full reservation。不能使用固定折扣或只看历史均值。

## 6. 产品定位与相关工作

[MISA-T](https://arxiv.org/abs/2608.11152)说明 rollout 服务的准入不应只看请求数量，而应
建模 KV commitment、session residency 和 continuation，并可作为 gateway 层策略接入。
我们的不同点是一次 Conditional IS 调用内部具有明确的
`candidate -> rollout -> barrier -> next step` 两阶段 claim，可进行 exact transition 和
安全序列判断，而不是训练系统中的通用 session admission。

[TOPAS](https://arxiv.org/abs/2608.25523)展示了 workflow DAG、剩余关键路径与 prefix state
联合调度的价值；本项目只在单次 CIS 调用内部使用 barrier/continuation 信息，不扩展成昂贵
的多轮 Agent 调度。[BatchLLM](https://arxiv.org/abs/2412.03594)表明 shared-prefix 分组需要与
token batching、attention 执行联合考虑，也解释了为什么此前只做 sibling 小批分组反而
欠填设备。

vLLM 的通用 priority、continuous batching 和可替换 `scheduler_cls` 是产品的执行原语；
它们本身没有 CIS tree、两阶段 maximum claim 或 barrier transition。近期的
[scheduler plugin RFC](https://github.com/vllm-project/vllm/issues/51608)可以作为未来标准接入
接口，但不替代本策略。

## 7. 可复现入口

```bash
python3 experiments/swebench/simulate_cis_two_phase_admission.py \
  P0_ALGORITHM_TRACE P1_ALGORITHM_TRACE \
  --labels P0-real-EOS P1-fixed-work \
  --capacity-tokens 458035 \
  --max-num-seqs 256 \
  --shuffle-trials 500 \
  --output docs/experiments/data/cis_two_phase_admission_20260914.json
```

模拟器只输出容量证据。是否保留策略必须由相同 workload、sampling、MNS/MBT 下的 NPU
fixed-work A/B 决定。
