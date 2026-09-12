# Conditional IS EngineCore Fork-on-token A/B

日期：2026-09-12
分支：`cis-engine-tree`
模型：`Qwen3-Coder-30B-A3B-Instruct` BF16
部署：vLLM-Ascend v0.18 MRV1，双卡 TP2，`MNS=256`，`MBT=32768`，
memory=0.90，native categorical sampler，APC，chunked prefill，NPU graph
负载：固定公开 SWE-agent 调用，`B=128`、`L=512`

## 1. 实验目标

验证 candidate 到达 B token 后是否值得在 EngineCore 内直接派生 rollout，而不是
返回 Python、重新提交 C x R 个普通 request。所有实验保持 C/R/B、reward、seed
派生和 resample 语义不变；只改变 child 何时进入 scheduler，以及 parent KV 如何
转交给 child。

## 2. 调度演进

### 2.1 原始扁平执行

原始路径先提交 C 个 candidate，等待全体完成，再提交 C x R 个 rollout。vLLM 只
看到普通请求，不知道 job、step、parent-child 和 reduce barrier。

### 2.2 已证实有效的 Step 调度

`Step cap + step_fifo` 在算法层限制同时展开的完整 step 数，并给同一 step 的
candidate/rollout 相同优先级。它仍产生两批普通 EngineCore request，但减少不同
step 的无约束交错。历史正式 64-request 结果相对扁平 baseline 为：P0 `+15.3%`、
P1 `+17.0%`、P2 `+7.0%` jobs/s。

### 2.3 已排除的近似实现

- host candidate streaming 会把 rollout 拆成碎片化提交，P0 正式结果回退；
- sibling 小组限流造成小 batch，P0 相对 Step control 回退约 17.8%；
- direct KV hint 可以让 child adopt parent block table，但仍有 frontend readmission，
  相对最佳 Step control 只有约 1% 增量；
- full-path proxy 用分段 RNG 让 R 条路径的前 B token 一致，但会把 candidate 计算
  重复 R 次。P0/P1/P2 的 jobs/s 分别比 Step control 低约 24.3%/29.2%/14.8%，
  证明“重复 candidate 来模拟 fork”不可用。

### 2.4 本轮真实 EngineCore fork

当前原型在同一次 frontend batch 中预注册轻量 rollout waiter。waiter 在 parent
完成前不进入模型；candidate 只计算一次。parent 到达 B 后，EngineCore：

1. 捕获并临时持有 parent 的物理 KV block table；
2. 把 candidate token 追加到对应 child 的 engine-side prompt state；
3. 更新 child block hash，并将 R 个 child 放入正常 waiting queue；
4. child allocation 直接采用 parent blocks，全部 child adopt 后释放 lease；
5. parent 若 EOS/stop，则直接取消 child，取消项的 prefill/FTS 计数为零。

child 仍是实际 attention batch 的单位，因此多棵树可以共同 continuous batch；树只
负责依赖、KV 所有权和 barrier。它已经消除了 candidate 完成后的 Python 二次提交，
但仍预注册完整 child Request；最终动态 tree-root 协议可继续消除 placeholder
lifecycle。

为单独量化 placeholder payload，又实现了 `compact waiter`：前端仍保存完整 prompt
用于输出构造，但发送给 EngineCore 的 dormant child 只携带一个永不执行的 sentinel
token。parent 完成后，scheduler 从 parent token state 重建 child prompt、block hash
和物理 KV 引用。它保留 child future/request identity，只消除长 prompt 的 IPC、复制
和 EngineCore 停放开销。

## 3. Immediate fork 正式结果

下表均为 Host A、固定 64-request burst。百分比以同一 P 配置的历史最佳 Step
control 为基线；FTS/s 是工作归一化吞吐。

| 配置 | 方案 | jobs/s | Job mean / P95 | Step mean / P95 | Barrier mean / P95 | FTS/s | Preempt |
|---|---|---:|---:|---:|---:|---:|---:|
| P0 `C15/R3` | Step control | 0.1548 | 149.4 / 272.9s | 132.5 / 259.8s | 21.5 / 53.8s | 2748 | 0 |
| P0 | candidate 完成即 fork | 0.1552 `+0.2%` | 148.2 / 260.1s | 131.2 / 233.2s | 24.2 / 65.3s | 2595 `-5.6%` | 0 |
| P1 `C8/R3` | Step control | 0.2830 | 91.1 / 169.1s | 81.9 / 151.9s | 25.7 / 74.6s | 4218 | 0 |
| P1 | candidate 完成即 fork | 0.2666 `-5.8%` | 94.7 / 175.4s | 83.9 / 158.2s | 32.5 / 92.7s | 3901 `-7.5%` | 0 |
| P2 `C4/R2` | Step control | 0.3814 | 62.9 / 143.7s | 54.2 / 132.0s | 16.7 / 51.5s | 5138 | 1 |
| P2 | candidate 完成即 fork | 0.3469 `-9.1%` | 67.2 / 149.1s | 54.2 / 138.9s | 23.5 / 69.9s | 4687 `-8.8%` | 11 |

所有被激活 child 的 direct fork 命中率均为 100%，所以失败原因不是 KV 交接失效。
真正的问题是无限制 fork 破坏 candidate/rollout 阶段局部性，增加混合 batch、
barrier tail 和 P2 KV preemption；同时 placeholder 使 EngineCore request 数约翻倍。

## 4. Controlled release 短筛选

为分离“fork 本身”和“何时 admission”，实现三种 release：

- `barrier`：等本 step 所有 candidate 完成后一次放行；
- `tail-2`：剩两条 candidate 时放行已完成 parent 的 child，之后完成即放行；
- `adaptive-0.5`：仅当普通 runnable branch 低于 `0.5 x MNS` 时按需补位，最终
  barrier 兜底。

下表为 16 个不同请求一次性发出、workers=16 的方向筛选。它不是正式饱和结果，
只用于淘汰明显差策略；P0/P1 的 tail-2 在同型号 Host B，P2 三项为 Host A 同机。

| 配置 | release | jobs/s | Job mean / P95 | Step mean / P95 | Barrier mean / P95 | FTS/s | Preempt |
|---|---|---:|---:|---:|---:|---:|---:|
| P0 | barrier | 0.1142 | 57.3 / 132.9s | 50.6 / 127.6s | 25.7 / 61.2s | 2450 | 0 |
| P0 | tail-2 | 0.1184 | 59.4 / 132.2s | 52.3 / 130.5s | 27.6 / 64.2s | 2527 | 0 |
| P1 | barrier | **0.1708** | **66.2 / 90.5s** | **55.5 / 86.8s** | **27.9 / 52.2s** | **3283** | 0 |
| P1 | tail-2 | 0.1463 | 70.1 / 107.3s | 55.8 / 89.1s | 27.9 / 56.6s | 2859 | 0 |
| P2 | barrier | 0.1635 | 47.8 / 78.9s | **33.1 / 62.5s** | **8.1** / 22.2s | 2897 | 0 |
| P2 | tail-2 | **0.1899** | **46.2 / 77.6s** | 33.4 / 64.1s | 9.1 / **20.3s** | **3353** | 0 |
| P2 | adaptive-0.5 | 0.1823 | 51.3 / 80.2s | 37.1 / 68.0s | 11.9 / 33.5s | 3232 | 0 |

P0 tail-2 只有约 3% FTS 信号；P1 明确偏好完整 barrier；P2 tail-2 在同机短筛选
中约 `+16%` jobs/s/FTS/s。固定 tail 不是跨配置通用答案。第一版 adaptive 只在
22 个 P2 step 中提前放行 1 个 parent，说明全局 runnable 数看不到高优先级 step
的 barrier urgency，不能作为完成的策略。

## 5. 当前判断与下一门槛

### 5.1 P0 正式 controlled-release 结果

下表为 Host A、64 个不同请求一次性发出；百分比均相对同一 P0 Step control。

| 方案 | jobs/s | Job mean / P95 | Step mean / P95 | Barrier mean / P95 | FTS/s | Engine requests | Preempt |
|---|---:|---:|---:|---:|---:|---:|---:|
| Step control | 0.1548 | 149.4 / 272.9s | 132.5 / 259.8s | 21.5 / 53.8s | 2748 | 2016 | 0 |
| immediate fork | 0.1552 `+0.2%` | 148.2 `-0.8%` / 260.1s `-4.7%` | 131.2 `-1.0%` / 233.2s `-10.2%` | 24.2 `+12.4%` / 65.3s `+21.4%` | 2595 `-5.6%` | 4275 `+112.1%` | 0 |
| full barrier | 0.1443 `-6.8%` | 158.8 `+6.3%` / 311.7s `+14.2%` | 133.2 `+0.5%` / 307.6s `+18.4%` | 23.0 `+6.8%` / 63.2s `+17.5%` | 2424 `-11.8%` | 4470 `+121.7%` | 0 |
| tail-2 | 0.1542 `-0.4%` | 145.6 `-2.6%` / 243.5s `-10.8%` | 125.6 `-5.3%` / 234.3s `-9.8%` | 22.7 `+5.5%` / 60.8s `+13.1%` | 2605 `-5.2%` | 4350 `+115.8%` | 0 |

`tail-2` 能明显降低 Job/Step 尾延迟，但没有提高 jobs/s，而且单位 forward 工作
吞吐仍下降 5.2%。这更像用提前 admission 换取尾延迟，而不是减少了模型执行成本。
EngineCore fork 的 915 个已激活 child 全部直接采用 parent KV，`unresolved_children=0`，
因此剩余损失不能归因于 fork miss。

### 5.2 P1/P2 正式结果

P1 在 Host B 同一 4/5 卡补跑了 Step control；P2 使用 Host B 已有同配置 Step
control。P1 fork 与 P2 fork 曾并行运行，因此结果只用于否定收益，不用于报告小幅
正收益。两项均为 64-request burst。

| 配置 | 方案 | jobs/s | Job mean / P95 | Step mean / P95 | Barrier mean / P95 | FTS/s | Engine requests | Preempt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| P1 `C8/R3` | Step control | 0.2914 | 89.0 / 162.8s | 80.0 / 158.5s | 32.3 / 76.4s | 4347 | 1015 | 0 |
| P1 | full barrier | 0.2965 `+1.8%` | 88.9 `-0.1%` / 156.5s `-3.9%` | 84.6 `+5.8%` / 155.7s `-1.8%` | 33.5 `+3.9%` / 92.2s `+20.6%` | 4239 `-2.5%` | 2144 `+111.2%` | 0 |
| P2 `C4/R2` | Step control | 0.3977 | 65.2 / 144.1s | 54.7 / 133.5s | 16.2 / 59.7s | 5794 | 480 | 0 |
| P2 | tail-2 | 0.3500 `-12.0%` | 61.8 `-5.2%` / 133.3s `-7.5%` | 51.1 `-6.5%` / 130.3s `-2.4%` | 20.6 `+27.2%` / 71.1s `+19.1%` | 4652 `-19.7%` | 908 `+89.2%` | 12 |

P1 的 jobs/s 小幅上升来自完成顺序和实际生成工作量差异，FTS/s 没有提升，且
barrier P95 明显恶化；P2 的短测正信号在饱和负载下完全消失。两项 activated
child 的 fork hit 均为 100%，`unresolved_children=0`。

### 5.3 实现竞态及修复

首个 P0 full-barrier 运行曾永久留下 3 个 child：parent 在对应 child AddRequest 到达
前已经完成，旧实现没有 waiter 可唤醒，稍后到达的 child 会永远 parked。修复后
Scheduler 为完成的 parent 保留短期 tombstone，并分别记录 expected/remaining
children；late child 到达时立即继承、放行或取消。group 的固定大小和剩余 parent
数也已拆成两个状态，避免乱序 AddRequest 改写 barrier 计数。修复后的 P0
barrier/tail-2 均为 `unresolved_children=0`。

### 5.4 Compact waiter 正式结果

下表为 Host B、64 个不同请求、P0 tail-2；百分比相对同机同参数 Step control。两次
运行使用相同 workload、workers、MNS、MBT、step cap 和 general 优化。卡号不同但
均为同一台机器的 910B3；结果同时以 FTS/s 归一化实际生成工作量。

| 方案 | jobs/s | Job mean / P95 | Step mean / P95 | Barrier mean / P95 | FTS/s | Preempt |
|---|---:|---:|---:|---:|---:|---:|
| Step control | 0.1765 | 140.8 / 212.4s | 130.3 / 212.0s | 19.8 / 47.3s | 3026 | 0 |
| compact EngineCore fork | 0.1702 `-3.6%` | 148.2 `+5.3%` / 290.9s `+37.0%` | 135.2 `+3.7%` / 264.9s `+24.9%` | 23.4 `+18.4%` / 55.5s `+17.5%` | 2848 `-5.9%` | 0 |

compact 路径为 3150 个 dormant child 省去了 `43,002,180` 个 prompt token 的
frontend-to-EngineCore payload；882 个实际 rollout 的物理 KV fork 命中率为 100%，
2268 个 terminal-parent child 零计算取消，`unresolved_children=0`。因此实验已经
隔离并否定“完整 prompt placeholder 是主要瓶颈”：即便删除这部分数据搬运，模型
工作归一化吞吐仍下降 5.9%，barrier 和 tail latency 仍恶化。

与 Host A 的结果交叉看也一致：non-compact tail-2 相对其同机 Step control 的 FTS/s
下降 5.2%，compact tail-2 相对 Host B 同机 control 下降 5.9%。compact 并未恢复
fork 调度造成的 batch-shape/关键路径损失。

### 5.5 当前判断

1. EngineCore 内物理 KV fork 已功能成立，但“完成即倾倒”和“完整 barrier”均被
   正式结果否定；KV 共享本身不等于更好的执行形状。
2. P0 tail-2 只改善尾延迟，没有恢复 FTS；P1 full-barrier 和 P2 full-tail-2 也
   没有工作归一化吞吐收益。固定 release 阈值已完成收敛，不再继续搜索。
3. compact waiter 已进一步删除 4300 万 token 的 placeholder payload，但仍未改善
   FTS/s。由此可推断，仅把预注册 child 改成动态 tree-root 所剩的 request identity
   和 future 管理开销不足以扭转结果；暂不继续重写 frontend/output protocol。
4. 当前主要损失来自 fork 后的 admission、batch shape 和 barrier critical path，
   不是 KV 交接或 prompt 复制。固定 release 阈值和 scheduler 微调主线停止。
5. 下一主线转向 FIA decode 中重复读取 sibling shared trunk 的两层 Forest
   Attention；现有 trace 已证明 sibling 几乎总在同一 forward 中，具备 kernel 侧
   共享的前提。

## 6. 原始与派生数据

- `docs/experiments/data/cis_engine_fork_immediate_p0_p1_p2_20260912.json`
- `docs/experiments/data/cis_engine_fork_controlled_short16_host111_20260912.json`
- `docs/experiments/data/cis_engine_fork_controlled_short16_host167_20260912.json`
- `docs/experiments/data/cis_engine_fork_p0_full_release_20260912.json`
- `docs/experiments/data/cis_engine_fork_p1_full_control_barrier_20260912.json`
- `docs/experiments/data/cis_engine_fork_p2_full_control_tail2_20260912.json`
- `docs/experiments/data/cis_engine_fork_compact_p0_full_20260912.json`
- Host A/B 原始目录：`/data/disk/wangzili/cis-tree-ab-20260911`
