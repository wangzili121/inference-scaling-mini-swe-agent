# CIS 树级 Ready Frontier 实现与 A/B 门槛

日期：2026-09-15；分支：`cis-tree-frontier`。本文只描述当前已实现与待测事项，不把未跑 NPU 的策略称为性能优化。

## 执行边界

一棵树是一个 `job_id + step_index`，包含 C 个 candidate parent、各自 R 个 rollout child，以及 candidate barrier、rollout barrier。树保存依赖、ready bundle、活跃 child 和 KV lease 生命周期；**实际 forward batch 仍由 vLLM 原生 Scheduler 从不同树的 branch request 中组成**。这既不是强制整组独占 NPU，也不是把 `C×R` 个 rollout 物理并入一条 vLLM request。

当前实现复用已经验证功能正确的 v0.18 物理 fork-waiter 和 compact-waiter patch。child 的 frontend identity 预注册，但不执行模型，parent 完成后才补入 candidate tokens、继承物理 KV，然后进入 waiting queue。新策略只改变 parent bundle **何时进入 waiting**，不改 C/R/B/L、seed、reward、resample 或 token 选择。

## 策略

- candidate 阶段未闭合且引擎已有足够 runnable work 时，ready child 保持停放；只有 running/queue 均不足时提前释放，避免重现即时 fork 的混合 batch 负结果。
- candidate 阶段闭合后，按实际 MNS、活跃树数、已放行 child 数和 runnable deficit 计算每棵树的弹性 branch quota；每次 scheduler tick 至多释放配置数量的 sibling bundle。`running+waiting` 达到目标容量时不放行。接近目标时允许一个完整 sibling bundle 越过目标至多 `R-1` 条，实际运行容量仍由原生 vLLM MNS 限制。此前的硬边界使 `running=204/256` 时无法放行任何宽度为 3 的 bundle，已修复并加入测试。
- 优先补位只差最后一两组 child 的树，其他树按活跃 child/配额比例及进入顺序竞争。KV usage 达到 high watermark 时通常停止放行，但关闭且无活跃 child 的树至少允许一组进展，避免纯 admission 死锁。
- 原生 vLLM scheduler 仍负责 MNS/MBT、batch shape、APC、KV allocation 和 preemption；插件不复制它的 token scheduling loop。
- `tree-frontier.jsonl` 记录每个 candidate 完成、每个 sibling 放行、rollout 完成，以及每秒一次的待放行 frontier stall。与算法 trace、容器日志、benchmark 和 NPU profile 结合才能判断 queue、barrier、lease 与 batch 形状。

当前默认 `target_fraction=1.0`、`kv_high_watermark=0.92`、`max_bundles_per_tick=8`，只作为实验起点，**不是最优值**。`0.80` 在首次 B 机 P1 短试中发生明显欠填，不应被当作树设计的性能结论。

## 质量闸门

1. 目标 vLLM-Ascend 0.18 镜像必须能同时应用 fork、compact-waiter 和 KV-capacity patch，并导入 `CISTreeFrontierScheduler`。
2. policy 与 scheduler 契约测试覆盖 terminal candidate 零 rollout、candidate barrier、欠填提前补位、bounded sibling release、优先 barrier-close 树、parent/child completion 计数、parked child 被取消和错误 group metadata。
3. 真实 4-job smoke 要达到全部 CIS job 成功、`unresolved_children=0`、被激活 child fork hit 接近 100%、非 terminal parent 有 R 个 child、terminal parent child 零 forward、没有持久 KV lease 或 OOM。任何失败先修正确性，不进入性能 A/B。
4. 同设备顺序运行 `job-runtime-kv-budget` control 与 `tree-frontier`；P1 使用固定输出工作量，P0/P2 用真实 EOS。相同模型、轨迹顺序、算法参数、runtimeKV、priority、MNS/MBT 和 graph；报告 jobs/s、generated tokens/s、实际 prefill/APC、Job/Step/Barrier mean/P95、preemption、fork hit/miss、mixed prefill+decode、running/queue 曲线。输出内容可能因顺序分叉，不能只凭 jobs/s 归因。
5. 先 `smoke`、`short-p1`，再 `full-p1`、`full-p0`、`full-p2`。若较旧 Step control 回退，必须先看 trace：是本策略欠填/过早注入、KV lease 压力、额外 child lifecycle、prefix miss，还是算法生成工作量变化。每种原因对应不同修复，不能把第一版回退直接判定树思想无效。

## 当前状态与局限

已完成：独立纯 policy、vLLM adapter、实验入口；本机纯策略 9 个测试通过，目标镜像里组合 patch 与 6 个 scheduler 契约测试通过（近目标放行边界修复后已复验）。2026-09-15 在 A 机 2/3 号卡跑完 4-job NPU smoke：4/4 成功，`unresolved_children=0`，无 KV preemption/OOM。容器日志统计 parent capture 180 次、child fork hit 540 次、`no_hint` 36 次；后者准确对应 12 个以 stop token 结束的 terminal parent 各 3 个 child，运行时让 child 零计算结束，所以非 terminal child 为 540/540 物理 fork hit。树 trace 有 candidate completion 192、sibling bundle admit 192、rollout finish 576；APC 命中约 98.85%。这轮 4-job smoke 未饱和，不能用于性能结论。

16-job 成对 A/B 于 A 机 2/3 启动后，PACE 进程进入同一对卡；为避免影响他人和实验污染，主动停止自己的 control 容器（exit 137）。该不完整 run 不进入结果表。随后在 B 机 1/2 运行；仅有常驻 boom MCP 的 110 MB 空闲设备上下文，未停止或修改该进程。预检脚本已修复 `npu-smi` 进程行解析，只有明确记录的空闲 PID 可通过；如出现其他计算进程，实验无效。目标镜像缺少 pytest；已有 pytest-based 算法测试目前未在该镜像跑，不能把它们写作已验证。

### B 机 P1 短试结果（固定工作量，非最终性能结论）

模型、双卡 TP2、C8/R3/B128/L512、16 个同序工作、固定生成长度、MNS256/MBT32768、runtime-KV admission、`job_fifo` 和 graph 均一致。控制组为现有 `job-runtime-kv-budget`，实验组为 `tree-frontier` + compact waiters + 物理 KV fork。两组均 16/16 成功、1664 个 engine request、360448 个生成 token、0 preemption；输出 hash 不同，不能声称生成内容逐字相同。

| 指标 | 控制组 | 修正后 tree | tree 相对控制组 |
|---|---:|---:|---:|
| jobs/s | 0.04498 | 0.03887 | -13.6% |
| Job mean / P95 | 308.4 / 350.3 s | 378.0 / 392.1 s | +22.6% / +11.9% |
| Step mean / P95 | 66.4 / 170.2 s | 80.3 / 152.3 s | +20.9% / -10.5% |
| rollout barrier-tail mean / P95 | 3.49 / 27.51 s | 28.84 / 52.06 s | +727% / +89.3% |
| engine queue mean / P95 | 7.65 / 54.72 s | 24.33 / 83.88 s | +218% / +53.3% |
| forward token slots/s | 2022.6 | 1578.5 | -22.0% |
| actual prefill tokens | 360664 | 291040 | -19.3% |

Tree 物理 KV fork `1407/1440` 次命中（97.7%）；33 次 `no_hint` 与 terminal parent 相关，未发现 unresolved child。先前的 `target_fraction=0.8` 实验在 `running=204/256`、无 waiting、仍有大量 ready child 时欠填，已停止并保存于 `tree-f080-underfill/`，不进入上表。修正后 tree 曾达到 `running=256/256`，但整体 forward 工作率仍下降，barrier 和 engine queue 变长；**第一版树级 frontier 不优于已有调度，也不能凭物理 fork 命中率证明有效**。目前只能定位到 admission/queue 与 batch 执行效率合计抵消了 prefill 节省；具体占比尚缺对照 NPU profile，不能把原因单独归给 attention 或 fork。下一步若继续树方案，先做同控制组下的 `scheduler-only` 与 `fork-only` 拆分，并对齐 batch-shape/NPU trace，而不是直接扫更多阈值。

### 跨模型边界

`job/step/candidate/rollout/barrier` 和 ready frontier 是 Conditional IS 的语义，与模型权重无关；MNS 相对放行策略也不写死 Qwen 参数。但当前物理 KV fork patch 在 `num_kv_cache_groups != 1` 时直接跳过；runtime-KV 估算也按单组 KV block 容量。DeepSeek-V4-Flash 在 vLLM-Ascend v0.18 使用 Compress-4/Compress-128 混合注意力和多组 KV cache，不是把 Qwen TP2 manifest 换个模型名就能验证。迁移时至少要重做 group-aware fork/资源估算，核对 tokenizer、tool parser 与 reward statistics，再用该模型自己的双/多卡基线 A/B。若仅保留逻辑树调度而回退为普通 prefill，可以复用调度接口，但不能宣称复用了当前物理 KV 收益或保留了当前性能数据。

仍有两个产品化风险要特别看：其一，child identity 仍预注册，所以这不是零 lifecycle 成本的动态 tree-root；其二，旧补丁原本对 parked child 取消没有完整 KV hint 计数。当前 adapter 已补 child-cancel→parent remaining/ready bundle/KV hint decrement，并通过 parent 完成后取消的目标镜像状态测试；**尚未用真实 NPU 超时/取消流复核**。该路径在真实故障注入通过前不得宣称生产级错误恢复。

实施入口：`experiments/swebench/run_cis_tree_frontier_ab_remote.sh`。A/B 服务器的新工作目录均为 `/data/disk/wangzili/inference-scaling-mini-swe-agent-tree-frontier-20260915`；A 机 artifact 目录：`/data/disk/wangzili/cis-tree-frontier-ab-20260915`。不修改、停止或复用其他人的容器和文件。
