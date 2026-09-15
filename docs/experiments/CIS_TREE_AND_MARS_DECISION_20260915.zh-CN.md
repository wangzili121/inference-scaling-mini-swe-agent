# Conditional IS 调度单位与 MARS 组合的阶段性决策

日期：2026-09-15；实验分支：`cis-mars-bridge`。这里的目标是确定执行边界和下一轮实验，不把尚未验证的完整树调度写成既有收益。

## 1. 什么已经验证，什么还没有

- 现有有效方案是 step/job 级 admission 与 priority 映射。candidate 和 rollout 仍是独立 EngineCore request；step 是逻辑组，不是强制同 batch 或独占 NPU。
- EngineCore 物理 fork 已验证：parent 的 KV 可被 child 直接采用，激活 child 命中率 100%。但 P0 即时 fork 对最佳 Step control 的 jobs/s 仅 +0.2%、FTS/s -5.6%；P1 即时 fork jobs/s -5.8%，P2 -9.1%。P0 compact waiter 已去掉 4300 万 prompt-token placeholder payload，FTS/s 仍 -5.9%。因此不能再称“真正 fork 从未试过”，也不能把 fork 命中当作端到端增益。
- 尚未实现的是**一个持有全局树依赖/预算/ready frontier 的调度策略，与受控原地 fork、跨树 continuous batching、barrier 进度和公平性联动**。它与已经失败的“完成即无限派生 child”以及“同胞小批量限流”不同，但目前没有性能结论。
- 把整棵树做成不可拆分的一条执行 request 会让阶段依赖、可变生成长度和 `C + C×R` 分支占用与 vLLM 的 MNS/MBT/KV allocator 冲突；已有 sibling K8 小批量对 P0 jobs/s 约 -17.8%。默认架构应为 tree 作为逻辑 admission/accounting/KV ownership 单位，ready branch 作为可跨 tree 合批的执行单位。

## 2. 本轮 MARS 组合实验

MARS 源于 Muyuan MR/6 `9396647`，本轮只装载其 `MarsScheduler`，**未启用 LMCache-Ascend CPU KV offload，也未运行其 agent/tool 在线控制器**；所以这是“将其通用 EngineCore scheduler 套到 CIS 子请求”的消融，不是 MARS 完整产品 benchmark。所有 16-job 运行在服务器 A 同一对 910B3 卡 4/5 顺序执行：Qwen3-Coder-30B-A3B-Instruct BF16，vLLM-Ascend 0.18，TP2、MNS256、MBT32768、HBM0.90、APC、chunked prefill、native sampler、full graph，P1 `C8/R3/B128/L512`、固定长度、同一公开轨迹顺序/seed。每个 arm 均成功，生成 token 数 360448。

| 16-job arm | jobs/s | Job mean / P95 | prefill tokens | preemption |
|---|---:|---:|---:|---:|
| flat vLLM | 0.04648 | 324.7 / 339.5 s | 360664 | 0 |
| MARS scheduler only, window256 | 0.03634 | 422.0 / 438.1 s | 1593560 | 52 |
| CIS outer runtime-KV f0.80 + job FIFO | 0.04488 | 309.6 / 351.5 s | 360664 | 0 |
| MARS+CIS, MARS-level first | 0.03262 | 437.6 / 481.9 s | 1589080 | 58 |
| MARS+CIS, CIS-priority first | 0.03239 | 436.9 / 477.2 s | 1716056 | 49 |

另一次同卡 MARS-only window128 jobs/s 0.02369、prefill 2965720、preemption122，单纯缩小活跃窗口并没有救回这个负载。轻负载下 CIS admission 也不能保证增加吞吐。注意此轮 bridge 还替换了 MARS 原本 FCFS 的 overflow queue；后续代码已纠正为保持 MARS 原行为，但**旧的组合 arm 不能被当作只增加 CIS priority 的严格消融**，MARS-only 负结果则不受此影响。

32-job 只复验 flat vLLM 与 CIS outer（相同设备、顺序、不与另一个模型并行），每个 arm 的 EngineCore request 数 3328、generated tokens 720896、成功率 100%。

| 32-job arm / round | jobs/s | Job mean / P95 | prefill tokens | preemption |
|---|---:|---:|---:|---:|
| flat / 1 | 0.04267 | 749.0 / 749.9 s | 2782888 | 0 |
| CIS / 1 | 0.05345 | 412.5 / 595.3 s | 627240 | 0 |
| flat / 2 | 0.04287 | 745.2 / 746.0 s | 2699816 | 2 |
| CIS / 2 | 0.05292 | 410.5 / 596.9 s | 627368 | 0 |

CIS 对两轮 flat 的 jobs/s 分别约 +25.3%/+23.4%，Job P95 约 -20.6%/-20.0%，但它缩小了同时进行的 step 数，并把一部分等待移动到外层：round1 的 step admission wait mean 约 23.5 s、P95 约 274.5 s；step E2E P95 从 flat 的约 308 s 升到 CIS 的约 402 s。不可只报告吞吐与 Job P95。

**归因限制**：固定生成 token 个数不等于固定生成的 token *内容*。不同排队顺序下输出 hash 不同，后续 selected candidate/prefix 可能分叉；flat 产生约 270-278 万 prefill，而 CIS 仅约 63 万。重复轮次强化了端到端现象，但不能把全部 +23-25% 宣称为 tree priority 的精确因果贡献。下一轮应记录首次分叉位置及每阶段实际 token/hash、APC miss、batch shape，并作顺序交叉复验。

原始 artifact（服务器 A）：`/data/disk/wangzili/cis-mars-bridge-ab-20260915/{smoke,full,paired-r2,window128}`；每 arm 均有 `benchmark.json`、请求和算法 trace、容器日志、环境/launch 索引。本地实施入口为 `experiments/swebench/run_cis_mars_bridge_ab_remote.sh`；可选桥位于 `src/inference_scaling/arllm/backends/mars_bridge.py`。当前 bridge 仅作实验消融，不能宣称其策略已胜出。

## 3. 最小下一轮实验与准入门槛

1. **先确认真正需要 EngineCore 内树状态的瓶颈**：在相同 32/64-job 饱和负载和真实 EOS 下，与最佳 outer runtime-KV、静态最优和 flat 同卡比较，标注每 step 的 candidate 完成到 child admit、rollout 最后一条到 barrier、KV/APC missed tail、scheduler running/waiting/branch 组成。另加入 16K-32K、32K-64K 的真实轨迹，避免只为当前 6-8K 刷分。
2. **两种执行单位的有限 A/B**：A 现有 outer group+native priority；B outer group+原地 fork 但沿用原有全 candidate barrier；C 逻辑 tree frontier 按 KV blocks、ready branch 数和 barrier 剩余工作控制 child 的 release，同时允许跨树合批。P0/P1/P2 分别测 jobs/s、generated tokens/s、Job/Step/Barrier mean/P95、有效 prefill、preemption、batch shape。C 若不能在 P0/P1 至少不低于 A 且在一个场景取得可复现的 >5% job 级收益，就不承担核心 fork patch 的产品维护成本。
3. **MARS 与 CIS 分层而非重复门控**：MARS Muyuan MR/6 的 `OnlineMarsController` 管 `program_id/round_id`、tool waiting 和跨轮 KV；CIS 管一次模型调用内部 step/candidate/rollout。当前 MARS online API `/v1/mars/generate` 只接受普通单次 generation，不能直接代替整次 CIS 调用。组合实验必须使用并发 mini-SWE-agent+Docker，多轮 agent/task 情况，并约定只由上层做 agent admission、CIS 在获准调用内做 tree budget；如果使用 MARS scheduler+offload，必须共用 KV telemetry 或只在一层做容量门控。四格：native、MARS 完整组件、CIS、两者分层；同时报告 offload 读写、KV 命中和成本。当前 scheduler-only 在冻结 CIS workload 的负结果不代表 MARS 整体无用。
4. **四卡**：在 `2×TP2` 先把整棵树绑定单个 instance 以保留 APC/可能的 parent-child KV；对多 job 测静态均分、least-outstanding、实际剩余 branch-token 工作量与 KV 余量感知的路由。仅在单棵重型 tree 的 job latency 已成为瓶颈且两卡空置时，试 candidate-subtree 跨 instance 分区；winner state/重采样和下个 step 的 KV 重建/传输必须计入总时间，不能只报告并行 forward。

## 4. 产品形态

优先拆出轻量 `cis_scheduler` wheel：版本化 `job/step/node/parent` 元数据、外层 claim 生命周期、KV/branch 双预算策略、可选 vLLM-Ascend 0.18 priority adapter、统一 trace。非 CIS 请求维持 vLLM 行为。选中的 EngineCore fork、跨树 frontier、MARS bridge 和多卡 route 作为**分别受 A/B 控制的可选适配器**，不强行把 Muyuan MARS 的 scheduler 和 CIS 的 `scheduler_cls` 同时启用。若树状态策略验证有效，再实现单一复合 scheduler hook；否则保留低侵入 outer+priority 版本，避免为插件形式牺牲端到端效果。
