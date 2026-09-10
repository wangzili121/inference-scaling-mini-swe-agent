# Conditional IS General Profiling 总结（2026-09-10）

## 1. 范围

模型固定为 `Qwen3-Coder-30B-A3B-Instruct` BF16，runtime 为 vLLM-Ascend
v0.18 MRV1。General baseline 已启用 persistent AsyncLLM、continuous batching、
APC、chunked prefill、request-local seed、native categorical sampler、融合 reward
statistics、CPU binding 和 `FULL_DECODE_ONLY + Npugraph_ex`。

部署固定为：

- 两卡：单个 `TP2`，`MNS=256`、`MBT=32768`、memory `0.90`、partial prefill
  `(1,1)`；P0/P2/P3 workers=32，P1 workers=24。
- 四卡：两个相同 `TP2` instance，round-robin；总 workers 分别为两卡的两倍。
- `PP2` 已做 capability 与调优前验证，未优于 TP2；P/D 因 v0.18 镜像缺失
  `ascend_transport.so` 不具备可用 capability，没有伪造结果。

## 2. 八组正式结果

性能只引用无 profiler pass；Service/Torch pass 用于归因。

| Profile | 两卡 jobs/s | 两卡 P95 | 两卡 slots/s | 四卡 jobs/s | 四卡 P95 | 四卡 slots/s | work scaling |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 C15/R3 logprob | 0.156288 | 313.91s | 3,706.8 | 0.270663 | 166.85s | 4,761.1 | 1.284x |
| P1 C8/R3 logprob | 0.289787 | 172.26s | 4,322.9 | 0.356120 | 135.47s | 5,552.6 | 1.285x |
| P2 C4/R2 logprob | 0.405948 | 136.31s | 5,541.0 | 0.538301 | 98.22s | 7,273.2 | 1.313x |
| P3 C15/R3 Consilience | 0.160988 | 344.65s | 4,297.1 | 0.363746 | 167.18s | 6,345.2 | 1.477x |

八组 none baseline 全部 100% 成功、零 OOM、零 KV preemption。不同 reward
改变了采样轨迹与实际生成工作量，因此 jobs/s 的表观扩展不能脱离 forward slots/s
解释。

## 3. 代表性 P0 四卡归因

P0 四卡 Torch pass 包含 224,786 条采集窗口内的 kernel 记录：

- 四个 rank 的 NPU busy ratio 中位数为 `90.51%`；逐 rank 为
  `94.20%/94.16%/86.87%/83.60%`（完整 profiler span）。
- 暴露 HCCL 占各 rank profile window 的 `16.52%/16.75%/21.48%/18.04%`，
  汇总为 `18.37%`；通信与计算重叠很少。
- `FusedInferAttentionScore` 四 rank 累计约 `23.38 device-seconds`，稳定为第一
  kernel；`hcom_allReduce` 累计约 `8.98 device-seconds`。
- Service pass 的 296 个 batch 中 P50/P95/P99 batch size 均为 `256`，mixed
  prefill/decode batch 比例为 `57.77%`。设备已被喂满，问题不是继续加普通并发。
- endpoint 平均完成时间仍有 `18.15%` skew，说明相同 job 数不等于相同实际工作。
- APC token hit 为 `97.49%`，`score_calls=0`；普通 prefix cache 与 reward
  rescoring 都不是主要剩余瓶颈。

## 4. Mixed-stage 补采

检查 epoch 微秒时间戳后确认，现有 P0-P3 的 Torch 10 秒窗口几乎全部落在
candidate 阶段；P0 四卡窗口中 profile-0 约有 30–31 个 candidate job 活跃，
profile-1 为 32 个，rollout 只短暂出现一次。已有原始 trace 足以分析 candidate
期间的 kernel/HCCL/空洞，但不足以比较 rollout 与阶段切换的 NPU 形状。

因此新增了一遍 P0 四卡延迟触发采集：ramp-up 从 15 秒改为 60 秒，采集 15 秒。
64 请求全部成功、零 preemption。该窗口实际同时覆盖两类阶段：profile-0 的
candidate 峰值为 27，rollout 峰值为 13，二者共同活跃 11.13 秒；profile-1 的
candidate 峰值为 6，rollout 峰值为 13，二者共同活跃 7.02 秒，随后有 7.98 秒
rollout-only。

补采得到 396,158 条窗口内 kernel 记录。按 1ms 原始 kernel 覆盖与算法绝对
时间戳联结后，profile-1 两个 rank 在 mixed 阶段的平均 busy 为
`71.8%`，rollout-only 为 `82.3%`，下降 `10.5` 个百分点；这是 candidate 与
rollout batch shape 混合会降低执行效率的直接证据。`FusedInferAttentionScore`
四 rank 累计时长约 `41.03 device-seconds`，相当于 device-busy union 的
`75.4%`；APC 已达 `97.69%`，说明共享前缀 prefill 已被缓存，但分支 decode 对
共享 KV 的读取仍是主要成本。

mixed-stage 窗口的暴露 HCCL/profile-window 为 `6.10%`，明显低于此前
candidate-only 窗口的 `18.37%`。因此通信压力具有阶段性，不能根据单一窗口把
HCCL 写成全程首要瓶颈。补采只用于补齐阶段归因，不重新用于选择部署参数。

## 5. 算法专属 A/B

| 方案 | work-normalized 变化 | P95 | preemption | 结论 |
|---|---:|---:|---:|---|
| candidate completion streaming | -12.57% | 367.35s | 39 | 淘汰 |
| per-job bounded=15 | -15.77% | 399.17s | 30 | 淘汰 |
| global frontier=128 | -4.08% | 400.45s | 9 | 淘汰 |
| global frontier=192 | +0.78% | 425.42s | 47 | 收益不足且尾延迟恶化 |
| global frontier=240 | -3.38% | 372.65s | 52 | 淘汰 |

candidate streaming 的中位提前量只有 `5.85ms`。固定 B 使 candidate 接近同步
结束，host 侧拆批不能形成有意义的阶段 overlap；三类 admission 方案已给出一致
负结果，不再继续扫 capacity/batch size。

静态 whole-job routing 在 P1 提升 `22.65%`，在 P3 却下降 `11.39%`。随机 EOS
令输出长度不可预知，静态 prompt/C/R/B/L 估计不够稳健，只适合作为在线
remaining-work routing 的先验。

## 6. 后续建议（尚未作为收益结论）

1. `in-engine branch-on-token`：candidate 到 B token 后直接 fork R 个 block-table
   child，减少 host callback、重新 admission、prefix hash 和 request lifecycle。
   vLLM 的 `SamplingParams(n=R)` 在请求入口展开独立 EngineCoreRequest，不等价。
2. `Forest Attention CANN op`：一次读取 shared trunk，在 C×R query 间复用；同一
   candidate suffix 再在 R 个 sibling 间复用，并在片上合并 softmax/LSE。现有三
   FIA 串联只在 32K C15/R3 达到 1.333x，证明必须真正融合并动态门控。
3. `candidate-subtree parallel`：一个重型 CIS job 按 candidate subtree 分到两个
   TP2 instance，reduce 只交换 C 个权重与 winner B-token delta。
4. `online remaining-work routing`：使用已完成 token、当前 branch 数和 barrier
   余量更新代价；不再依赖已被 P1/P3 反例否定的静态估计。

## 7. 数据与可视化

- 完整原始路径与各 pass 记录：
  `docs/experiments/SWEBENCH_NPU_VALIDATION_20260909.zh-CN.md`
- scheduler 机器可读结果：
  `docs/experiments/data/cis_scheduler_ab_20260910.json`
- 可复现导出器：`experiments/swebench/export_profile_dashboard.py`
- mixed-stage 原始目录：
  `/data/disk/wangzili/cis-mixed-profile/p0-four-card-torch`
- 交互式 viewer 同时保留完整算法 Gantt、15 秒 NPU/HCCL 忙闲、按阶段统计的
  busy/HCCL 表、120ms 的 4,556 条原始 kernel 下钻、296 条逐 batch Service
  事件，以及五项带证据的优化空间标注。

原始 `*ascend_pt`、`analysis.db`、kernel/operator CSV、Service 目录和算法 JSONL
仍留在服务器，viewer 使用的是从这些文件可重复生成的紧凑数据，不替代原始证据。
