# Conditional IS Packed Forest Attention 实验记录

日期：2026-09-12

## 1. 问题与边界

Conditional IS 的每个 candidate 会派生 `R` 条 rollout。普通 vLLM 将这些
rollout 当作独立 sequence；APC 可以避免重复 prefill，但 decode attention
仍会让每条 query 分别读取相同的长 candidate 前缀。P0 `C15/R3` 每个 step
有 15 个三分支 sibling group，这种两层树形 KV 共享是普通请求没有的。

本实验不改变 `C/R/B/L`、采样 seed、token 集合或 resample 次序，只改变
decode attention 对已经存在的物理 KV block 的读取方式。

## 2. 从失败的两 FIA 到单 FIA packing

最初按 Hydragen 形式将共享前缀和独立 tail 分别执行 FIA，再使用 LSE 做 exact
online-softmax merge。这个实现虽数学等价，但第二次 kernel launch 和 merge
开销过高：8K P0 只有约 `0.53x`，32K P0 才达到约 `1.19x`，不能作为 6-8K
主 workload 的实现。

新的实现为每个 candidate 构造一行逻辑 paged block table：

```text
[shared candidate prefix][rollout tail 0][rollout tail 1]...[rollout tail R-1]
```

将 `R` 条 query 放入连续 BSH 的 query sequence 维度，并使用布尔 mask 令 query
`r` 只访问共享前缀和 tail `r`。因此一个原生 FIA 同时完成整个 sibling group，
不需要复制 KV、第二次 FIA 或 LSE merge。mask 只描述逻辑可见性，Q/K 已包含
原始位置编码，所以结果仍对应每条 rollout 的原始 attention。

## 3. 早期通用 FIA primitive 结果

设备为 Ascend 910B3，Qwen3-Coder-30B-A3B 形状为 16 个 query heads、2 个 KV
heads、head dim 128、BF16、block size 128。下表固定 candidate suffix 128、
rollout unique tail 256，数值为相对通用 FIA 的 P50 speedup：

| 共享 trunk | C4/R3 | C8/R3 | C15/R3 |
|---:|---:|---:|---:|
| 4K | 0.783x | 0.949x | 1.084x |
| 8K | 0.839x | 1.007x | 1.229x |
| 16K | 0.976x | 1.296x | 1.461x |
| 32K | 1.190x | 1.400x | 1.721x |

unique tail 128/384 的矩阵呈现同一趋势。所有测试的最大输出绝对误差不超过
`9.77e-4`，典型值为 `2.44e-4` 到 `4.88e-4`；LSE 最大误差不超过
`1.91e-6`，符合 BF16 kernel 的数值差异。

4D batch-specific mask 已通过验证。8K C15/R3 中，共享 2D mask 为 `1.216x`，
每组独立 4D mask 为 `1.217x`，未观察到额外开销。这允许同一 forward 中的
candidate group 使用不同的实际 prefix/tail 长度。

这些结果用于建立共享读取的形状规律，不能直接当作 vLLM 端到端收益。后续源码与
trace 检查确认，vLLM-Ascend v0.18 的正常 pure decode 使用更快的私有
`torch_npu._npu_paged_attention`，而非这里的通用 FIA。所有正式结论必须以 native
paged attention 为 baseline。

## 4. 256-request 通用 FIA mixed decode 结果

下表固定 8K trunk、128 candidate、256 unique tail，以普通请求补齐 batch 256。
它模拟真实 continuous-batching forward，而不是只测一个孤立 CIS step。

| 配置 | 同时存在的完整 CIS step | packed rollout 数 | 普通请求数 | P50 speedup |
|---|---:|---:|---:|---:|
| P0 C15/R3 | 1 | 45 | 211 | 1.044x |
| P0 C15/R3 | 2 | 90 | 166 | 1.223x |
| P0 C15/R3 | 3 | 135 | 121 | 1.497x |
| P0 C15/R3 | 4 | 180 | 76 | 1.803x |
| P0 C15/R3 | 5 | 225 | 31 | 2.317x |
| P1 C8/R3 | 1 | 24 | 232 | 0.975x |
| P1 C8/R3 | 2 | 48 | 208 | 1.051x |
| P1 C8/R3 | 4 | 96 | 160 | 1.285x |
| P1 C8/R3 | 6 | 144 | 112 | 1.551x |
| P1 C8/R3 | 8 | 192 | 64 | 1.975x |
| P1 C8/R3 | 10 | 240 | 16 | 2.686x |
| P2 C4/R2 | 4 | 32 | 224 | 1.000x |
| P2 C4/R2 | 8 | 64 | 192 | 1.106x |
| P2 C4/R2 | 16 | 128 | 128 | 1.320x |
| P2 C4/R2 | 24 | 192 | 64 | 1.513x |
| P2 C4/R2 | 32 | 256 | 0 | 1.810x |

结果说明收益不应由 `P0/P1/P2` 名称或固定 candidate 数硬编码。当前动态门控
使用：

```text
estimated_saved_kv_token_reads
  = sum(shared_prefix_tokens * (sibling_count - 1))
```

初始阈值为 `196608` token-reads，约等于 24 个 sibling group 在 8K 前缀下
避免两次重复读取。低于阈值时保留原始 attention，避免 P1 单 step 这类负收益。

同样，这一节是相对通用 FIA 的探索性结果。它说明完整 sibling group 密度是关键，
但表中的 speedup 不应写成相对 native paged attention 或完整服务的收益。

## 5. Native paged attention 复测

改为与 `_npu_paged_attention` 比较后，单个或少量 sibling group 往往更慢；只有
同时存在较多完整 group，且共享 prefix 足够长时，packed Forest 才出现稳定机会。
代表性 trace-shaped 结果：

| 场景 | 完整组/查询数 | 相对 native PA |
|---|---:|---:|
| prefix 9728、unique 128、R3 | 22/66 | 1.915x |
| prefix 9728、unique 640、R3 | 22/66 | 1.745x |
| prefix 16384、unique 640、R3 | 11/40 | 1.037x |
| prefix 24576、unique 512、R3 | 15/45 | 1.954x |

P2 R2 的许多 16K/32K 形状仍低于 1x。因此正式路径必须动态门控，不能按 P0/P1/P2
名称硬编码。

## 6. vLLM-Ascend v0.18 集成

当前分支已实现以下代码路径：

1. Conditional IS 为 rollout 请求附加结构化 `group_id/branch_index/group_size`。
2. Async vLLM backend 仅在开关打开时将元数据写入 `SamplingParams.extra_args`。
3. NPU model runner 在纯 decode forward 中检查完整 sibling、真实物理 block ID
   共享和动态收益阈值。
4. 命中时该 forward 退出 FULL graph，将 sibling 通过 packed BSH FIA 执行，普通
   请求继续使用 native paged attention，并按原 query index 合并 output。
5. PCP、DCP、speculative decoding、sliding-window、incomplete sibling 或物理前缀
   不一致时自动回退。

v3 不再由 Python materialize 巨型 bool mask，只将 shared/tail lengths 传到设备，
在 NPU 上用缓存 `arange` 构造 mask；all-Forest batch 还具有 direct-return fast
path。最新 targeted tests 为 `135 passed`；补丁 dry-run 和修改后源码 py_compile
均已通过。

## 7. 严格同 namespace 端到端结果

为了固定 request-local seed，A/B 除相同 `--seed` 外还必须使用相同
`--run-namespace`。以下固定 P0/P1/P2 的 workload、顺序、C/R/B/L、workers 和
engine 参数。

### 7.1 FULL graph 与 dual graph control

普通 decode 的真正最佳路径是 `FULL_DECODE_ONLY + Npugraph_ex`。为允许 Forest
forward 退出图，原型使用 `FULL_AND_PIECEWISE`；仅这一变化已经产生明显开销：

| 配置 | 路径 | jobs/s | FTS/s | Job mean/P95 |
|---|---|---:|---:|---:|
| P0 | FULL | 0.166006 | 2896.83 | 151.51/292.47s |
| P0 | dual control | 0.146693 | 2602.53 | 134.85/274.83s |
| P1 | FULL | 0.243368 | 3708.91 | 92.89/171.75s |
| P1 | dual control | 0.214017 | 3307.84 | 105.49/183.94s |
| P2 | FULL | 0.319405 | 5364.44 | 97.59/166.00s |

P0/P1 的 FULL jobs/s 分别高约 13.2%/13.7%，FTS/s 高约 11.3%/12.1%。因此
Forest 只和 dual control 打平仍不能称为最终收益。

### 7.2 Forest v2

| 配置 | 路径 | jobs/s | FTS/s | Job mean/P95 |
|---|---|---:|---:|---:|
| P0 | dual control | 0.146693 | 2602.53 | 134.85/274.83s |
| P0 | Forest v2 | 0.138433 | 2429.69 | 170.56/314.20s |
| P1 | dual control | 0.214017 | 3307.84 | 105.49/183.94s |
| P1 | Forest v2 | 0.182402 | 2814.62 | 104.13/214.42s |

v2 的主要额外开销是每个 decode step 在 Host 构造并 H2D 传输 50 万到 100 万以上
bool，以及 mixed batch 的 native/Forest split 和 gather/reshape/scatter。

### 7.3 Forest v3

同 namespace P0 4-job smoke 中，v3 相对 v2 的 jobs/s/FTS/s 提升
`11.6%/10.6%`，Job mean/P95 降低 `27.6%/16.0%`，证明去掉 Host mask 是有效
修复。

P0 64-request v3：

| jobs/s | FTS/s | Job mean/P95 | preemption |
|---:|---:|---:|---:|
| 0.143149 | 2540.65 | 161.22/249.15s | 0 |

它相对 dual control 的 jobs/s/FTS/s 均约 `-2.4%`，P95 `-9.3%`；但相对真正 FULL
baseline 的 jobs/s/FTS/s 仍约 `-13.8%/-12.3%`。因此 Python/FIA bridge 已接近
自己的同图 control，却尚未跨过丢失 FULL graph 的成本。

P2 v3 为 0.360361 jobs/s、4865.12 FTS/s、Job mean/P95 88.96/152.68s，Forest
因 R2 和 saved-read 门槛基本未激活。其 jobs/s 不能单独解释为收益，因为实际生成
work 不同；FTS/s 相对 P2 FULL 低约 9.3%，符合 dual graph 开销判断。

P1 v3 在服务器 A 补跑 64-request 时也没有 Forest activation，说明严格门控会保护
完整 sibling 密度不足的 P1 batch。由于 P1 的 FULL/dual control 位于服务器 B，
该轮绝对性能不做跨机精确比较。

### 7.4 Forest decode window

为验证 `93.26%` mixed prefill/decode 是否能通过树语义改善，增加了一个窄 scheduler
实验：当当前 running decode 已包含足够完整 sibling group 时，最多连续 4 个 tick
不接纳新的 waiting prefill，之后强制放行一次避免饥饿。

| workload | 路径 | jobs/s | FTS/s | Job mean/P95 |
|---|---|---:|---:|---:|
| P0 4-job | Forest v3 | 0.137397 | 2041.37 | 16.67/27.02s |
| P0 4-job | Forest v3 + window4 | 0.203629 | 2962.55 | 14.39/19.02s |
| P0 64-request | Forest v3 | 0.143149 | 2540.65 | 161.22/249.15s |
| P0 64-request | Forest v3 + window4 | 0.142763 | 2537.63 | 150.65/265.25s |

4-job smoke 的大幅正值没有在饱和 workload 复现。64-request 的 jobs/s/FTS/s 为
`-0.27%/-0.12%`，mean `-6.6%`、P95 `+6.5%`；整轮只触发一次 scheduler window。
该方向判定为无稳定吞吐收益，不再扩窗口长度或阈值网格。

当前结论是：共享 KV attention 的 kernel 机会成立，但正式实现必须是接受紧凑
segment metadata、可在 FULL graph 内 capture 的融合 CANN op。继续打磨 Python
mask/FIA 兼容层不是目标架构。

## 8. 与既有调度优化的关系

Step cap/step_fifo 改善的是 admission、barrier 和 batch shape；Packed Forest
Attention 改善的是同一 decode forward 内的共享 KV 读取。前者还能提高完整
sibling 同批出现的概率，后者反过来为 step locality 提供更强的硬件收益，两者
是互补关系。

Direct KV fork 解决 candidate 完成到 rollout admission 间的物理 KV 交接，但
APC 已使普通路径具有很高 prefill 命中，端到端增益不稳定。Packed Forest
Attention 针对的是 APC/fork 都不会消除的 decode 阶段重复 HBM 读取，因此是
当前更值得验证的算法专属方向。

Forest decode window 的负结果进一步说明：现有 sibling 已高度同批，单独阻止新
prefill 不能稳定提高饱和吞吐。scheduler 后续应等待融合 CANN op 先成立，再依据新
trace 判断是否仍需 tree-aware batch shaping。

## 9. 原始数据

- `cis_packed_candidate_group_u128_20260912.json`
- `cis_packed_candidate_group_u256_20260912.json`
- `cis_packed_candidate_group_u384_20260912.json`
- `cis_packed_mixed_decode_p0_8k_u256_20260912.json`
- `cis_packed_mixed_decode_p1_8k_u256_20260912.json`
- `cis_packed_mixed_decode_p2_8k_u256_20260912.json`
- `cis_native_pa_p0_8192_20260912.json` 等 native PA 对照矩阵
- `cis_native_pa_trace-g22-b66-u128_20260912.json` 等 trace-shaped 对照
- `cis_packed_forest_e2e_20260912.json`

CANN 的 FIA 接口存在 shared-prefix 参数，但官方约束中 shared-prefix mode 不支持
PageAttention；vLLM-Ascend v0.18 也未将其接入 paged KV decode。本实现因此使用
原生 FIA 已支持的 block table 与 batch-specific mask，而不是假设框架已有对应
能力。
