# CIS Scheduler

`muyuan-cis-scheduler` 为 Conditional Importance Sampling（CIS）提供压力感知的
准入与优先级策略。它将 `job -> step -> candidate -> rollout` 依赖关系显式传给
推理运行时，避免高 fanout 或长上下文下的队列拥塞、重复 prefill 和 KV preemption。

## 功能

- `CISSchedulerPlugin`：绑定 runtime，并自动管理 step 准入、异常释放、metadata 和 priority。
- `CISNode`：描述 job、step、candidate 和 rollout 的结构化元数据。
- `CISPriorityPolicy`：将同一 job 或 step 映射到运行时 priority。
- `VLLMRequestPolicy`：校验 CIS 元数据并附加到运行时请求。
- `RuntimeKVCapacityProvider`：从单 KV pool 或 hybrid/multi-pool layout 读取可用容量。
- `TokenBudgetAdmissionController`：按预计 KV 占用放行 step，支持 reservation resize、
  连续 step transition 和 pressure gate。

pressure gate 在低压力时保持运行时默认 priority；当 KV 占用或 branch fanout 达到阈值
后，才启用 CIS 分组优先级和动态准入。

```text
Conditional IS adapter
  -> CIS metadata + runtime KV capacity
  -> pressure-aware admission + priority
  -> vLLM/MARS scheduler and KV allocator
```

## 使用

```bash
python3 -m pip install plugins/cis-scheduler
```

```python
from cis_scheduler import CISSchedulerPlugin

scheduler = CISSchedulerPlugin(
    backend,
    candidate_count=8,
    rollout_count=3,
    max_num_seqs=256,
    max_active_steps=32,
)

with scheduler.step("job-7", 0, estimated_tokens=240_000) as step:
    priority, extra_args = step.prepare_request(
        "candidate",
        candidate_index=0,
        candidate_max_tokens=128,
        rollout_max_tokens=512,
    )
    # 将 priority 和 extra_args 传给对应的 vLLM 请求。

    step.resize(estimated_tokens=180_000)

scheduler.finish_job("job-7")
```

插件会从 `kv_cache_geometry` 或 vLLM `cache_config` 读取 KV 容量；若 backend 提供
`bind_cis_admission_controller()`，还会自动完成绑定。上下文退出时 reservation 会正常
释放，包括请求异常场景。

## 适用范围

该策略不依赖具体模型结构。CIS adapter 提供树状请求元数据和预计 token 需求，runtime
提供实际 KV 容量；插件据此动态控制准入，不使用与模型或上下文绑定的固定 step cap。

目前已在 Qwen3-Coder-30B TP2、Qwen3-4B TP1 及 8--32K Agent 请求上完成端到端验证。
单 KV pool 和 hybrid KV layout 均受支持；DeepSeek-V4-Flash 仍待八卡环境性能验证。
非 CIS 请求保持 runtime 默认 priority。

## 验证结果

以下结果来自 Ascend 910B3、vLLM-Ascend v0.18、`C8/R3/B128/L512`，对照组为相同
部署下的原生扁平 vLLM 调度：

| 模型与负载 | jobs/s | Job mean | Job P95 | prefill tokens | KV preemption |
|---|---:|---:|---:|---:|---:|
| 30B TP2，8 workers | `-0.10%` | `+0.09%` | `-1.53%` | `0.00%` | `0 -> 0` |
| 30B TP2，16 workers | `-0.05%` | `+0.46%` | `+0.19%` | `0.00%` | `0 -> 0` |
| 30B TP2，32-worker burst | `+9.39%` | `-35.31%` | `-9.58%` | `-60.95%` | `0 -> 0` |
| 30B TP2，0.07 bursty arrival | `+9.39%` | `-29.46%` | `-30.92%` | `-59.75%` | `0 -> 0` |
| 30B TP2，16--32K burst | `+32.75%` | `-49.32%` | `-25.02%` | `-80.84%` | `19 -> 0` |
| 4B TP1，mixed-context burst | `+26.76%` | `-53.05%` | `-22.09%` | `-79.72%` | `38 -> 0` |

低压 case 中 gate 未介入。持续 Poisson 过载下 jobs/s 为 `-1.27%`，但 Job mean/P95
分别改善 `8.02%/11.10%`；插件不保证所有负载下的所有指标同时提升。

## 测试

```bash
PYTHONPATH=plugins/cis-scheduler/src \
  python3 -m unittest discover -s plugins/cis-scheduler/tests -v
```

要求 Python 3.11+，无第三方运行时依赖。
