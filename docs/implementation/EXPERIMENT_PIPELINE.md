# Conditional IS SWE-agent experiment pipeline

The current general-tuning and formal profiling protocol is specified in
[`GENERAL_TUNING_AND_PROFILE.zh-CN.md`](../profiling/GENERAL_TUNING_AND_PROFILE.zh-CN.md).

## Implemented entry points

All commands use one persistent Conditional IS service. A model call is one
complete ordinary Conditional IS job; request-level overrides are restricted to
`candidate_count`, `rollout_count`, and `block_size`.

```bash
conditional-is-serve \
  --config configs/swebench/conditional_is_smoke.toml \
  --port 8123
```

Deployment-time values can be overridden without editing the source config.
Values are parsed as JSON and the dotted path must already exist.

```bash
conditional-is-serve \
  --config configs/swebench/conditional_is_smoke.toml \
  --set vllm.max_num_seqs=192 \
  --set vllm.max_num_batched_tokens=65536
```

The trace records the outer job plus candidate, rollout, scoring, reward,
weight, resample, and block intervals. The same stage names are emitted through
PyTorch `record_function`, so they are visible in an Ascend/PyTorch profile.

## Historical reward screen

The reward screen creates one `C8/R2/B128` first-step rollout pool for each
proposal temperature and computes sequence-logprob and Consilience values from
the same compact generation-time statistics. It deliberately stops after this
reward-neutral first decision, so a sequence-logprob selection cannot bias later
blocks before Consilience is evaluated. It does not run a second generation for
the second reward.

```bash
conditional-is-reward-screen \
  --config configs/swebench/conditional_is_smoke.toml \
  --workload artifacts/workloads/pilot.jsonl \
  --temperature 0.25 --temperature 0.7 --temperature 1.0 \
  --output artifacts/results/reward-screen.json
```

## Historical C/R grid

The six `C={4,8,15}`, `R={2,3}`, `B=128` arms run against the same resident
engine and fixed request order. The result includes throughput, P95, forward
token slots, median ESS ratio, and the performance/compute Pareto set.

```bash
conditional-is-algorithm-grid \
  --workload artifacts/workloads/pilot.jsonl \
  --endpoint http://127.0.0.1:8123 \
  --workers 16 --limit 16 \
  --output artifacts/results/cr-grid.json
```

## Frozen workload

After collecting at least 128 unique successful calls, freeze disjoint tuning
and holdout manifests. Messages are not padded, truncated, or reordered after
the seeded shuffle.

```bash
conditional-is-freeze-workload \
  --trace artifacts/traces/model_calls.jsonl \
  --output-directory artifacts/workloads/swe-agent-128 \
  --seed 20260908 --total 128
```

## Two-card runtime tuning

The tuner compares TP2 and PP2, applies feature A/B, scans the coarse MNS/MBT
grid with 16/32/64-request successive halving, expands a winning upper bound,
and then refines memory, partial-prefill, and saturation concurrency. Each arm
starts a fresh engine and is checkpointed independently.

```bash
conditional-is-runtime-tune \
  --config configs/swebench/conditional_is_smoke.toml \
  --workload artifacts/workloads/swe-agent-128/tune-64.jsonl \
  --warmup-workload artifacts/workloads/warmup.jsonl \
  --devices 0,1 \
  --output-directory artifacts/results/tp2-tuning
```

Use `--dry-run` first to inspect the staged search. Hard gates are 100%
successful requests, no OOM or reported preemption, and P95 no more than 1.25
times the round minimum. The final ranking maximizes complete jobs/s.

## NPU profiling

After selecting the best two-card or four-card configuration, capture only the
measured burst through vLLM-Ascend's worker-aware profiler. The launcher loads
and warms the persistent engine first, then calls vLLM `start_profile` on every
TP/PP worker, runs the fixed burst, and calls `stop_profile` before shutdown.

```bash
conditional-is-profile \
  --config configs/swebench/conditional_is_smoke.toml \
  --workload artifacts/workloads/swe-agent-128/holdout-64.jsonl \
  --warmup-workload artifacts/workloads/warmup.jsonl \
  --devices 0,1 --workers 64 --profiler torch \
  --set vllm.max_num_seqs=BEST_MNS \
  --set vllm.max_num_batched_tokens=BEST_MBT \
  --output-directory artifacts/profiles/tp2-best
```

The output contains per-rank CPU/NPU traces from vLLM-Ascend, the fixed-burst
benchmark, service logs, and the algorithm-level candidate/rollout/reward trace.
Do not enable `--profile-memory` or `--profile-stack` in the first timeline pass;
both materially increase profiling overhead and should be separate diagnostic
runs.

## Selected-token fallback

`infra/vllm_ascend/selected_token_scoring` contains a hash-guarded vLLM 0.18
patch. It prevents the long-context FP32 full-vocabulary temporary, but remains
off by default because the Python tiled reduction was not bit-exact and did not
deliver production-quality throughput in the earlier 8K experiment.
