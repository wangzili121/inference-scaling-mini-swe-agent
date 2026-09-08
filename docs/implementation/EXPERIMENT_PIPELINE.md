# Conditional IS SWE-agent experiment pipeline

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

## Reward screen

The reward screen creates one rollout pool for each proposal temperature and
computes sequence-logprob and Consilience values from compact generation-time
statistics. It does not run a second generation for the second reward.

```bash
conditional-is-reward-screen \
  --config configs/swebench/conditional_is_smoke.toml \
  --workload artifacts/workloads/pilot.jsonl \
  --temperature 0.25 --temperature 0.7 --temperature 1.0 \
  --output artifacts/results/reward-screen.json
```

## C/R grid

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

The tuner first chooses a worker count on the baseline, scans the full
`MNS x MBT x memory` grid, and applies 16/32/64-request successive halving.
Each measured arm starts a fresh engine. This is intentional: otherwise later
arms would inherit exact-prefix APC state and gain an unfair advantage.

```bash
conditional-is-runtime-tune \
  --config configs/swebench/conditional_is_smoke.toml \
  --workload artifacts/workloads/swe-agent-128/tune-64.jsonl \
  --warmup-workload artifacts/workloads/warmup.jsonl \
  --devices 0,1 \
  --output-directory artifacts/results/tp2-tuning
```

Use `--dry-run` first to inspect the exact 36 engine configurations. Hard gates
are 100% successful requests, no OOM or reported preemption, and P95 no more
than 1.25 times the round minimum. The final ranking maximizes complete jobs/s.

## Selected-token fallback

`infra/vllm_ascend/selected_token_scoring` contains a hash-guarded vLLM 0.18
patch. It prevents the long-context FP32 full-vocabulary temporary, but remains
off by default because the Python tiled reduction was not bit-exact and did not
deliver production-quality throughput in the earlier 8K experiment.
