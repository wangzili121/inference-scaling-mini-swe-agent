# Conditional IS x mini-SWE-agent implementation

## Execution contract

One `ConditionalISModel.query(messages)` call is one complete ordinary
Conditional IS job. Candidate and rollout completions remain internal to the
service. Only the selected completion is parsed, returned to mini-SWE-agent,
and executed in the SWE-bench Docker container. The resulting tool observation
is appended to the next model-call context by mini-SWE-agent.

The service owns one persistent inference backend. It records each real agent
call as JSONL with messages, prompt token IDs, selected output, latency,
candidate ESS, and backend counter deltas. These records are the source for the
later `tune-64` and `holdout-64` manifests.

## Pinned baseline

- Algorithm source: `04492fe1e227e7171f9f20c971240c2d5ac3c535`.
- Agent: mini-SWE-agent `v2.4.6`.
- Runtime target: vLLM-Ascend `v0.18`, MRV1.
- Model: `Qwen3-Coder-30B-A3B-Instruct`.
- Smoke only: `C4/R2/B128/L512`, temperature 1, no top-k/top-p truncation,
  sequence-logprob reward.

The smoke configuration is not a tuned performance claim. In particular,
`max_num_seqs=128`, `max_num_batched_tokens=32768`, and memory utilization 0.90
are merely runnable starting values.

## Implemented foundation

1. A mini-SWE-agent custom model client with deterministic request-local seeds,
   retries, tool-call validation, and trajectory serialization.
2. A persistent concurrent HTTP service around one ordinary Conditional IS
   backend.
3. Qwen tool-call parsing into mini-SWE-agent's OpenAI-style bash action format.
4. Compact generation-time selected-token logprob and top-K confidence
   trajectories.
5. Sample-aware sequence-logprob and Consilience rewards. They reuse the
   generation forward only when model and policy identities match; otherwise
   they fall back to exact scoring.
6. Fixed-workload reward, C/R, burst, and two-card successive-halving tools,
   plus stage-level trace annotations and TP/PP engine configuration support.

The exact command sequence is documented in
[`EXPERIMENT_PIPELINE.md`](EXPERIMENT_PIPELINE.md).

## Remaining experiment gates

1. Validate one task and then three consecutive tasks, including Docker,
   context accumulation, EOS, timeout, recovery, and saved trajectories.
2. Apply and validate the v0.18 MRV1 categorical sampler and Ascend graph
   settings on the remote NPU host.
3. Run the reward/temperature screen, then the six C/R combinations.
4. Collect 128 real calls and freeze the two 64-request manifests.
5. Tune TP2 and compare the prescribed four-card topologies.
6. Profile the best two-card and four-card configurations before choosing any
   algorithm-aware scheduling, KV-affinity, communication, or kernel work.
