# CIS native parent grouping prototype

This experiment reuses vLLM V1 `ParentRequest` for the first, deliberately
limited layer of a Conditional IS tree runtime:

- all candidate siblings in one step enter through one parent call;
- the R rollout siblings of one candidate enter through one parent call;
- every child keeps the exact seed assigned by Conditional IS;
- final child outputs are exposed as each child finishes, so candidate callbacks
  can still release rollout work before the whole parent completes.

It does not yet make the two-level candidate/rollout forest one EngineCore
request. vLLM still materializes one `EngineCoreRequest` per child, and it does
not retain candidate KV as a forkable continuation. The prototype is intended
to measure frontend lifecycle savings and establish the output/seed plumbing
needed by a later dynamic tree request.

`vllm-0.18-kv-fork.patch` contains three separately gated experiments:

- direct fork captures the candidate's full computed physical blocks before
  teardown and lets hash-matching rollout children adopt them directly;
- suffix lease temporarily increments references only for candidate-specific
  full blocks, then releases them after all children are admitted or after a
  bounded timeout;
- branch eviction marks pure rollout-generated full blocks as disposable and
  moves only unreferenced blocks to the front of the free-cache eviction queue.

Direct fork safely falls back to APC when any parent block is stale. It proves
scheduler-visible parent/child block-table handoff, but children are still new
EngineCore requests. It does not replace the eventual branch-on-token design,
which must increment child references before the parent block table is freed.

The patches are applied only inside experiment containers. Enable individual
features with `vllm.native_parallel_sampling`, `vllm.native_kv_fork`,
`vllm.native_kv_fork_lease`, and `vllm.native_kv_branch_eviction`; the backend
fails fast when the corresponding runtime marker is absent.
