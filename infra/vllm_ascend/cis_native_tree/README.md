# CIS native tree-runtime prototypes

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

`vllm-0.18-kv-fork.patch` contains separately gated experiments:

- direct fork captures the candidate's full computed physical blocks before
  teardown and lets hash-matching rollout children adopt them directly;
- suffix lease temporarily increments references only for candidate-specific
  full blocks, then releases them after all children are admitted or after a
  bounded timeout;
- full-parent handoff holds every reusable parent block across the EngineCore
  completion-to-admission gap, so each rollout adopts the exact parent block
  table before the final hold is released;
- branch eviction marks pure rollout-generated full blocks as disposable and
  moves only unreferenced blocks to the front of the free-cache eviction queue.
- resample GC records candidate/rollout-owned suffix blocks at completion. Once
  the exact winner is known, one EngineCore utility call evicts loser-candidate
  suffixes and every rollout tail while protecting blocks shared with the
  winner. Block IDs are paired with their captured hashes, so IDs recycled
  before the reduce transition are counted as stale instead of being evicted.

Direct fork safely falls back to APC when any parent block is stale. Setting
`vllm.native_kv_fork_lease_scope=full_parent` makes the handoff exact: the
parent reference is acquired before teardown and released after all expected
children have acquired their references. Children are still separate
EngineCore requests, so this does not yet remove host round-trips or request
lifecycle overhead as an eventual branch-on-token tree request would.

Full-parent retention is guarded in two ways. A candidate that ended on an EOS
stop token is recognized inside EngineCore and is not captured because the CIS
algorithm will not create rollout children. An optional
`vllm.native_kv_fork_lease_max_fraction` caps the number of unique physical KV
blocks held by fork leases; parents that exceed the credit fall back to a
candidate-suffix lease or ordinary direct-fork/APC lookup. This prevents fork
handoff from consuming all evictable blocks and stalling admission.

The patches are applied only inside experiment containers. Enable individual
features with `vllm.native_parallel_sampling`, `vllm.native_kv_fork`,
`vllm.native_kv_fork_lease`, `vllm.native_kv_fork_lease_scope`, and
`vllm.native_kv_branch_eviction`; the backend fails fast when the corresponding
runtime marker is absent.

Enable reduce-transition collection with `vllm.native_kv_resample_gc`. Unlike
immediate rollout-tail demotion, this waits for the algorithm's resample point
and expresses the exact `winner/live` versus `loser/dead` lifecycle. It remains
a soft cache reclamation experiment: finished requests have already released
their active references, and the hook removes only validated prefix-cache
entries. It does not alter candidate selection or generated tokens.
