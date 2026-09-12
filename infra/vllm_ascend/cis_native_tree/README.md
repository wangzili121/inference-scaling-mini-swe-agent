# CIS native tree-runtime prototypes

This experiment reuses vLLM V1 `ParentRequest` for the first, deliberately
limited layer of a Conditional IS tree runtime:

- all candidate siblings in one step enter through one parent call;
- the R rollout siblings of one candidate enter through one parent call;
- every child keeps the exact seed assigned by Conditional IS;
- final child outputs are exposed as each child finishes, so candidate callbacks
  can still release rollout work before the whole parent completes.

It does not make the two-level candidate/rollout forest one EngineCore request.
vLLM still materializes one `EngineCoreRequest` per child. The parent-request
prototype is intended to measure frontend lifecycle savings and establish the
output/seed plumbing needed by a later dynamic tree request.

`vllm-0.18-cis-fork-waiters.patch` is the deeper fork-on-token prototype. The
frontend pre-registers rollout children, but the scheduler parks them outside
the model until their candidate reaches B tokens. EngineCore then expands each
child with the candidate tokens, transfers the parent's physical KV blocks,
and admits the child without a second frontend submission. A completion
tombstone makes parent-first and child-first AddRequest order equivalent;
terminal parents cancel late children without model work. Children remain
ordinary branch scheduling units, so branches from multiple CIS trees can
still form a continuous batch.

`vllm-0.18-cis-fork-compact-waiters.patch` removes the largest remaining
placeholder payload without changing the output protocol. The frontend keeps
the real prompt for output construction, while EngineCore receives a one-token
sentinel for each dormant child. On release, the scheduler replaces that
sentinel with the completed parent's full token state, rebuilds block hashes,
and adopts the parent KV. The sentinel never enters a model forward. This
still creates one child future/request identity, but avoids serializing and
materializing `C x R` copies of a long prompt inside EngineCore.

The full P0 run elided 43,002,180 prompt tokens across 3,150 parked children,
with 100% physical fork hits and no preemption. It nevertheless reduced
jobs/s by 3.6% and forward-token-slots/s by 5.9% versus the same-host Step
control, while worsening Job P95 by 37.0%. This is a useful negative result:
placeholder prompt transport is not the cause of the fork prototype's loss,
so replacing the remaining child identities with a dynamic tree-root protocol
is not currently justified. The patches remain reproducible experiments, not
recommended defaults.

Waiter release has three explicit policies:

- immediate release models unconstrained fork-on-token and is retained as a
  negative control;
- group barrier preserves the original candidate-phase boundary while removing
  the frontend round trip;
- tail-N releases completed parents when N candidates remain.

`vllm-0.18-cis-fork-adaptive-release.patch` adds an experimental global
runnable-occupancy refill policy. It is intentionally separate because the
first 0.5-MNS heuristic rarely fired in the short P2 run and is not a validated
default policy.

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
children have acquired their references. The direct-handoff path still uses a
second frontend submission. Fork waiters remove that post-candidate round trip,
but retain pre-registered child request lifecycle overhead. A final dynamic
tree-root protocol would create children inside EngineCore and report them
through their logical parent.

Full-parent retention is guarded in two ways. A candidate that ended on an EOS
stop token is recognized inside EngineCore and is not captured because the CIS
algorithm will not create rollout children. An optional
`vllm.native_kv_fork_lease_max_fraction` caps the number of unique physical KV
blocks held by fork leases; parents that exceed the credit fall back to a
candidate-suffix lease or ordinary direct-fork/APC lookup. This prevents fork
handoff from consuming all evictable blocks and stalling admission.

The patches are applied only inside experiment containers. Enable individual
features with `vllm.native_parallel_sampling`, `vllm.native_kv_fork`,
`vllm.native_kv_fork_waiters`, `vllm.native_kv_fork_compact_waiters`,
`vllm.native_kv_fork_lease`, `vllm.native_kv_fork_lease_scope`, and
`vllm.native_kv_branch_eviction`; the backend fails fast when the
corresponding runtime marker is absent.

Enable reduce-transition collection with `vllm.native_kv_resample_gc`. Unlike
immediate rollout-tail demotion, this waits for the algorithm's resample point
and expresses the exact `winner/live` versus `loser/dead` lifecycle. It remains
a soft cache reclamation experiment: finished requests have already released
their active references, and the hook removes only validated prefix-cache
entries. It does not alter candidate selection or generated tokens.
