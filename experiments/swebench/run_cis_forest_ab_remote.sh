#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 VARIANT DEVICES OUTPUT PORT CONTAINER" >&2
  echo "VARIANT: baseline | step-cap-only | step-priority-only | job-gang | step-cohort-auto | step-cohort-N | step-tree-adaptive | step-tree-fractional | step-tree-tail-borrow | step-tree-peak-budget | step-tree-continuation | step-tree-work-conserving | step-peak-budget | step-runtime-kv-budget | step-peak-budget-lpt | step-peak-budget-balanced | step-gang | step-gang-6 | step-gang-piecewise | step-gang-dual-graph | step-forest-attention | step-forest-attention-piecewise | step-forest-attention-dual-graph | step-forest-window-N | step-fused-paths | step-engine-fork | step-engine-fork-barrier | step-engine-fork-tail-N | step-engine-fork-adaptive | step-engine-fork-compact | step-engine-fork-compact-barrier | step-engine-fork-compact-tail-N | step-elastic | step-window-rollout-first | step-subtree-K | step-subtree-fork-K | step-decode-guard-N | step-occupancy-aware | step-fork | step-fork-lease | step-fork-handoff | step-fork-handoff-bounded | step-streaming-fork-handoff | step-streaming-fork-handoff-bounded | step-resample-gc | step-fork-resample-gc | step-branch-evict | step-fork-branch-evict | step-fork-lease-branch-evict | step-window-rollout-first-fork | step-streaming | step-parent | step-parent-streaming | streaming | bounded | frontier-CAPACITY-BATCH" >&2
  exit 2
fi

variant=$1
devices=$2
output=$3
port=$4
container=$5

image=${CIS_IMAGE:-wangzili/vllm-ascend:v0.18.0-msprof1.2.2-tzdata2025.3-pandas2.2.3}
workspace=${CIS_WORKSPACE:-/data/disk/wangzili/inference-scaling-mini-swe-agent-cis-forest}
model=${CIS_MODEL_DIR:-/data/disk/models/Qwen3-Coder-30B-A3B-Instruct}
public_workload=${CIS_PUBLIC_WORKLOAD_DIR:-/data/disk/wangzili/cis-artifacts-fc11386/workloads/public-256}
self_workload=${CIS_SELF_WORKLOAD_DIR:-/data/disk/wangzili/cis-artifacts-629451f/workloads/self-128}
categorical=${CIS_CATEGORICAL_DIR:-/data/disk/wangzili/vllm-categorical-runtime}
cache=${CIS_VLLM_CACHE:-/data/disk/wangzili/vllm-cache-v018-cis-forest}
limit=${CIS_LIMIT:-64}
workers=${CIS_WORKERS:-32}
run_namespace=${CIS_RUN_NAMESPACE:-cis-forest-ab-20260912}
candidate_count=${CIS_CANDIDATE_COUNT:-15}
rollout_count=${CIS_ROLLOUT_COUNT:-3}
block_size=${CIS_BLOCK_SIZE:-128}
active_step_limit=${CIS_ACTIVE_STEP_LIMIT:-6}
active_step_max_limit=${CIS_ACTIVE_STEP_MAX_LIMIT:-32}
active_step_token_budget=${CIS_ACTIVE_STEP_TOKEN_BUDGET:-}
active_step_kv_capacity_fraction=${CIS_ACTIVE_STEP_KV_CAPACITY_FRACTION:-0.90}
fork_lease_max_fraction=${CIS_FORK_LEASE_MAX_FRACTION:-0.20}
active_step_borrow_limit=${CIS_ACTIVE_STEP_BORROW_LIMIT:-12}
active_step_borrow_below=${CIS_ACTIVE_STEP_BORROW_BELOW:-64}
engine_fork_runnable_fraction=${CIS_ENGINE_FORK_RUNNABLE_FRACTION:-0.5}
tree_target_running_fraction=${CIS_TREE_TARGET_RUNNING_FRACTION:-0.50}
tree_kv_low_watermark=${CIS_TREE_KV_LOW_WATERMARK:-0.80}
tree_kv_high_watermark=${CIS_TREE_KV_HIGH_WATERMARK:-0.92}
tree_adjust_interval=${CIS_TREE_ADJUST_INTERVAL:-1.0}
tree_candidate_idle_grace=${CIS_TREE_CANDIDATE_IDLE_GRACE:-5.0}
tree_initial_window=${CIS_TREE_INITIAL_WINDOW:-$active_step_limit}
tree_max_window=${CIS_TREE_MAX_WINDOW:-}
tree_tail_borrow_limit=${CIS_TREE_TAIL_BORROW_LIMIT:-2}
tree_continuation_grace_ms=${CIS_TREE_CONTINUATION_GRACE_MS:-1500}
tree_peak_token_budget=${CIS_TREE_PEAK_TOKEN_BUDGET:-}
fixed_length=${CIS_FIXED_LENGTH:-0}
step_coalesce_seconds=${CIS_STEP_COALESCE_SECONDS:-1.0}
allow_reserved_devices=${CIS_ALLOW_DOCKER_RESERVED_DEVICES:-0}
native_runtime_setup=:
variant_docker_env=()
fixed_length_arg=

for value in "$limit" "$workers" "$candidate_count" "$rollout_count" "$block_size" "$active_step_limit" "$active_step_max_limit"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "workload and scheduling values must be positive integers" >&2
    exit 2
  }
done
[[ "$tree_initial_window" =~ ^[1-9][0-9]*$ ]] || {
  echo "tree initial window must be a positive integer" >&2
  exit 2
}
if [[ -n "$tree_max_window" && ! "$tree_max_window" =~ ^[1-9][0-9]*$ ]]; then
  echo "tree max window must be empty or a positive integer" >&2
  exit 2
fi
if [[ -n "$tree_peak_token_budget" && ! "$tree_peak_token_budget" =~ ^[1-9][0-9]*$ ]]; then
  echo "tree peak token budget must be empty or a positive integer" >&2
  exit 2
fi
if [[ -n "$active_step_token_budget" && ! "$active_step_token_budget" =~ ^[1-9][0-9]*$ ]]; then
  echo "active step token budget must be empty or a positive integer" >&2
  exit 2
fi
if [[ ! "$active_step_kv_capacity_fraction" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]]; then
  echo "active step KV capacity fraction must be in (0, 1]" >&2
  exit 2
fi
[[ "$tree_continuation_grace_ms" =~ ^[0-9]+$ ]] || {
  echo "tree continuation grace must be a non-negative integer" >&2
  exit 2
}
(( workers <= limit )) || {
  echo "workers must not exceed workload limit" >&2
  exit 2
}
[[ "$run_namespace" =~ ^[A-Za-z0-9._:-]+$ ]] || {
  echo "run namespace contains unsupported characters" >&2
  exit 2
}
[[ "$fixed_length" == 0 || "$fixed_length" == 1 ]] || {
  echo "CIS_FIXED_LENGTH must be 0 or 1" >&2
  exit 2
}
[[ "$allow_reserved_devices" == 0 || "$allow_reserved_devices" == 1 ]] || {
  echo "CIS_ALLOW_DOCKER_RESERVED_DEVICES must be 0 or 1" >&2
  exit 2
}
if [[ "$fixed_length" == 1 ]]; then
  fixed_length_arg='--set generation.ignore_eos=true'
fi

for path in "$workspace" "$model" "$public_workload" "$self_workload" "$categorical"; do
  [[ -e "$path" ]] || { echo "missing dependency: $path" >&2; exit 1; }
done
[[ -x /usr/local/bin/npu-smi ]] || { echo "npu-smi is unavailable" >&2; exit 1; }
if ss -H -ltn "sport = :$port" | grep -q .; then
  echo "port is already in use: $port" >&2
  exit 1
fi

case "$variant" in
  baseline)
    variant_args=()
    ;;
  step-cap-only)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
    )
    ;;
  step-priority-only)
    variant_args=(
      --set 'vllm.request_priority_policy=\"step_fifo\"'
    )
    ;;
  job-gang)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"job_fifo\"'
    )
    ;;
  step-cohort-auto)
    variant_args=(
      --set 'vllm.request_priority_policy=\"step_cohort\"'
      --set 'vllm.request_priority_cohort_size=\"auto\"'
    )
    ;;
  step-cohort-*)
    cohort_size=${variant##*-}
    [[ "$cohort_size" =~ ^[1-9][0-9]*$ ]] || {
      echo "invalid step cohort size: $cohort_size" >&2
      exit 2
    }
    variant_args=(
      --set 'vllm.request_priority_policy=\"step_cohort\"'
      --set vllm.request_priority_cohort_size="$cohort_size"
    )
    ;;
  step-tree-adaptive)
    variant_args=(
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set 'vllm.scheduler_cls=\"inference_scaling.arllm.backends.cis_scheduler.CISTreeScheduler\"'
    )
    variant_docker_env=(
      -e VLLM_CIS_INITIAL_WINDOW="$tree_initial_window"
      -e VLLM_CIS_TARGET_RUNNING_FRACTION="$tree_target_running_fraction"
      -e VLLM_CIS_KV_LOW_WATERMARK="$tree_kv_low_watermark"
      -e VLLM_CIS_KV_HIGH_WATERMARK="$tree_kv_high_watermark"
      -e VLLM_CIS_WINDOW_ADJUST_INTERVAL="$tree_adjust_interval"
      -e VLLM_CIS_CANDIDATE_IDLE_GRACE="$tree_candidate_idle_grace"
      -e VLLM_CIS_SCHEDULER_TRACE=/artifacts/cis-scheduler-trace.jsonl
    )
    ;;
  step-tree-fractional)
    variant_args=(
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set 'vllm.scheduler_cls=\"inference_scaling.arllm.backends.cis_scheduler.CISTreeScheduler\"'
    )
    variant_docker_env=(
      -e VLLM_CIS_ADMISSION_MODE=fractional_work
      -e VLLM_CIS_INITIAL_WINDOW="$tree_initial_window"
      -e VLLM_CIS_CANDIDATE_IDLE_GRACE="$tree_candidate_idle_grace"
      -e VLLM_CIS_SCHEDULER_TRACE=/artifacts/cis-scheduler-trace.jsonl
    )
    ;;
  step-tree-tail-borrow)
    variant_args=(
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set 'vllm.scheduler_cls=\"inference_scaling.arllm.backends.cis_scheduler.CISTreeScheduler\"'
    )
    variant_docker_env=(
      -e VLLM_CIS_ADMISSION_MODE=tail_borrow
      -e VLLM_CIS_INITIAL_WINDOW="$tree_initial_window"
      -e VLLM_CIS_KV_LOW_WATERMARK="$tree_kv_low_watermark"
      -e VLLM_CIS_TAIL_BORROW_LIMIT="$tree_tail_borrow_limit"
      -e VLLM_CIS_CANDIDATE_IDLE_GRACE="$tree_candidate_idle_grace"
      -e VLLM_CIS_SCHEDULER_TRACE=/artifacts/cis-scheduler-trace.jsonl
    )
    ;;
  step-tree-peak-budget)
    variant_args=(
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set 'vllm.scheduler_cls=\"inference_scaling.arllm.backends.cis_scheduler.CISTreeScheduler\"'
    )
    variant_docker_env=(
      -e VLLM_CIS_ADMISSION_MODE=peak_token_budget
      -e VLLM_CIS_INITIAL_WINDOW="$tree_initial_window"
      -e VLLM_CIS_CANDIDATE_IDLE_GRACE="$tree_candidate_idle_grace"
      -e VLLM_CIS_SCHEDULER_TRACE=/artifacts/cis-scheduler-trace.jsonl
    )
    ;;
  step-tree-continuation)
    variant_args=(
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set 'vllm.scheduler_cls=\"inference_scaling.arllm.backends.cis_scheduler.CISTreeScheduler\"'
    )
    variant_docker_env=(
      -e VLLM_CIS_INITIAL_WINDOW="$tree_initial_window"
      -e VLLM_CIS_CONTINUATION_GRACE_MS="$tree_continuation_grace_ms"
      -e VLLM_CIS_CANDIDATE_IDLE_GRACE="$tree_candidate_idle_grace"
      -e VLLM_CIS_SCHEDULER_TRACE=/artifacts/cis-scheduler-trace.jsonl
    )
    ;;
  step-tree-work-conserving)
    variant_args=(
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set 'vllm.scheduler_cls=\"inference_scaling.arllm.backends.cis_scheduler.CISTreeScheduler\"'
    )
    variant_docker_env=(
      -e VLLM_CIS_INITIAL_WINDOW="$tree_initial_window"
      -e VLLM_CIS_CONTINUATION_GRACE_MS="$tree_continuation_grace_ms"
      -e VLLM_CIS_CONTINUATION_WORK_CONSERVING=1
      -e VLLM_CIS_TARGET_RUNNING_FRACTION="$tree_target_running_fraction"
      -e VLLM_CIS_CANDIDATE_IDLE_GRACE="$tree_candidate_idle_grace"
      -e VLLM_CIS_SCHEDULER_TRACE=/artifacts/cis-scheduler-trace.jsonl
    )
    ;;
  step-gang|step-gang-6)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
    )
    ;;
  step-peak-budget)
    variant_args=(
      --set conditional_is.candidate_count="$candidate_count"
      --set conditional_is.rollout_count="$rollout_count"
      --set conditional_is.block_size="$block_size"
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'conditional_is.active_step_admission=\"peak_token_budget\"'
      --set conditional_is.active_step_max_limit="$active_step_max_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
    )
    ;;
  step-runtime-kv-budget)
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --check /workspace/infra/vllm_ascend/cis_scheduler/vllm-0.18-kv-capacity.patch && git apply /workspace/infra/vllm_ascend/cis_scheduler/vllm-0.18-kv-capacity.patch'
    variant_args=(
      --set conditional_is.candidate_count="$candidate_count"
      --set conditional_is.rollout_count="$rollout_count"
      --set conditional_is.block_size="$block_size"
      --set 'conditional_is.active_step_admission=\"runtime_kv_budget\"'
      --set conditional_is.active_step_max_limit="$active_step_max_limit"
      --set conditional_is.active_step_kv_capacity_fraction="$active_step_kv_capacity_fraction"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
    )
    if [[ -n "$active_step_token_budget" ]]; then
      variant_args+=(
        --set conditional_is.active_step_token_budget="$active_step_token_budget"
      )
    fi
    ;;
  step-peak-budget-lpt)
    variant_args=(
      --set conditional_is.candidate_count="$candidate_count"
      --set conditional_is.rollout_count="$rollout_count"
      --set conditional_is.block_size="$block_size"
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'conditional_is.active_step_admission=\"peak_token_budget\"'
      --set conditional_is.active_step_max_limit="$active_step_max_limit"
      --set 'conditional_is.active_step_queue_policy=\"largest_fit\"'
      --set conditional_is.active_step_coalesce_seconds="$step_coalesce_seconds"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
    )
    ;;
  step-peak-budget-balanced)
    variant_args=(
      --set conditional_is.candidate_count="$candidate_count"
      --set conditional_is.rollout_count="$rollout_count"
      --set conditional_is.block_size="$block_size"
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'conditional_is.active_step_admission=\"peak_token_budget\"'
      --set conditional_is.active_step_max_limit="$active_step_max_limit"
      --set 'conditional_is.active_step_queue_policy=\"balanced_fit\"'
      --set conditional_is.active_step_coalesce_seconds="$step_coalesce_seconds"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
    )
    ;;
  step-gang-piecewise)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set 'vllm.engine_kwargs.compilation_config={\"cudagraph_mode\":\"PIECEWISE\"}'
    )
    ;;
  step-gang-dual-graph)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set 'vllm.engine_kwargs.compilation_config={\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}'
    )
    native_runtime_setup='cd /vllm-workspace/vllm-ascend && git apply --recount --check /workspace/infra/vllm_ascend/cis_forest_attention/vllm-ascend-0.18-packed-forest-attention.patch && git apply --recount /workspace/infra/vllm_ascend/cis_forest_attention/vllm-ascend-0.18-packed-forest-attention.patch'
    ;;
  step-forest-window-*)
    forest_window_steps=${variant##*-}
    [[ "$forest_window_steps" =~ ^[1-9][0-9]*$ ]] || {
      echo "invalid Forest window length: $forest_window_steps" >&2
      exit 2
    }
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_packed_forest_attention=true
      --set 'vllm.engine_kwargs.compilation_config={\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}'
    )
    variant_docker_env=(
      -e VLLM_CIS_FOREST_WINDOW_MAX_STEPS="$forest_window_steps"
      -e VLLM_CIS_FOREST_WINDOW_MIN_SAVED_READS=196608
      -e VLLM_CIS_FOREST_WINDOW_MIN_PACKED_FRACTION=0.75
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --recount --check /workspace/infra/vllm_ascend/cis_forest_attention/vllm-0.18-cis-forest-window.patch && git apply --recount /workspace/infra/vllm_ascend/cis_forest_attention/vllm-0.18-cis-forest-window.patch && cd /vllm-workspace/vllm-ascend && git apply --recount --check /workspace/infra/vllm_ascend/cis_forest_attention/vllm-ascend-0.18-packed-forest-attention.patch && git apply --recount /workspace/infra/vllm_ascend/cis_forest_attention/vllm-ascend-0.18-packed-forest-attention.patch'
    ;;
  step-forest-attention|step-forest-attention-piecewise|step-forest-attention-dual-graph)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_packed_forest_attention=true
    )
    if [[ "$variant" == step-forest-attention-piecewise ]]; then
      variant_args+=(
        --set 'vllm.engine_kwargs.compilation_config={\"cudagraph_mode\":\"PIECEWISE\"}'
      )
    elif [[ "$variant" == step-forest-attention-dual-graph ]]; then
      variant_args+=(
        --set 'vllm.engine_kwargs.compilation_config={\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}'
      )
    fi
    native_runtime_setup='cd /vllm-workspace/vllm-ascend && git apply --recount --check /workspace/infra/vllm_ascend/cis_forest_attention/vllm-ascend-0.18-packed-forest-attention.patch && git apply --recount /workspace/infra/vllm_ascend/cis_forest_attention/vllm-ascend-0.18-packed-forest-attention.patch'
    ;;
  step-fused-paths)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set conditional_is.fused_candidate_rollout_paths=true
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_segmented_rng=true
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --check /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-cis-segmented-rng-prefix-sync.patch && git apply /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-cis-segmented-rng-prefix-sync.patch && cd /vllm-workspace/vllm-ascend && git apply --check /workspace/infra/vllm_ascend/cis_native_tree/vllm-ascend-0.18-cis-segmented-rng-prefix-sync.patch && git apply /workspace/infra/vllm_ascend/cis_native_tree/vllm-ascend-0.18-cis-segmented-rng-prefix-sync.patch'
    ;;
  step-engine-fork|step-engine-fork-barrier|step-engine-fork-tail-*|step-engine-fork-adaptive|step-engine-fork-compact|step-engine-fork-compact-barrier|step-engine-fork-compact-tail-*)
    engine_fork_variant=$variant
    engine_fork_compact=false
    if [[ "$variant" == step-engine-fork-compact* ]]; then
      engine_fork_compact=true
      engine_fork_variant="step-engine-fork${variant#step-engine-fork-compact}"
    fi
    engine_fork_release_args=()
    if [[ "$engine_fork_variant" == step-engine-fork-barrier ]]; then
      engine_fork_release_args+=(
        --set conditional_is.engine_fork_release_remaining_candidates=0
      )
    elif [[ "$engine_fork_variant" == step-engine-fork-tail-* ]]; then
      engine_fork_tail=${engine_fork_variant##*-}
      [[ "$engine_fork_tail" =~ ^[0-9]+$ ]] || {
        echo "invalid engine fork tail: $engine_fork_tail" >&2
        exit 2
      }
      (( engine_fork_tail < candidate_count )) || {
        echo "engine fork tail must be smaller than candidate count" >&2
        exit 2
      }
      engine_fork_release_args+=(
        --set conditional_is.engine_fork_release_remaining_candidates="$engine_fork_tail"
      )
    elif [[ "$engine_fork_variant" == step-engine-fork-adaptive ]]; then
      engine_fork_release_args+=(
        --set conditional_is.engine_fork_release_remaining_candidates=0
        --set conditional_is.engine_fork_adaptive_release=true
        --set conditional_is.engine_fork_adaptive_runnable_fraction="$engine_fork_runnable_fraction"
      )
    fi
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set conditional_is.engine_fork_candidate_rollouts=true
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_kv_fork=true
      --set vllm.native_kv_fork_waiters=true
      --set vllm.native_kv_fork_lease=true
      --set 'vllm.native_kv_fork_lease_scope=\"full_parent\"'
      "${engine_fork_release_args[@]}"
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --recount /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch && git apply --recount --check /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-cis-fork-waiters.patch && git apply --recount /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-cis-fork-waiters.patch'
    if [[ "$engine_fork_compact" == true ]]; then
      variant_args+=(--set vllm.native_kv_fork_compact_waiters=true)
      native_runtime_setup+=' && git apply --recount --check /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-cis-fork-compact-waiters.patch && git apply --recount /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-cis-fork-compact-waiters.patch'
    fi
    if [[ "$engine_fork_variant" == step-engine-fork-adaptive ]]; then
      native_runtime_setup+=' && git apply --recount --check /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-cis-fork-adaptive-release.patch && git apply --recount /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-cis-fork-adaptive-release.patch'
    fi
    ;;
  step-elastic)
    variant_args=(
      --set 'vllm.request_priority_policy=\"rollout_first\"'
    )
    ;;
  step-window-rollout-first)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"rollout_first\"'
    )
    ;;
  step-subtree-*|step-subtree-fork-*)
    subtree_max=${variant##*-}
    [[ "$subtree_max" =~ ^[1-9][0-9]*$ ]] || {
      echo "invalid subtree max active batches: $subtree_max" >&2
      exit 2
    }
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set conditional_is.rollout_subtree_max_active_batches="$subtree_max"
    )
    if [[ "$variant" == step-subtree-fork-* ]]; then
      variant_args+=(--set vllm.native_kv_fork=true)
      native_runtime_setup='cd /vllm-workspace/vllm && git apply --recount /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch'
    fi
    ;;
  step-decode-guard-*)
    guard_prefills=${variant##*-}
    [[ "$guard_prefills" =~ ^[1-9][0-9]*$ ]] || {
      echo "invalid decode guard prefill limit: $guard_prefills" >&2
      exit 2
    }
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
    )
    variant_docker_env=(
      -e VLLM_CIS_DECODE_GUARD_MAX_PREFILLS="$guard_prefills"
      -e VLLM_CIS_DECODE_GUARD_MIN_DECODES=32
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --check /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-cis-decode-guard.patch && git apply /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-cis-decode-guard.patch'
    ;;
  step-occupancy-aware)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set conditional_is.active_step_borrow_limit="$active_step_borrow_limit"
      --set conditional_is.active_step_borrow_below_requests="$active_step_borrow_below"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
    )
    ;;
  step-fork)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_kv_fork=true
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --recount /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch'
    ;;
  step-fork-lease)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_kv_fork=true
      --set vllm.native_kv_fork_lease=true
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --recount /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch'
    ;;
  step-fork-handoff)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_kv_fork=true
      --set vllm.native_kv_fork_lease=true
      --set 'vllm.native_kv_fork_lease_scope=\"full_parent\"'
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --check /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch && git apply /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch'
    ;;
  step-fork-handoff-bounded)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_kv_fork=true
      --set vllm.native_kv_fork_lease=true
      --set 'vllm.native_kv_fork_lease_scope=\"full_parent\"'
      --set vllm.native_kv_fork_lease_max_fraction="$fork_lease_max_fraction"
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --check /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch && git apply /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch'
    ;;
  step-streaming-fork-handoff|step-streaming-fork-handoff-bounded)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set conditional_is.stream_candidate_rollouts=true
      --set conditional_is.rollout_stream_candidate_batch_size=1
      --set conditional_is.rollout_stream_max_batches="$candidate_count"
      --set vllm.native_kv_fork=true
      --set vllm.native_kv_fork_lease=true
      --set 'vllm.native_kv_fork_lease_scope=\"full_parent\"'
    )
    if [[ "$variant" == step-streaming-fork-handoff-bounded ]]; then
      variant_args+=(
        --set vllm.native_kv_fork_lease_max_fraction="$fork_lease_max_fraction"
      )
    fi
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --check /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch && git apply /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch'
    ;;
  step-resample-gc|step-fork-resample-gc)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_kv_resample_gc=true
    )
    if [[ "$variant" == step-fork-resample-gc ]]; then
      variant_args+=(--set vllm.native_kv_fork=true)
    fi
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --check /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch && git apply /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch'
    ;;
  step-branch-evict)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_kv_branch_eviction=true
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --recount /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch'
    ;;
  step-fork-branch-evict)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_kv_fork=true
      --set vllm.native_kv_branch_eviction=true
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --recount /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch'
    ;;
  step-fork-lease-branch-evict)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_kv_fork=true
      --set vllm.native_kv_fork_lease=true
      --set vllm.native_kv_branch_eviction=true
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --recount /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch'
    ;;
  step-window-rollout-first-fork)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"rollout_first\"'
      --set vllm.native_kv_fork=true
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --recount /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-kv-fork.patch'
    ;;
  step-streaming)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set conditional_is.stream_candidate_rollouts=true
      --set conditional_is.rollout_stream_candidate_batch_size=1
      --set conditional_is.rollout_stream_max_batches="$candidate_count"
    )
    ;;
  step-parent)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_parallel_sampling=true
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --recount /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-parent-request.patch'
    ;;
  step-parent-streaming)
    variant_args=(
      --set conditional_is.active_step_limit="$active_step_limit"
      --set 'vllm.request_priority_policy=\"step_fifo\"'
      --set vllm.native_parallel_sampling=true
      --set conditional_is.stream_candidate_rollouts=true
      --set conditional_is.rollout_stream_candidate_batch_size=1
      --set conditional_is.rollout_stream_max_batches="$candidate_count"
    )
    native_runtime_setup='cd /vllm-workspace/vllm && git apply --recount /workspace/infra/vllm_ascend/cis_native_tree/vllm-0.18-parent-request.patch'
    ;;
  streaming)
    variant_args=(
      --set conditional_is.stream_candidate_rollouts=true
      --set conditional_is.rollout_stream_candidate_batch_size=5
      --set conditional_is.rollout_stream_max_batches=2
    )
    ;;
  bounded)
    # Requests are candidate-major, so 15 requests admit five C15/R3 subtrees.
    variant_args=(--set conditional_is.rollout_submission_batch_size=15)
    ;;
  frontier-*)
    frontier=${variant#frontier-}
    capacity=${frontier%-*}
    batch_size=${frontier##*-}
    [[ "$capacity" =~ ^[1-9][0-9]*$ ]] || {
      echo "invalid frontier capacity: $capacity" >&2
      exit 2
    }
    [[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || {
      echo "invalid frontier batch size: $batch_size" >&2
      exit 2
    }
    variant_args=(
      --set conditional_is.rollout_frontier_capacity="$capacity"
      --set conditional_is.rollout_frontier_batch_size="$batch_size"
    )
    ;;
  *)
    echo "unknown variant: $variant" >&2
    exit 2
    ;;
esac

if [[ -n "$tree_max_window" && "$variant" == step-tree-* ]]; then
  variant_docker_env+=(
    -e VLLM_CIS_MAX_WINDOW="$tree_max_window"
  )
fi
if [[ -n "$tree_peak_token_budget" && "$variant" == step-tree-peak-budget ]]; then
  variant_docker_env+=(
    -e VLLM_CIS_PEAK_TOKEN_BUDGET="$tree_peak_token_budget"
  )
fi

IFS=, read -r -a device_ids <<<"$devices"
device_args=()
logical_ids=()
for logical in "${!device_ids[@]}"; do
  id=${device_ids[$logical]}
  [[ "$id" =~ ^[0-7]$ ]] || { echo "invalid device: $id" >&2; exit 2; }
  device_args+=(--device "/dev/davinci$id:/dev/davinci$logical")
  logical_ids+=("$logical")
done
logical_devices=$(IFS=,; echo "${logical_ids[*]}")

if [[ -e "$output" ]] && ! rm -rf "$output" 2>/dev/null; then
  output_parent=$(dirname "$output")
  output_name=$(basename "$output")
  docker run --rm \
    --entrypoint /bin/bash \
    -v "$output_parent":/cleanup \
    "$image" -lc 'rm -rf -- "/cleanup/$1"' _ "$output_name"
fi
mkdir -p "$output" "$cache"
docker rm -f "$container" >/dev/null 2>&1 || true
reserved_devices=()
for running_container in $(docker ps -q); do
  running_name=$(docker inspect -f '{{.Name}}' "$running_container")
  while IFS= read -r host_device; do
    for id in "${device_ids[@]}"; do
      if [[ "$host_device" == "/dev/davinci$id" ]]; then
        reserved_devices+=("$id:${running_name#/}")
      fi
    done
  done < <(
    docker inspect -f '{{range .HostConfig.Devices}}{{println .PathOnHost}}{{end}}' \
      "$running_container"
  )
done
if (( ${#reserved_devices[@]} > 0 )); then
  reserved_list=$(IFS=,; echo "${reserved_devices[*]}")
  if [[ "$allow_reserved_devices" != 1 ]]; then
    echo "selected NPU is reserved by a running container: $reserved_list" >&2
    echo "set CIS_ALLOW_DOCKER_RESERVED_DEVICES=1 only after owner confirmation" >&2
    exit 1
  fi
  echo "warning: overriding running-container reservations: $reserved_list" >&2
fi
npu_snapshot=$(/usr/local/bin/npu-smi info)
printf '%s\n' "$npu_snapshot" >"$output/npu-before.txt"
process_snapshot=$(printf '%s\n' "$npu_snapshot" | sed -n '/Process id/,$p')
for id in "${device_ids[@]}"; do
  if printf '%s\n' "$process_snapshot" \
    | grep -Eq "^\\|[[:space:]]*$id[[:space:]]+\\|"; then
    echo "selected NPU $id already has a process" >&2
    exit 1
  fi
done

printf '%q ' "$0" "$@" >"$output/launch-command.txt"
printf '\n' >>"$output/launch-command.txt"
cat >"$output/run-environment.txt" <<EOF
CIS_IMAGE=$image
CIS_WORKSPACE=$workspace
CIS_MODEL_DIR=$model
CIS_PUBLIC_WORKLOAD_DIR=$public_workload
CIS_SELF_WORKLOAD_DIR=$self_workload
CIS_CATEGORICAL_DIR=$categorical
CIS_VLLM_CACHE=$cache
CIS_LIMIT=$limit
CIS_WORKERS=$workers
CIS_RUN_NAMESPACE=$run_namespace
CIS_CANDIDATE_COUNT=$candidate_count
CIS_ROLLOUT_COUNT=$rollout_count
CIS_BLOCK_SIZE=$block_size
CIS_ACTIVE_STEP_LIMIT=$active_step_limit
CIS_ACTIVE_STEP_MAX_LIMIT=$active_step_max_limit
CIS_ACTIVE_STEP_TOKEN_BUDGET=$active_step_token_budget
CIS_ACTIVE_STEP_KV_CAPACITY_FRACTION=$active_step_kv_capacity_fraction
CIS_ACTIVE_STEP_BORROW_LIMIT=$active_step_borrow_limit
CIS_ACTIVE_STEP_BORROW_BELOW=$active_step_borrow_below
CIS_ENGINE_FORK_RUNNABLE_FRACTION=$engine_fork_runnable_fraction
CIS_FORK_LEASE_MAX_FRACTION=$fork_lease_max_fraction
CIS_TREE_TARGET_RUNNING_FRACTION=$tree_target_running_fraction
CIS_TREE_KV_LOW_WATERMARK=$tree_kv_low_watermark
CIS_TREE_KV_HIGH_WATERMARK=$tree_kv_high_watermark
CIS_TREE_ADJUST_INTERVAL=$tree_adjust_interval
CIS_TREE_CANDIDATE_IDLE_GRACE=$tree_candidate_idle_grace
CIS_TREE_MAX_WINDOW=$tree_max_window
CIS_TREE_TAIL_BORROW_LIMIT=$tree_tail_borrow_limit
CIS_TREE_CONTINUATION_GRACE_MS=$tree_continuation_grace_ms
CIS_TREE_PEAK_TOKEN_BUDGET=$tree_peak_token_budget
CIS_FIXED_LENGTH=$fixed_length
CIS_STEP_COALESCE_SECONDS=$step_coalesce_seconds
CIS_ALLOW_DOCKER_RESERVED_DEVICES=$allow_reserved_devices
EOF

docker run -d \
  --name "$container" \
  --network host \
  --entrypoint /bin/bash \
  -e CIS_MODEL_PATH=/models/conditional-is \
  -e CIS_HOST_NPU_PREFLIGHT_VERIFIED=1 \
  -e ASCEND_RT_VISIBLE_DEVICES="$logical_devices" \
  "${variant_docker_env[@]}" \
  "${device_args[@]}" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$model":/models/conditional-is:ro \
  -v "$workspace":/workspace:ro \
  -v "$public_workload":/workloads/public:ro \
  -v "$self_workload":/workloads/self:ro \
  -v "$categorical":/categorical:ro \
  -v "$categorical"/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so:/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so:ro \
  -v "$categorical"/runtime/sampler.py:/vllm-workspace/vllm-ascend/vllm_ascend/sample/sampler.py:ro \
  -v "$categorical"/vllm_ascend/libvllm_ascend_kernels.so:/vllm-workspace/vllm-ascend/vllm_ascend/libvllm_ascend_kernels.so:ro \
  -v "$categorical"/vllm_ascend/_cann_ops_custom:/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom:ro \
  -v "$cache":/root/.cache/vllm \
  -v "$output":/artifacts \
  "$image" -lc "
    set -euo pipefail
    $native_runtime_setup
    export PYTHONPATH=/workspace/src:\$PYTHONPATH
    cd /workspace
    /usr/local/python3.11.14/bin/python -c 'import inference_scaling.swe_agent.profile as p; print(p.__file__)'
    exec /usr/local/python3.11.14/bin/python -m inference_scaling.swe_agent.profile \
      --config /workspace/configs/swebench/conditional_is_smoke.toml \
      --workload /workloads/public/public-256.jsonl \
      --warmup-workload /workloads/self/warmup-4.jsonl \
      --tensor-parallel-size ${#device_ids[@]} \
      --pipeline-parallel-size 1 \
      --profiler none \
      --limit $limit \
      --run-namespace $run_namespace \
      --categorical-root /categorical \
      --set generation.max_new_tokens=512 \
      $fixed_length_arg \
      --set vllm.max_model_len=65536 \
      --set vllm.max_num_seqs=256 \
      --set vllm.max_num_batched_tokens=32768 \
      --set vllm.gpu_memory_utilization=0.90 \
      --set vllm.engine_kwargs.max_num_partial_prefills=1 \
      --set vllm.engine_kwargs.max_long_partial_prefills=1 \
      ${variant_args[*]} \
      --env VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE=1 \
      --env ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend \
      --env LD_PRELOAD=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so \
      --output-directory /artifacts \
      --devices $logical_devices \
      --port $port \
      --routing round_robin \
      --workers $workers \
      --candidate-count $candidate_count \
      --rollout-count $rollout_count \
      --block-size $block_size \
      > /artifacts/launcher.log 2>&1
  "

echo "$container"
