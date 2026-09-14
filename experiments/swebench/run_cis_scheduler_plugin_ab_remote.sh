#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 DEVICES OUTPUT_ROOT PORT CONTAINER_PREFIX PHASE" >&2
  echo "PHASE: smoke | full | all" >&2
  exit 2
fi

devices=$1
output_root=$2
port=$3
container_prefix=$4
phase=$5
workspace=$(cd "$(dirname "$0")/../.." && pwd)
runner=$workspace/experiments/swebench/run_cis_forest_ab_remote.sh

run_matrix() {
  local size=$1
  local workers
  if [[ "$size" == smoke ]]; then
    workers=16
  else
    workers=32
  fi
  local specs=(
    "p0-static|step-gang|15|3|0|64"
    "p0-outer-runtime|step-runtime-kv-budget|15|3|0|64"
    "p0-engine-runtime|step-tree-runtime-kv|15|3|0|64"
    "p1-static|step-gang|8|3|1|32"
    "p1-outer-runtime|step-runtime-kv-budget|8|3|1|32"
    "p1-engine-runtime|step-tree-runtime-kv|8|3|1|32"
  )

  for spec in "${specs[@]}"; do
    IFS='|' read -r label variant candidates rollouts fixed_length full_limit \
      <<<"$spec"
    local limit=$full_limit
    [[ "$size" == smoke ]] && limit=16
    local output=$output_root/$size/$label
    local container=$container_prefix-$size-$label
    if [[ -s "$output/benchmark.json" ]]; then
      echo "[$(date -Is)] skipping completed $size/$label"
      continue
    fi
    CIS_WORKSPACE=$workspace \
    CIS_RUN_NAMESPACE="scheduler-plugin-ab-20260914:$size:$label" \
    CIS_LIMIT=$limit \
    CIS_WORKERS=$workers \
    CIS_CANDIDATE_COUNT=$candidates \
    CIS_ROLLOUT_COUNT=$rollouts \
    CIS_ACTIVE_STEP_LIMIT=16 \
    CIS_ACTIVE_STEP_MAX_LIMIT=64 \
    CIS_ACTIVE_STEP_KV_CAPACITY_FRACTION=0.90 \
    CIS_FIXED_LENGTH=$fixed_length \
      "$runner" "$variant" "$devices" "$output" "$port" "$container"
    local status
    status=$(docker wait "$container")
    docker logs "$container" >"$output/container.log" 2>&1 || true
    docker rm "$container" >/dev/null
    if [[ "$status" != 0 ]]; then
      echo "run failed: $size/$label (container exit $status)" >&2
      exit "$status"
    fi
  done

  local roots=()
  local spec label
  for spec in "${specs[@]}"; do
    IFS='|' read -r label _ <<<"$spec"
    [[ -s "$output_root/$size/$label/benchmark.json" ]] \
      && roots+=("$output_root/$size/$label")
  done
  if (( ${#roots[@]} > 0 )); then
    PYTHONPATH=$workspace/src python3 \
      "$workspace/experiments/swebench/summarize_cis_step_ab.py" \
      "${roots[@]}" --output "$output_root/$size/summary.json"
  fi
}

case "$phase" in
  smoke)
    run_matrix smoke
    ;;
  full)
    run_matrix full
    ;;
  all)
    run_matrix smoke
    run_matrix full
    ;;
  *)
    echo "unknown phase: $phase" >&2
    exit 2
    ;;
esac
