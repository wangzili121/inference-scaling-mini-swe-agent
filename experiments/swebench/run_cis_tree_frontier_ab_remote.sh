#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 DEVICES OUTPUT_ROOT PORT CONTAINER_PREFIX smoke|short-p1|full-p1|full-p0|full-p2" >&2
  exit 2
fi

devices=$1
output_root=$2
port=$3
container_prefix=$4
phase=$5
workspace=$(cd "$(dirname "$0")/../.." && pwd)
runner=$workspace/experiments/swebench/run_cis_forest_ab_remote.sh

case "$phase" in
  smoke)
    limit=4 workers=4 candidate_count=8 rollout_count=3 fixed_length=1
    specs=("tree-frontier|tree")
    ;;
  short-p1)
    limit=16 workers=16 candidate_count=8 rollout_count=3 fixed_length=1
    specs=("job-runtime-kv-budget|control" "tree-frontier|tree")
    ;;
  full-p1)
    limit=32 workers=32 candidate_count=8 rollout_count=3 fixed_length=1
    specs=("job-runtime-kv-budget|control" "tree-frontier|tree")
    ;;
  full-p0)
    limit=64 workers=32 candidate_count=15 rollout_count=3 fixed_length=0
    specs=("job-runtime-kv-budget|control" "tree-frontier|tree")
    ;;
  full-p2)
    limit=64 workers=32 candidate_count=4 rollout_count=2 fixed_length=0
    specs=("job-runtime-kv-budget|control" "tree-frontier|tree")
    ;;
  *)
    echo "unknown phase: $phase" >&2
    exit 2
    ;;
esac

for spec in "${specs[@]}"; do
  IFS='|' read -r variant label <<<"$spec"
  output=$output_root/$phase/$label
  container=$container_prefix-$phase-$label
  if [[ -s "$output/benchmark.json" ]]; then
    echo "[$(date -Is)] skipping completed $phase/$label"
    continue
  fi
  CIS_WORKSPACE=$workspace \
  CIS_RUN_NAMESPACE="cis-tree-frontier-20260915:$phase" \
  CIS_LIMIT=$limit \
  CIS_WORKERS=$workers \
  CIS_CANDIDATE_COUNT=$candidate_count \
  CIS_ROLLOUT_COUNT=$rollout_count \
  CIS_ACTIVE_STEP_MAX_LIMIT=32 \
  CIS_ACTIVE_STEP_KV_CAPACITY_FRACTION=0.80 \
  CIS_FIXED_LENGTH=$fixed_length \
    "$runner" "$variant" "$devices" "$output" "$port" "$container"
  status=$(docker wait "$container")
  docker logs "$container" >"$output/container.log" 2>&1 || true
  docker rm "$container" >/dev/null
  if [[ "$status" != 0 ]]; then
    echo "run failed: $phase/$label (container exit $status)" >&2
    exit "$status"
  fi
done

roots=()
for spec in "${specs[@]}"; do
  IFS='|' read -r _ label <<<"$spec"
  [[ -s "$output_root/$phase/$label/benchmark.json" ]] \
    && roots+=("$output_root/$phase/$label")
done
if (( ${#roots[@]} > 0 )); then
  PYTHONPATH=$workspace/src python3 \
    "$workspace/experiments/swebench/summarize_cis_step_ab.py" \
    "${roots[@]}" --output "$output_root/$phase/summary.json"
fi
