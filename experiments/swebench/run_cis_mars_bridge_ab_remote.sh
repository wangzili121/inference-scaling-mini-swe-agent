#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 DEVICES OUTPUT_ROOT PORT CONTAINER_PREFIX smoke|full" >&2
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
    limit=16
    workers=16
    ;;
  full)
    limit=32
    workers=32
    ;;
  *)
    echo "phase must be smoke or full" >&2
    exit 2
    ;;
esac

# P1 fixed work makes all four arms execute the same generated-token workload.
# arm|variant|order
specs=(
  "baseline|baseline|mars_first"
  "mars-only|mars-only|mars_first"
  "cis-only|job-runtime-kv-budget|mars_first"
  "mars-cis-mars-first|mars-cis|mars_first"
  "mars-cis-cis-first|mars-cis|cis_first"
)

for spec in "${specs[@]}"; do
  IFS='|' read -r label variant order <<<"$spec"
  output=$output_root/$phase/$label
  container=$container_prefix-$phase-$label
  if [[ -s "$output/benchmark.json" ]]; then
    echo "[$(date -Is)] skipping completed $phase/$label"
    continue
  fi
  CIS_WORKSPACE=$workspace \
  CIS_RUN_NAMESPACE="cis-mars-bridge-20260915:$phase" \
  CIS_LIMIT=$limit \
  CIS_WORKERS=$workers \
  CIS_CANDIDATE_COUNT=8 \
  CIS_ROLLOUT_COUNT=3 \
  CIS_ACTIVE_STEP_MAX_LIMIT=32 \
  CIS_ACTIVE_STEP_KV_CAPACITY_FRACTION=0.80 \
  CIS_FIXED_LENGTH=1 \
  CIS_MARS_ACTIVE_WINDOW=256 \
  CIS_MARS_ORDER=$order \
  CIS_MARS_CIS_PRIORITY_POLICY=job_fifo \
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
  IFS='|' read -r label _ <<<"$spec"
  [[ -s "$output_root/$phase/$label/benchmark.json" ]] \
    && roots+=("$output_root/$phase/$label")
done
if (( ${#roots[@]} > 0 )); then
  PYTHONPATH=$workspace/src python3 \
    "$workspace/experiments/swebench/summarize_cis_step_ab.py" \
    "${roots[@]}" --output "$output_root/$phase/summary.json"
fi
