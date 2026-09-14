#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 DEVICES OUTPUT_ROOT PORT CONTAINER_PREFIX MODE" >&2
  echo "MODE: context | subtree | all" >&2
  exit 2
fi

devices=$1
output_root=$2
port=$3
container_prefix=$4
mode=$5
workspace=$(cd "$(dirname "$0")/../.." && pwd)
runner=$workspace/experiments/swebench/run_cis_forest_ab_remote.sh
strata_root=${CIS_CONTEXT_STRATA_ROOT:-/data/disk/wangzili/cis-context-strata}
long_context_root=${CIS_LONG_CONTEXT_ROOT:-/data/disk/wangzili/cis-context-strata-source}
public_workload=${CIS_PUBLIC_WORKLOAD_DIR:-/data/disk/wangzili/cis-artifacts-fc11386/workloads/public-256}
fixed_length=${CIS_MATRIX_FIXED_LENGTH:-0}

case "$mode" in
  context)
    runs=(
      "medium-static|medium|public-256.jsonl|65536|step-gang|8|3|16|32|32"
      "medium-runtime|medium|public-256.jsonl|65536|step-runtime-kv-budget|8|3|16|32|32"
      "long-static|long|public-256.jsonl|65536|step-gang|8|3|16|32|32"
      "long-runtime|long|public-256.jsonl|65536|step-runtime-kv-budget|8|3|16|32|32"
      "very-long-static|32k-64k|public-32.jsonl|65536|step-gang|8|3|16|32|32"
      "very-long-runtime|32k-64k|public-32.jsonl|65536|step-runtime-kv-budget|8|3|16|32|32"
      "extended-static|64k-128k|public-32.jsonl|131072|step-gang|8|3|16|32|32"
      "extended-runtime|64k-128k|public-32.jsonl|131072|step-runtime-kv-budget|8|3|16|32|32"
    )
    ;;
  subtree)
    runs=(
      "p0-static|public|public-256.jsonl|65536|step-gang|15|3|16|32|64"
      "p0-runtime|public|public-256.jsonl|65536|step-runtime-kv-budget|15|3|16|32|64"
      "p0-subtree-k8|public|public-256.jsonl|65536|step-subtree-8|15|3|16|32|64"
    )
    ;;
  all)
    "$0" "$devices" "$output_root" "$port" "$container_prefix" context
    "$0" "$devices" "$output_root" "$port" "$container_prefix" subtree
    exit 0
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

mkdir -p "$output_root"
for spec in "${runs[@]}"; do
  IFS='|' read -r label stratum workload_file max_model_len variant candidates rollouts step_limit workers limit \
    <<<"$spec"
  output=$output_root/$label
  container=$container_prefix-$label
  if [[ -s "$output/benchmark.json" ]]; then
    echo "[$(date -Is)] skipping completed $label"
    continue
  fi
  if [[ "$stratum" == public ]]; then
    workload=$public_workload
  elif [[ "$stratum" == "32k-64k" || "$stratum" == "64k-128k" ]]; then
    workload=$long_context_root/$stratum
  else
    workload=$strata_root/$stratum
  fi
  [[ -s "$workload/$workload_file" ]] || {
    echo "missing workload: $workload/$workload_file" >&2
    exit 1
  }
  echo "[$(date -Is)] starting $label variant=$variant workload=$workload"
  CIS_WORKSPACE=$workspace \
  CIS_PUBLIC_WORKLOAD_DIR=$workload \
  CIS_PUBLIC_WORKLOAD_FILE=$workload_file \
  CIS_RUN_NAMESPACE="context-transfer-20260914:$label" \
  CIS_WORKERS=$workers \
  CIS_LIMIT=$limit \
  CIS_CANDIDATE_COUNT=$candidates \
  CIS_ROLLOUT_COUNT=$rollouts \
  CIS_ACTIVE_STEP_LIMIT=$step_limit \
  CIS_ACTIVE_STEP_MAX_LIMIT=64 \
  CIS_MAX_MODEL_LEN=$max_model_len \
  CIS_FIXED_LENGTH=$fixed_length \
    "$runner" "$variant" "$devices" "$output" "$port" "$container"
  status=$(docker wait "$container")
  docker logs "$container" >"$output/container.log" 2>&1 || true
  docker rm "$container" >/dev/null
  if [[ "$status" != 0 ]]; then
    echo "run failed: $label (container exit $status)" >&2
    exit "$status"
  fi
  [[ -s "$output/benchmark.json" ]] || {
    echo "run did not produce benchmark.json: $label" >&2
    exit 1
  }
  echo "[$(date -Is)] completed $label"
done

roots=()
for spec in "${runs[@]}"; do
  IFS='|' read -r label _ <<<"$spec"
  [[ -s "$output_root/$label/benchmark.json" ]] && roots+=("$output_root/$label")
done
if (( ${#roots[@]} > 0 )); then
  PYTHONPATH=$workspace/src python3 \
    "$workspace/experiments/swebench/summarize_cis_step_ab.py" \
    "${roots[@]}" --output "$output_root/$mode-summary.json"
fi
