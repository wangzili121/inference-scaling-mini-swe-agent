#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 MODE DEVICES OUTPUT_ROOT PORT CONTAINER_PREFIX" >&2
  echo "MODE: smoke | full | confirm" >&2
  exit 2
fi

mode=$1
devices=$2
output_root=$3
port=$4
container_prefix=$5
workspace=$(cd "$(dirname "$0")/../.." && pwd)
runner=$workspace/experiments/swebench/run_cis_forest_ab_remote.sh
namespace=${CIS_RUN_NAMESPACE:-cis-step-ablation-20260912}

case "$mode" in
  smoke)
    runs=(
      "cap-only-smoke|step-cap-only|16|16"
      "priority-only-smoke|step-priority-only|16|16"
    )
    ;;
  full)
    runs=(
      "a-flat-w32|baseline|32|64"
      "b-flat-w6|baseline|6|64"
      "c-cap6-w32|step-cap-only|32|64"
      "d-priority-w32|step-priority-only|32|64"
      "e-cap6-priority-w32|step-gang|32|64"
    )
    ;;
  confirm)
    runs=(
      "a-flat-w32|baseline|32|64"
      "d-priority-w32|step-priority-only|32|64"
      "e-cap6-priority-w32|step-gang|32|64"
    )
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

mkdir -p "$output_root"
for spec in "${runs[@]}"; do
  IFS='|' read -r label variant workers limit <<<"$spec"
  output=$output_root/$label
  container=$container_prefix-$label
  echo "[$(date -Is)] starting $label variant=$variant workers=$workers limit=$limit"
  CIS_WORKSPACE=$workspace \
  CIS_RUN_NAMESPACE=$namespace \
  CIS_WORKERS=$workers \
  CIS_LIMIT=$limit \
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
