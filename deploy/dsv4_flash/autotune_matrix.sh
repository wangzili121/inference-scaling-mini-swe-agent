#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SOURCE_ENV="${CIS_ENV_FILE:-$HERE/.env}"
SUITE="${1:-standard}"

if [[ ! -f "$SOURCE_ENV" ]]; then
  printf 'Missing %s; copy .env.example and configure the internal host first.\n' "$SOURCE_ENV" >&2
  exit 2
fi
case "$SUITE" in
  quick)
    ENGINE_PROFILES_DEFAULT="32:8192 64:16384"
    TUNE_PROMPTS=4
    FINAL_MEDIUM_PROMPTS=8
    FINAL_LONG_PROMPTS=0
    FINAL_EXTENDED_PROMPTS=0
    CIS_CONCURRENCIES="4"
    DIRECT_CONCURRENCIES="4 8"
    ARRIVAL_MODES="capacity"
    ;;
  standard)
    ENGINE_PROFILES_DEFAULT="32:8192 64:16384 128:32768"
    TUNE_PROMPTS=8
    FINAL_MEDIUM_PROMPTS=16
    FINAL_LONG_PROMPTS=8
    FINAL_EXTENDED_PROMPTS=0
    CIS_CONCURRENCIES="4 8"
    DIRECT_CONCURRENCIES="4 8 16"
    ARRIVAL_MODES="capacity steady bursty"
    ;;
  full)
    ENGINE_PROFILES_DEFAULT="32:8192 64:16384 128:32768 256:65536"
    TUNE_PROMPTS=16
    FINAL_MEDIUM_PROMPTS=32
    FINAL_LONG_PROMPTS=16
    FINAL_EXTENDED_PROMPTS=8
    CIS_CONCURRENCIES="2 4 8 16"
    DIRECT_CONCURRENCIES="4 8 16 32"
    ARRIVAL_MODES="capacity steady bursty"
    ;;
  *) printf 'Usage: %s {quick|standard|full}\n' "$0" >&2; exit 2 ;;
esac

set -a
source "$SOURCE_ENV"
set +a
: "${ARTIFACT_DIR:?set ARTIFACT_DIR}"
: "${DEVICES:?set DEVICES}"
: "${CONTAINER_NAME:?set CONTAINER_NAME}"

ENGINE_PROFILES="${AUTOTUNE_ENGINE_PROFILES:-$ENGINE_PROFILES_DEFAULT}"
MAX_MODEL_LEN_SUITE="${AUTOTUNE_MAX_MODEL_LEN:-65536}"
ROOT_ARTIFACT_DIR="$(mkdir -p "$ARTIFACT_DIR" && cd "$ARTIFACT_DIR" && pwd)"
MATRIX_ID="$(date -u +%Y%m%dT%H%M%SZ)-$SUITE"
MATRIX_DIR="$ROOT_ARTIFACT_DIR/autotune/$MATRIX_ID"
mkdir -p "$MATRIX_DIR/workloads" "$MATRIX_DIR/deployments"
printf '%s\n' "$MATRIX_ID" > "$MATRIX_DIR/matrix-id.txt"

if ! command -v flock >/dev/null 2>&1; then
  printf 'flock is required to reserve the selected NPUs.\n' >&2
  exit 2
fi
LOCK_KEY="${DEVICES//,/-}"
exec 9>"/tmp/cis-dsv4-npu-$LOCK_KEY.lock"
if ! flock -n 9; then
  printf 'Another DSV4 run holds the lock for NPUs %s\n' "$DEVICES" >&2
  exit 2
fi

stop_own_container() {
  if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    CIS_ENV_FILE="$SOURCE_ENV" bash "$HERE/run.sh" stop >/dev/null 2>&1 || true
  fi
}

render_report() {
  if find "$MATRIX_DIR" -name run-metadata.json -print -quit | grep -q .; then
    python3 "$HERE/render_autotune_report.py" "$MATRIX_DIR" \
      --output "$MATRIX_DIR/autotune-report.html" >/dev/null || true
  fi
}

trap 'render_report' EXIT

stop_own_container
python3 "$HERE/check_npu_idle.py" \
  --devices "$DEVICES" \
  --max-idle-hbm-mb "${MAX_IDLE_HBM_MB:-8192}" \
  --raw-output "$MATRIX_DIR/npu-before.txt" \
  --json-output "$MATRIX_DIR/npu-before.json"

python3 "$HERE/make_coding_workload.py" \
  --output "$MATRIX_DIR/workloads/tune.jsonl" \
  --count "$TUNE_PROMPTS" \
  --output-tokens 512 \
  --input-buckets 4096,8192,16384 \
  --profile-name tune-mixed
python3 "$HERE/make_coding_workload.py" \
  --output "$MATRIX_DIR/workloads/coding-medium.jsonl" \
  --count "$FINAL_MEDIUM_PROMPTS" \
  --output-tokens 512 \
  --input-buckets 4096,4096,8192,8192,16384 \
  --profile-name coding-medium
if ((FINAL_LONG_PROMPTS > 0)); then
  python3 "$HERE/make_coding_workload.py" \
    --output "$MATRIX_DIR/workloads/coding-long.jsonl" \
    --count "$FINAL_LONG_PROMPTS" \
    --output-tokens 1024 \
  --input-buckets 16384,24576,32768 \
    --profile-name coding-long
fi
if ((FINAL_EXTENDED_PROMPTS > 0)); then
  python3 "$HERE/make_coding_workload.py" \
    --output "$MATRIX_DIR/workloads/coding-extended.jsonl" \
    --count "$FINAL_EXTENDED_PROMPTS" \
    --output-tokens 2048 \
    --input-buckets 8192,16384,32768 \
    --profile-name coding-extended
fi

count_jsonl() {
  python3 - "$1" <<'PY'
import sys
print(sum(bool(line.strip()) for line in open(sys.argv[1], encoding="utf-8")))
PY
}
if [[ -n "${AUTOTUNE_TUNE_WORKLOAD:-}" ]]; then
  cp "$AUTOTUNE_TUNE_WORKLOAD" "$MATRIX_DIR/workloads/tune.jsonl"
  TUNE_PROMPTS="$(count_jsonl "$MATRIX_DIR/workloads/tune.jsonl")"
fi
if [[ -n "${AUTOTUNE_MEDIUM_WORKLOAD:-}" ]]; then
  cp "$AUTOTUNE_MEDIUM_WORKLOAD" "$MATRIX_DIR/workloads/coding-medium.jsonl"
  FINAL_MEDIUM_PROMPTS="$(count_jsonl "$MATRIX_DIR/workloads/coding-medium.jsonl")"
fi
if [[ -n "${AUTOTUNE_LONG_WORKLOAD:-}" ]]; then
  cp "$AUTOTUNE_LONG_WORKLOAD" "$MATRIX_DIR/workloads/coding-long.jsonl"
  FINAL_LONG_PROMPTS="$(count_jsonl "$MATRIX_DIR/workloads/coding-long.jsonl")"
fi

write_env() {
  local destination="$1" artifact="$2" variant="$3" mns="$4" mbt="$5" memory="$6"
  python3 - "$SOURCE_ENV" "$destination" "$artifact" "$variant" "$mns" "$mbt" "$memory" "$MAX_MODEL_LEN_SUITE" <<'PY'
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:3])
updates = {
    "ARTIFACT_DIR": sys.argv[3],
    "SCHEDULER_VARIANT": sys.argv[4],
    "MAX_NUM_SEQS": sys.argv[5],
    "MAX_NUM_BATCHED_TOKENS": sys.argv[6],
    "GPU_MEMORY_UTILIZATION": sys.argv[7],
    "MAX_MODEL_LEN": sys.argv[8],
    "MAX_NEW_TOKENS": "1024",
    "MAX_COMPLETION_TOKENS": "2048",
    "CONFIRM_DEVICES_FREE": "yes",
    "BENCH_NUM_PROMPTS": "16",
}
seen = set()
lines = []
for line in source.read_text(encoding="utf-8").splitlines():
    key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
    if key in updates:
        lines.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        lines.append(line)
for key, value in updates.items():
    if key not in seen:
        lines.append(f"{key}={value}")
destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

launch_deployment() {
  local name="$1" variant="$2" mns="$3" mbt="$4" memory="$5"
  local deployment="$MATRIX_DIR/deployments/$name"
  mkdir -p "$deployment/artifacts"
  write_env "$deployment/effective.env" "$deployment/artifacts" "$variant" "$mns" "$mbt" "$memory"
  stop_own_container
  if ! CIS_ENV_FILE="$deployment/effective.env" bash "$HERE/run.sh" launch \
    > >(tee "$deployment/launch.log" >&2) 2>&1; then
    printf '{"status":"launch_failed","mns":%s,"mbt":%s,"memory":%s,"variant":"%s"}\n' \
      "$mns" "$mbt" "$memory" "$variant" > "$deployment/failure.json"
    docker logs --timestamps "$CONTAINER_NAME" > "$deployment/container.failed.log" 2>&1 || true
    return 1
  fi
  printf '%s\n' "$deployment"
}

copy_workload() {
  local deployment="$1" source="$2"
  mkdir -p "$deployment/artifacts/workloads"
  cp "$source" "$deployment/artifacts/workloads/$(basename "$source")"
  if [[ -f "${source%.jsonl}.manifest.json" ]]; then
    cp "${source%.jsonl}.manifest.json" \
      "$deployment/artifacts/workloads/$(basename "${source%.jsonl}.manifest.json")"
  fi
  printf '%s\n' "$deployment/artifacts/workloads/$(basename "$source")"
}

run_bench() {
  local deployment="$1" env_file="$2" workload_source="$3" profile="$4"
  local api_mode="$5" label="$6" arrival="$7" concurrency="$8" rate="$9"
  local prompts="${10}" output_tokens="${11}"
  local workload
  workload="$(copy_workload "$deployment" "$workload_source")"
  BENCH_API_MODE="$api_mode" \
  BENCH_RUN_LABEL="$label" \
  BENCH_WORKLOAD_PROFILE="$profile" \
  BENCH_NUM_PROMPTS="$prompts" \
  BENCH_OUTPUT_LEN="$output_tokens" \
  BENCH_RESULT_SUBDIR=results \
  CIS_ENV_FILE="$env_file" \
    bash "$HERE/vllm_bench.sh" "$arrival" "$workload" "$concurrency" "$rate"
  render_report
}

latest_throughput() {
  local directory="$1" label="$2"
  python3 - "$directory" "$label" <<'PY'
import json, sys
from pathlib import Path
root, label = Path(sys.argv[1]), sys.argv[2]
found = []
for metadata_path in root.rglob("run-metadata.json"):
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("label") != label:
        continue
    for path in metadata_path.parent.glob("*.json"):
        try:
            value = json.loads(path.read_text())
        except Exception:
            continue
        if "request_throughput" in value and not value.get("failed"):
            found.append((float(value["request_throughput"]), int(metadata["max_concurrency"])))
if not found:
    raise SystemExit(1)
print(*max(found))
PY
}

TUNE_RESULTS="$MATRIX_DIR/tuning.tsv"
printf 'mns\tmbt\tmemory\tjobs_per_second\tstatus\tdeployment\n' > "$TUNE_RESULTS"
BEST_RATE=""
BEST_MNS=""
BEST_MBT=""
BEST_MEMORY=""

for profile in $ENGINE_PROFILES; do
  IFS=: read -r mns mbt memory <<< "$profile"
  memory="${memory:-0.90}"
  name="tune-mns${mns}-mbt${mbt}-mem${memory//./}"
  if ! deployment="$(launch_deployment "$name" baseline "$mns" "$mbt" "$memory")"; then
    printf '%s\t%s\t%s\t\tlaunch_failed\t%s\n' "$mns" "$mbt" "$memory" "$name" >> "$TUNE_RESULTS"
    continue
  fi
  env_file="$deployment/effective.env"
  label="tune-mns${mns}-mbt${mbt}"
  if run_bench "$deployment" "$env_file" "$MATRIX_DIR/workloads/tune.jsonl" \
      tune-mixed cis "$label" capacity 4 inf "$TUNE_PROMPTS" 512; then
    read -r rate _concurrency < <(latest_throughput "$deployment" "$label")
    printf '%s\t%s\t%s\t%s\tok\t%s\n' "$mns" "$mbt" "$memory" "$rate" "$name" >> "$TUNE_RESULTS"
    if [[ -z "$BEST_RATE" ]] || python3 - "$rate" "$BEST_RATE" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
    then
      BEST_RATE="$rate"; BEST_MNS="$mns"; BEST_MBT="$mbt"; BEST_MEMORY="$memory"
    fi
  else
    printf '%s\t%s\t%s\t\tbenchmark_failed\t%s\n' "$mns" "$mbt" "$memory" "$name" >> "$TUNE_RESULTS"
  fi
done

if [[ -z "$BEST_MNS" ]]; then
  printf 'All deployment candidates failed. See %s\n' "$TUNE_RESULTS" >&2
  exit 1
fi
printf '{"max_num_seqs":%s,"max_num_batched_tokens":%s,"gpu_memory_utilization":%s,"tuning_jobs_per_second":%s}\n' \
  "$BEST_MNS" "$BEST_MBT" "$BEST_MEMORY" "$BEST_RATE" > "$MATRIX_DIR/best-deployment.json"

run_final_mode() {
  local variant="$1" api_mode="$2" label="$3"
  local deployment_name="final-$variant"
  if [[ "$api_mode" == direct ]]; then
    deployment_name="final-baseline"
  fi
  local deployment="$MATRIX_DIR/deployments/$deployment_name"
  local env_file="$deployment/effective.env"
  if [[ ! -f "$env_file" ]]; then
    deployment="$(launch_deployment "$deployment_name" "$variant" "$BEST_MNS" "$BEST_MBT" "$BEST_MEMORY")"
    env_file="$deployment/effective.env"
  elif ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    CIS_ENV_FILE="$env_file" bash "$HERE/run.sh" launch
  fi
  local concurrencies="$CIS_CONCURRENCIES"
  [[ "$api_mode" == direct ]] && concurrencies="$DIRECT_CONCURRENCIES"
  local profile_name workload prompts output_tokens
  for profile_name in coding-medium coding-long coding-extended; do
    if [[ "$profile_name" == coding-medium ]]; then
      workload="$MATRIX_DIR/workloads/coding-medium.jsonl"
      prompts="$FINAL_MEDIUM_PROMPTS"
      output_tokens=512
    elif [[ "$profile_name" == coding-long ]]; then
      ((FINAL_LONG_PROMPTS > 0)) || continue
      workload="$MATRIX_DIR/workloads/coding-long.jsonl"
      prompts="$FINAL_LONG_PROMPTS"
      output_tokens=1024
    else
      ((FINAL_EXTENDED_PROMPTS > 0)) || continue
      workload="$MATRIX_DIR/workloads/coding-extended.jsonl"
      prompts="$FINAL_EXTENDED_PROMPTS"
      output_tokens=2048
    fi
    for concurrency in $concurrencies; do
      run_bench "$deployment" "$env_file" "$workload" "$profile_name" \
        "$api_mode" "$profile_name-$label" capacity "$concurrency" inf \
        "$prompts" "$output_tokens"
    done
    if [[ " $ARRIVAL_MODES " == *" steady "* ]]; then
      read -r capacity_rate best_concurrency < <(
        latest_throughput "$deployment" "$profile_name-$label"
      )
      steady_rate="$(python3 - "$capacity_rate" <<'PY'
import sys
print(f"{float(sys.argv[1]) * 0.70:.8f}")
PY
)"
      run_bench "$deployment" "$env_file" "$workload" "$profile_name" \
        "$api_mode" "$profile_name-$label" steady "$best_concurrency" "$steady_rate" \
        "$prompts" "$output_tokens"
      BENCH_BURSTINESS=0.3 run_bench "$deployment" "$env_file" "$workload" "$profile_name" \
        "$api_mode" "$profile_name-$label" bursty "$best_concurrency" "$steady_rate" \
        "$prompts" "$output_tokens"
    fi
  done
}

# Direct AR and flat CIS share one model load and identical engine settings.
run_final_mode baseline direct direct
run_final_mode baseline cis cis
stop_own_container
run_final_mode pressure_tree cis tree

render_report
docker logs --timestamps "$CONTAINER_NAME" > "$MATRIX_DIR/final-container.log" 2>&1 || true
printf '{"status":"ok","suite":"%s","artifacts":"%s"}\n' "$SUITE" "$MATRIX_DIR" \
  | tee "$MATRIX_DIR/result.json"
printf 'Autotune complete. Report: %s/autotune-report.html\n' "$MATRIX_DIR"
