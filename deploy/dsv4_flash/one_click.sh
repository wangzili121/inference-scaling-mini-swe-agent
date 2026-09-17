#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SOURCE_ENV="${CIS_ENV_FILE:-$HERE/.env}"
VARIANT_OVERRIDE=""
if [[ "${1:-}" == baseline || "${1:-}" == pressure_tree ]]; then
  VARIANT_OVERRIDE="$1"
  shift
fi

if [[ ! -f "$SOURCE_ENV" ]]; then
  cp "$HERE/.env.example" "$SOURCE_ENV"
  python3 - "$SOURCE_ENV" "$REPO/artifacts/dsv4_flash" <<'PY'
import sys
from pathlib import Path

path, artifacts = Path(sys.argv[1]), sys.argv[2]
lines = path.read_text(encoding="utf-8").splitlines()
path.write_text(
    "\n".join(
        f"ARTIFACT_DIR={artifacts}" if line.startswith("ARTIFACT_DIR=") else line
        for line in lines
    )
    + "\n",
    encoding="utf-8",
)
PY
  printf 'Created %s with local defaults. Review MODEL_DIR and IMAGE, then rerun.\n' "$SOURCE_ENV" >&2
  exit 2
fi

set -a
source "$SOURCE_ENV"
set +a
if [[ -n "$VARIANT_OVERRIDE" ]]; then
  SCHEDULER_VARIANT="$VARIANT_OVERRIDE"
fi
: "${ARTIFACT_DIR:?set ARTIFACT_DIR}"
: "${DEVICES:?set DEVICES}"
: "${CONTAINER_NAME:?set CONTAINER_NAME}"

ROOT_ARTIFACT_DIR="$(mkdir -p "$ARTIFACT_DIR" && cd "$ARTIFACT_DIR" && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${SCHEDULER_VARIANT:-baseline}"
RUN_DIR="$ROOT_ARTIFACT_DIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

if ! command -v flock >/dev/null 2>&1; then
  printf 'flock is required to prevent two launchers from claiming the same cards.\n' >&2
  exit 2
fi
LOCK_KEY="${DEVICES//,/-}"
exec 9>"/tmp/cis-dsv4-npu-$LOCK_KEY.lock"
if ! flock -n 9; then
  printf 'Another DSV4 one-click run holds the lock for NPUs %s\n' "$DEVICES" >&2
  exit 2
fi

EFFECTIVE_ENV="$RUN_DIR/effective.env"
python3 - "$SOURCE_ENV" "$EFFECTIVE_ENV" "$RUN_DIR" "${SCHEDULER_VARIANT:-baseline}" <<'PY'
import sys
from pathlib import Path

source, destination, run_dir = map(Path, sys.argv[1:4])
updates = {
    "ARTIFACT_DIR": str(run_dir),
    "CONFIRM_DEVICES_FREE": "yes",
    "SCHEDULER_VARIANT": sys.argv[4],
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

failure_artifacts() {
  status=$?
  if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    docker logs --timestamps "$CONTAINER_NAME" > "$RUN_DIR/container.final.log" 2>&1 || true
    docker inspect "$CONTAINER_NAME" > "$RUN_DIR/container.inspect.json" 2>&1 || true
  fi
  printf '{"status":"failed","exit_code":%s,"run_id":"%s"}\n' "$status" "$RUN_ID" > "$RUN_DIR/result.json"
  printf 'One-click run failed. Artifacts: %s\n' "$RUN_DIR" >&2
  exit "$status"
}
trap failure_artifacts ERR

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  if [[ "${ONE_CLICK_RESTART_OWN_CONTAINER:-yes}" != yes ]]; then
    printf 'Container %s exists. Set ONE_CLICK_RESTART_OWN_CONTAINER=yes to replace only this named test container.\n' "$CONTAINER_NAME" >&2
    exit 2
  fi
  CIS_ENV_FILE="$EFFECTIVE_ENV" bash "$HERE/run.sh" stop
fi

python3 "$HERE/check_npu_idle.py" \
  --devices "$DEVICES" \
  --max-idle-hbm-mb "${MAX_IDLE_HBM_MB:-8192}" \
  --raw-output "$RUN_DIR/npu-before.txt" \
  --json-output "$RUN_DIR/npu-before.json"

CIS_ENV_FILE="$EFFECTIVE_ENV" bash "$HERE/run.sh" launch \
  2>&1 | tee "$RUN_DIR/launch.log"

WORKLOAD="${1:-}"
if [[ -z "$WORKLOAD" ]]; then
  WORKLOAD="$RUN_DIR/coding-capacity.jsonl"
  python3 "$HERE/make_coding_workload.py" \
    --output "$WORKLOAD" \
    --count "${BENCH_NUM_PROMPTS:-16}" \
    --output-tokens "${BENCH_OUTPUT_LEN:-512}" \
    | tee "$RUN_DIR/workload-generation.log"
else
  if [[ ! -f "$WORKLOAD" ]]; then
    printf 'Benchmark workload does not exist: %s\n' "$WORKLOAD" >&2
    exit 2
  fi
  cp "$WORKLOAD" "$RUN_DIR/$(basename "$WORKLOAD")"
  WORKLOAD="$RUN_DIR/$(basename "$WORKLOAD")"
fi

BENCH_RESULT_SUBDIR="benchmark" \
BENCH_CONCURRENCIES="${BENCH_CONCURRENCIES:-2 4 8 16}" \
CIS_ENV_FILE="$EFFECTIVE_ENV" \
  bash "$HERE/benchmark_suite.sh" "$WORKLOAD" \
  2>&1 | tee "$RUN_DIR/benchmark.log"

python3 "$HERE/summarize_benchmark.py" "$RUN_DIR/benchmark" \
  | tee "$RUN_DIR/capacity-summary.txt"

curl -fsS "http://127.0.0.1:${PORT:-8123}/v1/diagnostics" \
  > "$RUN_DIR/service-diagnostics.final.json"
docker logs --timestamps "$CONTAINER_NAME" > "$RUN_DIR/container.final.log" 2>&1
docker inspect "$CONTAINER_NAME" > "$RUN_DIR/container.inspect.json"
printf '{"status":"ok","run_id":"%s","variant":"%s","artifacts":"%s"}\n' \
  "$RUN_ID" "${SCHEDULER_VARIANT:-baseline}" "$RUN_DIR" \
  | tee "$RUN_DIR/result.json"
printf 'Service remains running. Complete results: %s\n' "$RUN_DIR"
