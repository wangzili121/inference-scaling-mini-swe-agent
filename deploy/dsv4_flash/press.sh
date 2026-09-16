#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CIS_ENV_FILE:-$HERE/.env}"
: "${ARTIFACT_DIR:?set ARTIFACT_DIR}"
CONTAINER_NAME="${CONTAINER_NAME:-cis-dsv4-0731}"
PORT="${PORT:-8123}"
WORKERS="${1:-8}"
WORKLOAD="${2:-$ARTIFACT_DIR/workload.jsonl}"
if [[ ! -f "$WORKLOAD" ]]; then
  printf 'Missing fixed workload: %s\n' "$WORKLOAD" >&2
  exit 2
fi
ARTIFACT_DIR="$(cd "$ARTIFACT_DIR" && pwd)"
WORKLOAD="$(cd "$(dirname "$WORKLOAD")" && pwd)/$(basename "$WORKLOAD")"
if [[ "$WORKLOAD" != "$ARTIFACT_DIR/"* ]]; then
  printf 'Workload must be under ARTIFACT_DIR so the container can read it.\n' >&2
  exit 2
fi
INSIDE_WORKLOAD="/artifacts/${WORKLOAD#"$ARTIFACT_DIR/"}"
OUT="/artifacts/pressure/$(date -u +%Y%m%dT%H%M%SZ)-w${WORKERS}-$$.json"
mkdir -p "$ARTIFACT_DIR/pressure"
python3 - "$WORKLOAD" "${MAX_MODEL_LEN:-32768}" "${MAX_NEW_TOKENS:-512}" <<'PY'
import json
import sys

path, context, output = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
if not records:
    raise SystemExit("workload is empty")
if any(not isinstance(record.get("diagnostics"), dict) or
       not isinstance(record["diagnostics"].get("prompt_tokens"), int)
       for record in records):
    raise SystemExit("every frozen call must contain diagnostics.prompt_tokens")
too_long = [record.get("request_id") for record in records
            if int(record.get("diagnostics", {}).get("prompt_tokens", 0)) + output > context]
if too_long:
    raise SystemExit(f"{len(too_long)} requests exceed max_model_len; first: {too_long[0]}")
print(f"Frozen CIS calls: {len(records)}; max_model_len: {context}")
PY
docker exec "$CONTAINER_NAME" bash /workspace/deploy/dsv4_flash/container_python.sh \
  -m inference_scaling.swe_agent.benchmark \
  --workload "$INSIDE_WORKLOAD" \
  --endpoint "http://127.0.0.1:$PORT" \
  --workers "$WORKERS" \
  --candidate-count "${PRESS_CANDIDATE_COUNT:-${CANDIDATE_COUNT:-4}}" \
  --rollout-count "${PRESS_ROLLOUT_COUNT:-${ROLLOUT_COUNT:-2}}" \
  --block-size "${BLOCK_SIZE:-128}" \
  --output "$OUT"
printf 'Raw benchmark: %s/pressure/%s\n' "$ARTIFACT_DIR" "$(basename "$OUT")"
