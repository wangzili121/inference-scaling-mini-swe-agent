#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${CIS_ENV_FILE:-$HERE/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  printf 'Missing %s; run one_click.sh once to create it.\n' "$ENV_FILE" >&2
  exit 2
fi
set -a
source "$ENV_FILE"
set +a
: "${ARTIFACT_DIR:?set ARTIFACT_DIR}"
ROOT_ARTIFACT_DIR="$(mkdir -p "$ARTIFACT_DIR" && cd "$ARTIFACT_DIR" && pwd)"
AB_ID="$(date -u +%Y%m%dT%H%M%SZ)-ab"
AB_DIR="$ROOT_ARTIFACT_DIR/ab/$AB_ID"
mkdir -p "$AB_DIR"
WORKLOAD="$AB_DIR/coding-capacity.jsonl"

python3 "$HERE/make_coding_workload.py" \
  --output "$WORKLOAD" \
  --count "${BENCH_NUM_PROMPTS:-16}" \
  --output-tokens "${BENCH_OUTPUT_LEN:-512}"

CIS_ENV_FILE="$ENV_FILE" bash "$HERE/one_click.sh" baseline "$WORKLOAD"
CIS_ENV_FILE="$ENV_FILE" bash "$HERE/one_click.sh" pressure_tree "$WORKLOAD"

find "$ROOT_ARTIFACT_DIR/runs" -path '*/capacity-summary.md' -newer "$WORKLOAD" \
  -print | sort | tee "$AB_DIR/summary-files.txt"
printf 'A/B complete. Shared workload and result index: %s\n' "$AB_DIR"
