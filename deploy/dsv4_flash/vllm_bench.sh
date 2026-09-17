#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CIS_ENV_FILE:-$HERE/.env}"
: "${ARTIFACT_DIR:?set ARTIFACT_DIR}"
CONTAINER_NAME="${CONTAINER_NAME:-cis-dsv4-0731}"
PORT="${PORT:-8123}"
MODE="${1:-capacity}"
WORKLOAD="${2:-$ARTIFACT_DIR/vllm-bench/coding-prompts.jsonl}"
MAX_CONCURRENCY="${3:-8}"
REQUEST_RATE="${4:-inf}"
OUTPUT_LEN="${BENCH_OUTPUT_LEN:-512}"
NUM_PROMPTS="${BENCH_NUM_PROMPTS:-64}"
BURSTINESS="${BENCH_BURSTINESS:-1.0}"

case "$MODE" in
  capacity)
    REQUEST_RATE=inf
    ;;
  steady)
    [[ "$REQUEST_RATE" != inf ]] || {
      printf 'steady mode requires a finite request rate as argument 4\n' >&2
      exit 2
    }
    BURSTINESS=1.0
    ;;
  bursty)
    [[ "$REQUEST_RATE" != inf ]] || {
      printf 'bursty mode requires a finite request rate as argument 4\n' >&2
      exit 2
    }
    BURSTINESS="${BENCH_BURSTINESS:-0.3}"
    ;;
  *)
    printf 'Usage: %s {capacity|steady|bursty} WORKLOAD [MAX_CONCURRENCY] [REQUEST_RATE]\n' "$0" >&2
    exit 2
    ;;
esac

ARTIFACT_DIR="$(cd "$ARTIFACT_DIR" && pwd)"
WORKLOAD="$(cd "$(dirname "$WORKLOAD")" && pwd)/$(basename "$WORKLOAD")"
if [[ ! -f "$WORKLOAD" ]]; then
  printf 'Missing vLLM custom JSONL workload: %s\n' "$WORKLOAD" >&2
  exit 2
fi
if [[ "$WORKLOAD" != "$ARTIFACT_DIR/"* ]]; then
  printf 'Workload must be below ARTIFACT_DIR so the container can read it.\n' >&2
  exit 2
fi
INSIDE_WORKLOAD="/artifacts/${WORKLOAD#"$ARTIFACT_DIR/"}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="/artifacts/vllm-bench/results/$STAMP-$MODE-c$MAX_CONCURRENCY-r$REQUEST_RATE"
mkdir -p "$ARTIFACT_DIR/vllm-bench/results"

docker exec "$CONTAINER_NAME" bash -lc \
  'source /usr/local/Ascend/ascend-toolkit/set_env.sh && vllm bench serve --help >/dev/null'

docker exec "$CONTAINER_NAME" bash -lc "
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  vllm bench serve \
    --backend openai-chat \
    --host 127.0.0.1 \
    --port '$PORT' \
    --endpoint /v1/chat/completions \
    --model dsv4-cis \
    --tokenizer /models/dsv4 \
    --tokenizer-mode deepseek_v4 \
    --trust-remote-code \
    --dataset-name custom \
    --dataset-path '$INSIDE_WORKLOAD' \
    --custom-output-len '$OUTPUT_LEN' \
    --num-prompts '$NUM_PROMPTS' \
    --request-rate '$REQUEST_RATE' \
    --burstiness '$BURSTINESS' \
    --max-concurrency '$MAX_CONCURRENCY' \
    --temperature 1.0 \
    --top-p 1.0 \
    --no-stream \
    --percentile-metrics e2el \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --save-detailed \
    --result-dir '$RESULT_DIR' \
    --seed 20260908
"

curl -fsS "http://127.0.0.1:$PORT/v1/diagnostics" \
  > "$ARTIFACT_DIR/vllm-bench/results/$STAMP-$MODE-c$MAX_CONCURRENCY-r$REQUEST_RATE-diagnostics.json"
printf 'vLLM result directory: %s/%s\n' "$ARTIFACT_DIR" "${RESULT_DIR#/artifacts/}"
