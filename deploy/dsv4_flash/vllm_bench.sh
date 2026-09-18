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
RESULT_SUBDIR="${BENCH_RESULT_SUBDIR:-vllm-bench/results}"
RESULT_SUBDIR="${RESULT_SUBDIR#/}"
if [[ "$RESULT_SUBDIR" == *..* ]]; then
  printf 'BENCH_RESULT_SUBDIR cannot contain ..\n' >&2
  exit 2
fi
RESULT_DIR="/artifacts/$RESULT_SUBDIR/$STAMP-$MODE-c$MAX_CONCURRENCY-r$REQUEST_RATE"
HOST_RESULT_DIR="$ARTIFACT_DIR/$RESULT_SUBDIR/$STAMP-$MODE-c$MAX_CONCURRENCY-r$REQUEST_RATE"
mkdir -p "$HOST_RESULT_DIR"
BENCH_LOG="$HOST_RESULT_DIR/vllm-bench.stdout.log"

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
    --extra-body '{\"temperature\":1.0,\"top_p\":1.0,\"n\":1}' \
    --no-stream \
    --percentile-metrics e2el \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --save-detailed \
    --result-dir '$RESULT_DIR' \
    --seed 20260908
" | tee "$BENCH_LOG"

if grep -Eq '^Successful requests:[[:space:]]+0[[:space:]]*$' "$BENCH_LOG"; then
  printf 'Benchmark produced zero successful requests; refusing to report it as a performance run.\n' >&2
  printf 'Inspect the CIS rejection reason with: docker logs --tail 100 %s\n' "$CONTAINER_NAME" >&2
  exit 1
fi
if grep -Eq '^Failed requests:[[:space:]]+[1-9][0-9]*[[:space:]]*$' "$BENCH_LOG"; then
  printf 'Benchmark contains failed requests; results are invalid. See %s\n' "$BENCH_LOG" >&2
  exit 1
fi

curl -fsS "http://127.0.0.1:$PORT/v1/diagnostics" \
  > "$ARTIFACT_DIR/$RESULT_SUBDIR/$STAMP-$MODE-c$MAX_CONCURRENCY-r$REQUEST_RATE-diagnostics.json"
printf 'vLLM result directory: %s/%s\n' "$ARTIFACT_DIR" "${RESULT_DIR#/artifacts/}"
