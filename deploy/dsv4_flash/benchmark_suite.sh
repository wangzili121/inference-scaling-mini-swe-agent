#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOAD="${1:?pass a custom JSONL workload under ARTIFACT_DIR}"

# Capacity first. Use its measured jobs/s to choose finite rates for steady and
# bursty runs; those two are intentionally not guessed here.
read -r -a concurrencies <<< "${BENCH_CONCURRENCIES:-2 4 8 16}"
for concurrency in "${concurrencies[@]}"; do
  bash "$HERE/vllm_bench.sh" capacity "$WORKLOAD" "$concurrency"
done

printf '\nCapacity sweep complete. Pick approximately 50%%, 70%% and 85%% of the\n'
printf 'measured saturation jobs/s and run, for example:\n'
printf '  bash %s steady %s 16 RATE\n' "$HERE/vllm_bench.sh" "$WORKLOAD"
printf '  BENCH_BURSTINESS=0.3 bash %s bursty %s 16 RATE\n' "$HERE/vllm_bench.sh" "$WORKLOAD"
