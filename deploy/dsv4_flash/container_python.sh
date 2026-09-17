#!/usr/bin/env bash
set -euo pipefail

CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ ! -r "$CANN_ENV" ]]; then
  printf 'CANN environment script missing: %s\n' "$CANN_ENV" >&2
  exit 1
fi
source "$CANN_ENV"
if [[ "${CIS_ENABLE_JEMALLOC:-false}" == true ]]; then
  JEMALLOC=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
  if [[ -r "$JEMALLOC" ]]; then
    export LD_PRELOAD="$JEMALLOC${LD_PRELOAD:+:$LD_PRELOAD}"
  else
    printf 'Warning: jemalloc requested but missing at %s; continuing.\n' "$JEMALLOC" >&2
  fi
fi
export PYTHONPATH="/workspace/plugins/cis-scheduler/src:/workspace/src${PYTHONPATH:+:$PYTHONPATH}"
exec python "$@"
