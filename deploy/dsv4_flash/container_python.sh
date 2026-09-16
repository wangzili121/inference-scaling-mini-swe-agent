#!/usr/bin/env bash
set -euo pipefail

CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ ! -r "$CANN_ENV" ]]; then
  printf 'CANN environment script missing: %s\n' "$CANN_ENV" >&2
  exit 1
fi
source "$CANN_ENV"
export PYTHONPATH="/workspace/src${PYTHONPATH:+:$PYTHONPATH}"
exec python "$@"
