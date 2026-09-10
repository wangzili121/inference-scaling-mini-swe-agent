#!/usr/bin/env bash
set -euo pipefail

output=${1:-/data/disk/wangzili/cis-mixed-profile/p0-four-card-torch}
container=${2:-cis-p0-mixed-profile}
port=${3:-18800}
image=${CIS_IMAGE:-wangzili/vllm-ascend:v0.18.0-msprof1.2.2-tzdata2025.3-pandas2.2.3}
workspace=${CIS_WORKSPACE:-/data/disk/wangzili/inference-scaling-mini-swe-agent-cis-forest}
model=${CIS_MODEL_DIR:-/data/disk/models/Qwen3-Coder-30B-A3B-Instruct}
public_workload=${CIS_PUBLIC_WORKLOAD_DIR:-/data/disk/wangzili/cis-artifacts-fc11386/workloads/public-256}
self_workload=${CIS_SELF_WORKLOAD_DIR:-/data/disk/wangzili/cis-artifacts-629451f/workloads/self-128}
categorical=${CIS_CATEGORICAL_DIR:-/data/disk/wangzili/vllm-categorical-runtime}
cache=${CIS_VLLM_CACHE:-/data/disk/wangzili/vllm-cache-v018-cis-mixed}

for path in "$workspace" "$model" "$public_workload" "$self_workload" "$categorical"; do
  [[ -e "$path" ]] || { echo "missing dependency: $path" >&2; exit 1; }
done
[[ -x /usr/local/bin/npu-smi ]] || { echo "npu-smi is unavailable" >&2; exit 1; }
if ss -ltnH | grep -q ":$port "; then
  echo "port is already in use: $port" >&2
  exit 1
fi

if [[ -e "$output" ]] && ! rm -rf "$output" 2>/dev/null; then
  output_parent=$(dirname "$output")
  output_name=$(basename "$output")
  docker run --rm --entrypoint /bin/bash -v "$output_parent":/cleanup "$image" \
    -lc 'rm -rf -- "/cleanup/$1"' _ "$output_name"
fi
mkdir -p "$output" "$cache"
docker rm -f "$container" >/dev/null 2>&1 || true
printf '%q ' "$0" "$@" >"$output/launch-command.txt"
printf '\n' >>"$output/launch-command.txt"

docker run -d \
  --name "$container" \
  --network host \
  --entrypoint /bin/bash \
  -e CIS_MODEL_PATH=/models/conditional-is \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$model":/models/conditional-is:ro \
  -v "$workspace":/workspace:ro \
  -v "$public_workload":/workloads/public:ro \
  -v "$self_workload":/workloads/self:ro \
  -v "$categorical":/categorical:ro \
  -v "$categorical"/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so:/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so:ro \
  -v "$categorical"/runtime/sampler.py:/vllm-workspace/vllm-ascend/vllm_ascend/sample/sampler.py:ro \
  -v "$categorical"/vllm_ascend/libvllm_ascend_kernels.so:/vllm-workspace/vllm-ascend/vllm_ascend/libvllm_ascend_kernels.so:ro \
  -v "$categorical"/vllm_ascend/_cann_ops_custom:/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom:ro \
  -v "$cache":/root/.cache/vllm \
  -v "$output":/artifacts \
  "$image" -lc "
    export PYTHONPATH=/workspace/src:\$PYTHONPATH
    exec /usr/local/python3.11.14/bin/python -m inference_scaling.swe_agent.profile \
      --config /workspace/configs/swebench/conditional_is_smoke.toml \
      --workload /workloads/public/public-256.jsonl \
      --warmup-workload /workloads/self/warmup-4.jsonl \
      --tensor-parallel-size 2 --pipeline-parallel-size 1 \
      --profiler torch --limit 64 --categorical-root /categorical \
      --set generation.max_new_tokens=512 \
      --set vllm.max_model_len=65536 \
      --set vllm.max_num_seqs=256 \
      --set vllm.max_num_batched_tokens=32768 \
      --set vllm.gpu_memory_utilization=0.90 \
      --set vllm.engine_kwargs.max_num_partial_prefills=1 \
      --set vllm.engine_kwargs.max_long_partial_prefills=1 \
      --env VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE=1 \
      --env ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend \
      --env LD_PRELOAD=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so \
      --output-directory /artifacts \
      --devices 0,1 --devices 2,3 \
      --port $port --routing round_robin --workers 64 \
      --candidate-count 15 --rollout-count 3 --block-size 128 \
      --ramp-up-seconds 60 --profile-seconds 15 \
      --profile-prefix p0-mixed \
      > /artifacts/launcher.log 2>&1
  "

echo "$container"
