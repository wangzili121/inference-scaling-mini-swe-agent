#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 VARIANT DEVICES OUTPUT PORT CONTAINER" >&2
  echo "VARIANT: baseline | streaming | bounded" >&2
  exit 2
fi

variant=$1
devices=$2
output=$3
port=$4
container=$5

image=${CIS_IMAGE:-wangzili/vllm-ascend:v0.18.0-msprof1.2.2-tzdata2025.3-pandas2.2.3}
workspace=${CIS_WORKSPACE:-/data/disk/wangzili/inference-scaling-mini-swe-agent-cis-forest}
model=${CIS_MODEL_DIR:-/data/disk/models/Qwen3-Coder-30B-A3B-Instruct}
public_workload=${CIS_PUBLIC_WORKLOAD_DIR:-/data/disk/wangzili/cis-artifacts-fc11386/workloads/public-256}
self_workload=${CIS_SELF_WORKLOAD_DIR:-/data/disk/wangzili/cis-artifacts-629451f/workloads/self-128}
categorical=${CIS_CATEGORICAL_DIR:-/data/disk/wangzili/vllm-categorical-runtime}
cache=${CIS_VLLM_CACHE:-/data/disk/wangzili/vllm-cache-v018-cis-forest}

for path in "$workspace" "$model" "$public_workload" "$self_workload" "$categorical"; do
  [[ -e "$path" ]] || { echo "missing dependency: $path" >&2; exit 1; }
done
[[ -x /usr/local/bin/npu-smi ]] || { echo "npu-smi is unavailable" >&2; exit 1; }

case "$variant" in
  baseline)
    variant_args=()
    ;;
  streaming)
    variant_args=(
      --set conditional_is.stream_candidate_rollouts=true
      --set conditional_is.rollout_stream_candidate_batch_size=5
      --set conditional_is.rollout_stream_max_batches=2
    )
    ;;
  bounded)
    # Requests are candidate-major, so 15 requests admit five C15/R3 subtrees.
    variant_args=(--set conditional_is.rollout_submission_batch_size=15)
    ;;
  *)
    echo "unknown variant: $variant" >&2
    exit 2
    ;;
esac

IFS=, read -r -a device_ids <<<"$devices"
device_args=()
logical_ids=()
for logical in "${!device_ids[@]}"; do
  id=${device_ids[$logical]}
  [[ "$id" =~ ^[0-7]$ ]] || { echo "invalid device: $id" >&2; exit 2; }
  device_args+=(--device "/dev/davinci$id:/dev/davinci$logical")
  logical_ids+=("$logical")
done
logical_devices=$(IFS=,; echo "${logical_ids[*]}")

if [[ -e "$output" ]] && ! rm -rf "$output" 2>/dev/null; then
  output_parent=$(dirname "$output")
  output_name=$(basename "$output")
  docker run --rm \
    --entrypoint /bin/bash \
    -v "$output_parent":/cleanup \
    "$image" -lc 'rm -rf -- "/cleanup/$1"' _ "$output_name"
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
  -e ASCEND_RT_VISIBLE_DEVICES="$logical_devices" \
  "${device_args[@]}" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
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
      --tensor-parallel-size 2 \
      --pipeline-parallel-size 1 \
      --profiler none \
      --limit 64 \
      --categorical-root /categorical \
      --set generation.max_new_tokens=512 \
      --set vllm.max_model_len=65536 \
      --set vllm.max_num_seqs=256 \
      --set vllm.max_num_batched_tokens=32768 \
      --set vllm.gpu_memory_utilization=0.90 \
      --set vllm.engine_kwargs.max_num_partial_prefills=1 \
      --set vllm.engine_kwargs.max_long_partial_prefills=1 \
      ${variant_args[*]} \
      --env VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE=1 \
      --env ASCEND_CUSTOM_OPP_PATH=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend \
      --env LD_PRELOAD=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so \
      --output-directory /artifacts \
      --devices $logical_devices \
      --port $port \
      --routing round_robin \
      --workers 32 \
      --candidate-count 15 \
      --rollout-count 3 \
      --block-size 128 \
      > /artifacts/launcher.log 2>&1
  "

echo "$container"
