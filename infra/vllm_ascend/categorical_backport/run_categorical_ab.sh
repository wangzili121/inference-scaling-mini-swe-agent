#!/usr/bin/env bash
# A/B: retained batched-transform sampler vs native AscendC categorical sampler.
# Runs on ONE free NPU, sequentially, same config, same seed.
# Usage: NPU_ID=0 bash run_categorical_ab.sh [retained|native|both]
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/disk/wangzili/inference_scaling_ar_npu_agent}"
NPU_ID="${NPU_ID:-1}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.18.0}"
CACHE_ROOT="${CACHE_ROOT:-/data/disk/${USER}/vllm-cache-agent}"
CONFIG="${CONFIG:-configs/gsm8k_3090_aligned_npu_full_decode.toml}"
REQUESTS="${REQUESTS:-96}"
WORKERS="${WORKERS:-96}"
ARRIVAL_QPS="${ARRIVAL_QPS:-0}"
ARM="${1:-both}"
CATEGORICAL_ROOT="${CATEGORICAL_ROOT:-/data/disk/wangzili/vllm-ascend-v018-categorical-clean}"

BASE_MBT="${BASE_MBT:-10240}"
EXTRA_ARGS="--base-memory-fraction 0.54 \
  --proposal-memory-fraction 0.36 \
  --base-max-num-seqs 40 \
  --proposal-max-num-seqs 96 \
  --base-max-num-batched-tokens ${BASE_MBT} \
  --proposal-max-num-batched-tokens 12288 \
  --base-score-priority 1"

run_arm() {
  local arm="$1"
  local tag="${TAG:-}"
  local output="results/agent/sampler_ab_${arm}${tag}_npu${NPU_ID}.json"
  local log="${output%.json}.log"
  local patch_mount=()
  local categorical_env=()
  if [[ "${arm}" == "retained" ]]; then
    patch_mount=(-v "${REPO_ROOT}/patches/batched_transform/sampler.py:/vllm-workspace/vllm-ascend/vllm_ascend/sample/sampler.py:ro")
  elif [[ "${arm}" == "native" ]]; then
    patch_mount=(
      -v "${CATEGORICAL_ROOT}/runtime/sampler.py:/vllm-workspace/vllm-ascend/vllm_ascend/sample/sampler.py:ro"
      -v "${CATEGORICAL_ROOT}/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so:/vllm-workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so:ro"
      -v "${CATEGORICAL_ROOT}/vllm_ascend/libvllm_ascend_kernels.so:/vllm-workspace/vllm-ascend/vllm_ascend/libvllm_ascend_kernels.so:ro"
      -v "${CATEGORICAL_ROOT}/vllm_ascend/_cann_ops_custom:/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom:ro"
    )
    categorical_env=(
      -e VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE=1
      -e LD_LIBRARY_PATH=/vllm-workspace/vllm-ascend/vllm_ascend:/vllm-workspace/vllm-ascend/vllm_ascend/lib:/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64
    )
  elif [[ "${arm}" == "p4" ]]; then
    patch_mount=(-v "${REPO_ROOT}/patches/p4/sampler.py:/vllm-workspace/vllm-ascend/vllm_ascend/sample/sampler.py:ro")
  elif [[ "${arm}" == "p1" || "${arm}" == "p2" ]]; then
    patch_mount=(
      -v "${REPO_ROOT}/patches/${arm}/outputs.py:/vllm-workspace/vllm/vllm/v1/outputs.py:ro"
      -v "${REPO_ROOT}/patches/${arm}/gpu_model_runner.py:/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py:ro"
    )
  fi
  mkdir -p "${REPO_ROOT}/$(dirname "${output}")" "${CACHE_ROOT}"
  docker run --rm --network host \
    --name "agent-sampler-ab-${arm}-npu${NPU_ID}" \
    --device "/dev/davinci${NPU_ID}:/dev/davinci0" \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v "${CACHE_ROOT}:/root/.cache/vllm" \
    "${patch_mount[@]}" \
    -v /data/disk/wangzili/inference_scaling_ar_npu_opt/models:/workspace/models:ro \
    -v "${REPO_ROOT}:/workspace" \
    -w /workspace \
    -e ASCEND_RT_VISIBLE_DEVICES=0 \
    -e PYTHONPATH=/workspace/src:/workspace \
    "${categorical_env[@]}" \
    --entrypoint bash \
    "${IMAGE}" -lc \
    "python experiments/arllm/run_small_proposal_pressure.py \
      --config '${CONFIG}' \
      --dtype float16 \
      --requests '${REQUESTS}' \
      --arrival-qps '${ARRIVAL_QPS}' \
      --workers '${WORKERS}' \
      ${EXTRA_ARGS} \
      --output '${output}' > '${log}' 2>&1"
  echo "arm=${arm} done: ${output}"
}

case "${ARM}" in
  retained) run_arm retained ;;
  native) run_arm native ;;
  p1) run_arm p1 ;;
  p2) run_arm p2 ;;
  p4) run_arm p4 ;;
  both) run_arm retained && run_arm native ;;
  *) echo "unknown arm: ${ARM}"; exit 1 ;;
esac
