#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ENV_FILE="${CIS_ENV_FILE:-$HERE/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  printf 'Missing %s; start from .env.example\n' "$ENV_FILE" >&2
  exit 2
fi
set -a
# The operator owns this file; shell syntax permits paths with spaces.
source "$ENV_FILE"
set +a

: "${MODEL_DIR:?set MODEL_DIR}"
: "${IMAGE:?set IMAGE}"
: "${DEVICES:?set DEVICES}"
: "${ARTIFACT_DIR:?set ARTIFACT_DIR}"
PORT="${PORT:-8123}"
HOST="${HOST:-127.0.0.1}"
CONTAINER_NAME="${CONTAINER_NAME:-cis-dsv4-0731}"
SHM_SIZE="${SHM_SIZE:-512g}"
TP="${TP:-8}"
DP="${DP:-1}"
SCHEDULER_VARIANT="${SCHEDULER_VARIANT:-baseline}"
if [[ "$SCHEDULER_VARIANT" != baseline && "$SCHEDULER_VARIANT" != pressure_tree ]]; then
  printf 'SCHEDULER_VARIANT must be baseline or pressure_tree\n' >&2
  exit 2
fi
if [[ "${ALLOW_NONSTANDARD_ALGORITHM:-no}" != yes ]]; then
  if [[ "${CANDIDATE_COUNT:-8}" != 8 || "${ROLLOUT_COUNT:-3}" != 3 || "${REWARD_KIND:-consilience}" != consilience ]]; then
    printf 'This baseline requires C8/R3 + Consilience. Update .env or set ALLOW_NONSTANDARD_ALGORITHM=yes for an explicit control run.\n' >&2
    exit 2
  fi
fi
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
ARTIFACT_DIR="$(mkdir -p "$ARTIFACT_DIR" && cd "$ARTIFACT_DIR" && pwd)"

IFS=',' read -r -a npu_ids <<< "$DEVICES"
if [[ ${#npu_ids[@]} -ne $((TP * DP)) ]]; then
  printf 'DEVICES count must equal TP*DP (%s*%s)\n' "$TP" "$DP" >&2
  exit 2
fi
declare -a docker_mounts docker_devices
for id in "${npu_ids[@]}"; do
  if [[ ! "$id" =~ ^[0-9]+$ || ! -c "/dev/davinci$id" ]]; then
    printf 'NPU device unavailable: /dev/davinci%s\n' "$id" >&2
    exit 2
  fi
  docker_devices+=(--device "/dev/davinci$id")
done
for node in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  [[ -c "$node" ]] && docker_devices+=(--device "$node")
done
for path in \
  /usr/local/dcmi \
  /usr/local/Ascend/driver/tools/hccn_tool \
  /usr/local/bin/npu-smi \
  /usr/local/Ascend/driver/lib64 \
  /usr/local/Ascend/driver/version.info \
  /etc/ascend_install.info \
  /etc/hccn.conf; do
  [[ -e "$path" ]] && docker_mounts+=(-v "$path:$path:ro")
done

common=(--network host --privileged=true --shm-size "$SHM_SIZE"
  -w /workspace
  "${docker_devices[@]}" "${docker_mounts[@]}"
  -v "$REPO:/workspace:ro"
  -v "$MODEL_DIR:/models/dsv4:ro"
  -v "$ARTIFACT_DIR:/artifacts"
  -e CIS_MODEL_PATH=/models/dsv4
  -e ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
  -e VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  -e OMP_PROC_BIND=false
  -e OMP_NUM_THREADS=10
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL="${PREFIX_CACHE_RETENTION_INTERVAL:-4096}"
  -e CIS_ENABLE_JEMALLOC="${ENABLE_JEMALLOC:-false}"
  -e HCCL_BUFFSIZE=1024
  -e TASK_QUEUE_ENABLE=1
  -e HCCL_OP_EXPANSION_MODE=AIV)

check() {
  docker image inspect "$IMAGE" --format '{{.Id}}' > "$ARTIFACT_DIR/image-id.txt"
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info
  elif [[ -x /usr/local/bin/npu-smi ]]; then
    /usr/local/bin/npu-smi info
  fi
  docker run --rm "${common[@]}" --entrypoint bash "$IMAGE" \
    /workspace/deploy/dsv4_flash/container_python.sh \
    -m inference_scaling.swe_agent.dsv4_preflight \
    --model-dir /models/dsv4 --runtime | tee "$ARTIFACT_DIR/runtime-preflight.json"
  docker run --rm "${common[@]}" --entrypoint bash "$IMAGE" \
    /workspace/deploy/dsv4_flash/container_python.sh \
    -m inference_scaling.swe_agent.ascend_device_probe \
    | tee "$ARTIFACT_DIR/npu-device-probe.json"
  docker run --rm "${common[@]}" --entrypoint bash "$IMAGE" \
    -lc 'exec bash /workspace/deploy/dsv4_flash/container_python.sh \
      -m torch.distributed.run --master-addr=127.0.0.1 --master-port=29501 \
      --nproc-per-node="$1" \
      -m inference_scaling.swe_agent.ascend_device_probe --distributed' \
    bash "$((TP * DP))" | tee "$ARTIFACT_DIR/hccl-probe.log"
  printf 'Inspect NPU occupancy above before setting CONFIRM_DEVICES_FREE=yes.\n'
}

start() {
  check
  if [[ "${CONFIRM_DEVICES_FREE:-no}" != yes ]]; then
    printf 'Set CONFIRM_DEVICES_FREE=yes only after checking all selected cards are idle.\n' >&2
    exit 2
  fi
  if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    printf 'Container %s already exists; refusing a duplicate launch.\n' "$CONTAINER_NAME" >&2
    exit 2
  fi
  if python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(1); rc=s.connect_ex(("127.0.0.1",int(sys.argv[1]))); s.close(); sys.exit(0 if rc else 1)' "$PORT"; then
    :
  else
    printf 'Port %s is already in use; refusing a duplicate service.\n' "$PORT" >&2
    exit 2
  fi
  local -a overrides=(
    --set "vllm.tensor_parallel_size=${TP}"
    --set "vllm.data_parallel_size=${DP}"
    --set "vllm.max_model_len=${MAX_MODEL_LEN:-32768}"
    --set "vllm.max_num_seqs=${MAX_NUM_SEQS:-32}"
    --set "vllm.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}"
    --set "vllm.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.90}"
    --set "vllm.enable_prefix_caching=${ENABLE_PREFIX_CACHING:-true}"
    --set "vllm.enforce_eager=${ENFORCE_EAGER:-false}"
    --set "conditional_is.candidate_count=${CANDIDATE_COUNT:-8}"
    --set "conditional_is.rollout_count=${ROLLOUT_COUNT:-3}"
    --set "conditional_is.block_size=${BLOCK_SIZE:-128}"
    --set "conditional_is.reward_temperature=${REWARD_TEMPERATURE:-2.0}"
    --set "reward.kind=\"${REWARD_KIND:-consilience}\""
    --set "generation.max_new_tokens=${MAX_NEW_TOKENS:-512}"
    --set "service.max_completion_tokens=${MAX_COMPLETION_TOKENS:-2048}"
  )
  if [[ -n "${CUDAGRAPH_CAPTURE_SIZES:-}" ]]; then
    if [[ ! "$CUDAGRAPH_CAPTURE_SIZES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
      printf 'CUDAGRAPH_CAPTURE_SIZES must be a comma-separated list of integers.\n' >&2
      exit 2
    fi
    overrides+=(--set "vllm.engine_kwargs.compilation_config={\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CUDAGRAPH_CAPTURE_SIZES}]}")
  fi
  if [[ "$SCHEDULER_VARIANT" == pressure_tree ]]; then
    overrides+=(--set 'conditional_is.active_step_admission="pressure_plugin"')
    overrides+=(--set "conditional_is.active_step_max_limit=${ACTIVE_STEP_MAX_LIMIT:-${MAX_NUM_SEQS:-32}}")
    overrides+=(--set "conditional_is.active_step_kv_capacity_fraction=${ACTIVE_STEP_KV_FRACTION:-0.80}")
    overrides+=(--set 'vllm.request_priority_policy="job_fifo"')
  else
    overrides+=(--set 'conditional_is.active_step_admission="fixed"')
    overrides+=(--set 'conditional_is.active_step_limit=0')
    overrides+=(--set 'vllm.request_priority_policy="none"')
  fi
  if [[ "${ENFORCE_EAGER:-false}" == true ]]; then
    overrides+=(--set 'vllm.engine_kwargs.compilation_config={"cudagraph_mode":"NONE"}')
    overrides+=(--set 'vllm.engine_kwargs.additional_config={"enable_cpu_binding":true,"enable_dsa_cp":true,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}')
  fi
  printf '%s\n' "${overrides[@]}" > "$ARTIFACT_DIR/launch-overrides.txt"
  if ! git -C "$REPO" rev-parse HEAD > "$ARTIFACT_DIR/repo-commit.txt" 2>/dev/null; then
    printf 'unversioned-copy\n' > "$ARTIFACT_DIR/repo-commit.txt"
  fi
  docker run -d "${common[@]}" --name "$CONTAINER_NAME" \
    --entrypoint bash "$IMAGE" \
    /workspace/deploy/dsv4_flash/container_python.sh \
    -m inference_scaling.swe_agent.server \
    --config /workspace/configs/dsv4_flash/conditional_is_0731.toml \
    --host "$HOST" --port "$PORT" "${overrides[@]}" \
    | tee "$ARTIFACT_DIR/container-id.txt"
  if ! command -v curl >/dev/null 2>&1; then
    printf 'Started. curl is unavailable; use %s status and smoke.py.\n' "$0"
    return
  fi
  local elapsed=0
  local limit="${WAIT_SECONDS:-1800}"
  local probe_host="$HOST"
  [[ "$probe_host" == 0.0.0.0 ]] && probe_host=127.0.0.1
  while ((elapsed < limit)); do
    if curl -fsS --max-time 3 "http://$probe_host:$PORT/healthz" >/dev/null 2>&1; then
      printf 'Conditional IS service ready at http://%s:%s\n' "$probe_host" "$PORT"
      return
    fi
    if [[ "$(docker inspect "$CONTAINER_NAME" --format '{{.State.Running}}')" != true ]]; then
      docker logs --timestamps "$CONTAINER_NAME" > "$ARTIFACT_DIR/container-startup.full.log" 2>&1
      docker logs --tail 100 "$CONTAINER_NAME" >&2
      printf 'Full startup log: %s/container-startup.full.log\n' "$ARTIFACT_DIR" >&2
      printf 'Model container exited before healthz.\n' >&2
      exit 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  docker logs --tail 100 "$CONTAINER_NAME" >&2
  docker logs --timestamps "$CONTAINER_NAME" > "$ARTIFACT_DIR/container-startup.full.log" 2>&1
  printf 'Full startup log: %s/container-startup.full.log\n' "$ARTIFACT_DIR" >&2
  printf 'Healthz did not become ready within %ss; container remains for inspection.\n' "$limit" >&2
  exit 1
}

verify() {
  local probe_host="$HOST"
  [[ "$probe_host" == 0.0.0.0 ]] && probe_host=127.0.0.1
  python3 "$HERE/openai_smoke.py" \
    --endpoint "http://$probe_host:$PORT" \
    --max-tokens "${SMOKE_MAX_TOKENS:-256}"
}

case "${1:-}" in
  check) check ;;
  start) start ;;
  launch)
    start
    verify
    ;;
  verify) verify ;;
  status)
    docker ps --filter "name=^/${CONTAINER_NAME}$"
    docker logs --tail 80 "$CONTAINER_NAME"
    ;;
  stop)
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
    ;;
  *) printf 'Usage: %s {check|start|launch|verify|status|stop}\n' "$0" >&2; exit 2 ;;
esac
