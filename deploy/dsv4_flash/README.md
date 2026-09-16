# DeepSeek-V4-Flash-0731 W8A8 + ordinary Conditional IS

This package is an **offline, single-node baseline** for the internal host. It
does not enable the CIS tree scheduler, native KV fork, Forest Attention, DSpark,
MTP, or a Qwen-specific sampler patch. Do not interpret its default scheduler
numbers as tuned for DeepSeek-V4-Flash.

## What is included

- One persistent AsyncLLM engine, ordinary Conditional IS inside each `/v1/query`.
- Ascend W8A8 quantization, DeepSeek V4 tokenizer and the vLLM DSML parser.
- Expert parallelism, hybrid KV cache manager, chunked prefill, APC, full-decode
  NPU Graph and generation-time reward statistics. `ENFORCE_EAGER=true` disables
  graph for an initial debugging A/B. `ENABLE_PREFIX_CACHING=false` provides an
  APC-off control if the first model load fails.
- Fixed-workload burst benchmark with raw per-query timing and diagnostics.

The upstream [v0.26.0rc1 0731 deployment guide](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/tutorials/models/DeepSeek-V4-Flash.html)
uses `--quantization ascend`, `--tokenizer-mode deepseek_v4`, expert parallelism
and a DeepSeek-specific parser. It recommends multiple deployment topologies;
our first package uses eight visible NPUs as `TP8/DP1`. `max_model_len=32768`,
`MNS=32` and `MBT=8192` are conservative startup values for coding calls, **not**
the guide's 1M-context throughput configuration or a verified CIS optimum.

## Internal-host sequence

The repo, complete 0731 model snapshot, image and drivers must already be on the
host. The launcher never pulls an image or weight and never auto-selects cards;
the operator must confirm that the chosen devices are idle.

```bash
cd /path/to/inference-scaling-mini-swe-agent/deploy/dsv4_flash
cp .env.example .env
# Set MODEL_DIR, ARTIFACT_DIR, IMAGE and DEVICES in .env. On the current
# internal host MODEL_DIR=/workspace/models/DeepSeek-V4-Flash-0731-w8a8.
bash run.sh check
# Review npu-smi; only then set CONFIRM_DEVICES_FREE=yes in .env.
bash run.sh start
bash run.sh status
python3 smoke.py --endpoint http://127.0.0.1:8123
```

`check` verifies local image availability, selected device nodes, the container's
AscendCL Python module (`acl.rt`), complete model
shard index, vLLM engine arguments, DeepSeek V4 prompt encoding and DSML bash
parsing **before model loading**. The local image ID and preflight JSON are written
under `ARTIFACT_DIR`. This is a metadata hash plus shard-existence/size check,
not a full SHA256 of every large weight file. If the runtime lacks an expected
API, stop there and send the
preflight error; don't silently switch the internal image or apply the v0.18 patch.
`start` launches a container, waits up to 30 minutes for `/healthz`, and keeps a
failed container for log inspection. On startup failure it saves the full
timestamped log to `ARTIFACT_DIR/container-startup.full.log`; `status` shows
the last 80 log lines.
The service `/healthz` becomes available only after AsyncLLM/model construction.

The first smoke asks for `bash echo cis-ready`. A completed response without an
action is a **pipeline diagnostic**, not an accuracy verdict. Inspect raw text in
the server trace if the model refuses the tool. Then validate one mini-SWE-agent
task with its Docker executor and subsequent tool observations before loading a
large burst.

On an internal host with mini-SWE-agent 2.4.6, SWE-bench and Docker installed,
run the existing adapter using the new 0731 overlay (this client runs outside
the model container):

```bash
cd /path/to/inference-scaling-mini-swe-agent
set -a; source deploy/dsv4_flash/.env; set +a
PYTHONPATH=src python3 -m inference_scaling.swe_agent.swebench \
  --config configs/dsv4_flash/mini_swe_agent.yaml \
  --endpoint http://127.0.0.1:8123 \
  --output "$ARTIFACT_DIR/agent-one" --count 1 --workers 1
```

If the client and model are on different hosts, edit the overlay endpoint and
use a reachable service bind address; do not expose the endpoint outside the
trusted internal network. Continue with three tasks only after confirming
tool-call, Docker execution and observation carry-through in the first trace.

## First pressure benchmark

Use **distinct real model-call snapshots**, not 64 copies of one prompt. Once the
mini-agent has produced traces, freeze them with the existing CLI (inside the
container or another Python 3.11+ environment with this package):

```bash
docker exec cis-dsv4-0731 bash /workspace/deploy/dsv4_flash/container_python.sh \
  -m inference_scaling.swe_agent.workload \
  --trace /artifacts/traces/dsv4_model_calls.jsonl \
  --output-directory /artifacts/workloads \
  --total 64 --seed 20260908
```

Pass `ARTIFACT_DIR/workloads/tune-32.jsonl` or a larger manifest as the second
argument to `press.sh`. It must reside under the mounted artifact directory.
Run the **same manifest and order** at increasing
whole-job concurrency until jobs/s improves by less than 3% while engine backlog
is nonempty; record the curve, not only the fastest point:

```bash
cd /path/to/inference-scaling-mini-swe-agent/deploy/dsv4_flash
set -a; source .env; set +a
bash press.sh 4 "$ARTIFACT_DIR/workloads/tune-32.jsonl"
bash press.sh 8 "$ARTIFACT_DIR/workloads/tune-32.jsonl"
bash press.sh 16 "$ARTIFACT_DIR/workloads/tune-32.jsonl"
bash press.sh 32 "$ARTIFACT_DIR/workloads/tune-32.jsonl"
```

For the heavier representative algorithm, keep the workload unchanged and set
`PRESS_CANDIDATE_COUNT=15 PRESS_ROLLOUT_COUNT=3` before `press.sh`. The benchmark
writes complete JSON under `ARTIFACT_DIR/pressure/`; it reports jobs/s and Job
latency distribution, and retains each query's diagnostics. Use `run.sh stop` only
for this named container when a restart is needed.

## Acceptance and next comparison

Before calling this a baseline, verify: 100% completed CIS queries on the fixed
manifest, correct DeepSeek tool-call/observation flow, no OOM, no surprising KV
preemption, and repeatable jobs/s and P95. Inspect APC hit and graph fallback in
the logs/profiler. Tune MNS, MBT, memory fraction and context cap **on this model
and workload**, preserving the launch manifest and image ID. Do not apply the
old Qwen `256/896` base/proposal numbers or v0.18 MRV1 categorical patch.

After that, tree-aware scheduling can be enabled as a separate versioned patch
or plugin. Compare with exactly the same image, model snapshot, TP/DP, workload,
seed, reward, C/R/B/L and non-tree engine options. Keep the same local model
snapshot; the metadata hash alone does not prove identical weight contents.
Save both raw benchmark files and traces; only then attribute any gain to the tree.
