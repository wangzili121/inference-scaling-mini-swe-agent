# DeepSeek-V4-Flash-0731 W8A8 + ordinary Conditional IS

This package is an **offline, single-node baseline** for the internal host. It
does not enable the CIS tree scheduler, native KV fork, Forest Attention, DSpark,
MTP, or a Qwen-specific sampler patch. The source remains based on the imported
`04492fe` snapshot plus local infrastructure work; the reward behavior was
reviewed against Chang's `origin/main@1c3f65d` and ports its thinking-scope
Consilience fallback semantics without replacing the working DSV4 compatibility
layer wholesale.
Upstream `conditional_is.py` itself has no diff between those two SHAs; the
relevant mainline change is reward/output handling plus general model loading.

## What is included

- One persistent AsyncLLM engine, ordinary `C8/R3/B128` Conditional IS inside
  each `/v1/query` or OpenAI-compatible `/v1/chat/completions` request.
- Ascend W8A8 quantization, DeepSeek V4 tokenizer and the vLLM DSML parser.
- Expert parallelism, hybrid KV cache manager, chunked prefill, APC, full-decode
  NPU Graph, DSA context parallelism, FlashComm1, shared-expert multistream
  overlap, CPU binding, optional jemalloc and generation-time reward statistics.
- Consilience defaults to top-5, 20% windows, 5% initial skip, initial penalty 3,
  score temperature 1 and reward temperature 2. Complete recognized thinking is
  scored; absent, malformed or truncated thinking falls back to the full output
  and is counted in `diagnostics.reward_execution`.
- Fixed-workload burst benchmark with raw per-query timing and diagnostics.
- Standard OpenAI model discovery/chat endpoints so the image's unmodified
  `vllm bench serve` can generate capacity, Poisson and bursty traffic.

The upstream [v0.26.0rc1 0731 deployment guide](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/tutorials/models/DeepSeek-V4-Flash.html)
uses `--quantization ascend`, `--tokenizer-mode deepseek_v4`, expert parallelism
and a DeepSeek-specific parser. It recommends multiple deployment topologies;
our first package uses eight visible NPUs as `TP8/DP1`. `MNS=32`, `MBT=8192`,
memory utilization `0.90` and block size `128` follow the official eight-card
starting point. `max_model_len=32768` remains the already validated A2 startup
cap: this specific run reported only 44,688 KV tokens of runtime capacity, so
blindly changing it to 133,120 would likely make startup fail. Raise the cap only
after the service reports enough KV capacity; benchmark prompt length is a
separate workload property.

The official example disables APC for its generic path. CIS has repeated
candidate/rollout prefixes, so this package enables APC and initially retains
the official `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096`. This is a
hypothesis to verify with APC-on/off counters, not an assumed result.

## Internal-host sequence

The repo, complete 0731 model snapshot, image and drivers must already be on the
host. The launcher never pulls an image or weight and never auto-selects cards;
the one-click launcher verifies that every selected card is healthy, has no
reported process and remains below the configured idle-HBM threshold before it
sets the launch confirmation internally.

For the first baseline capacity run, review `.env` once and execute:

```bash
cd /path/to/inference-scaling-mini-swe-agent/deploy/dsv4_flash
bash one_click.sh baseline
```

That one command replaces only the named prior test container, checks and locks
the selected cards, runs the image/model/runtime preflight, starts the service,
executes a complete CIS smoke, generates 16 distinct mixed 4K/8K/16K coding
prompts, runs top-level CIS concurrency `2/4/8/16`, and preserves the effective
environment, raw benchmark files, service diagnostics, container inspect data
and full logs below `ARTIFACT_DIR/runs/<timestamp>-baseline/`. The service stays
running after the benchmark. The capacity curve is rendered directly as
`capacity-summary.{json,csv,md}`. Pass a custom vLLM JSONL workload as the
second argument when available.

The optional pressure-aware tree comparison imports the exact source snapshot
from [Muyuan PR 12](https://gitcode.com/openeuler/muyuan/pull/12), commit
`1a94a4e`. It keeps vLLM as the physical scheduler, uses CIS tree metadata for
token-budget admission and job priority only after pressure activates, and is
not enabled in the baseline:

```bash
bash one_click.sh pressure_tree
```

Run the two variants with the same workload before attributing a gain. PR 12
reports Qwen/Ascend results, but explicitly leaves DeepSeek-V4-Flash eight-card
performance unverified; this package is the first DSV4 A/B rather than a claim
that the plugin must improve it.

To run that complete same-workload comparison unattended:

```bash
bash one_click_ab.sh
```

The individual commands remain available for debugging:

```bash
cd /path/to/inference-scaling-mini-swe-agent/deploy/dsv4_flash
cp .env.example .env
# Set MODEL_DIR, ARTIFACT_DIR, IMAGE and DEVICES in .env. On the current
# internal host MODEL_DIR=/workspace/models/DeepSeek-V4-Flash-0731-w8a8.
bash run.sh check
# Review npu-smi; only then set CONFIRM_DEVICES_FREE=yes in .env.
# launch performs start + a 256-token OpenAI/CIS smoke in one command. This is
# long enough to exercise candidate -> rollout -> reward -> resample.
bash run.sh launch
bash run.sh status
python3 smoke.py --endpoint http://127.0.0.1:8123  # optional tool-call smoke
```

`check` verifies local image availability, selected device nodes, the container's
AscendCL Python module (`acl.rt`), complete model shard index, vLLM engine
arguments, DeepSeek V4 prompt encoding and DSML bash parsing **before model
loading**. It then performs a real single-NPU tensor round trip and a TP-sized
HCCL all-reduce. The local image ID and preflight output are written under
`ARTIFACT_DIR`, including `npu-device-probe.json` and `hccl-probe.log`. This is a
metadata hash plus shard-existence/size check, not a full SHA256 of every large
weight file. If either device probe fails, the problem is below Conditional IS
and model loading; stop there and send the small probe log. Do not silently
switch the internal image or apply the v0.18 patch.
The Python-library service explicitly uses
`VLLM_WORKER_MULTIPROC_METHOD=spawn`: unlike `vllm serve`, it constructs
`AsyncLLM` from a long-lived service process, so the vLLM library default of
forking accelerator workers is unsafe. Override this only for a controlled
diagnostic.
Jemalloc preloading is disabled by default for this embedded service. On some
Ascend stacks any `LD_PRELOAD` makes FunctionLoader prefer `RTLD_DEFAULT`, which
can make `aclInit` resolve an incompatible runtime symbol and fail before the
SoC version is detected. Treat jemalloc as a later performance A/B only after
the NPU and HCCL probes pass.
`start` launches a container, waits up to 30 minutes for `/healthz`, and keeps a
failed container for log inspection. On startup failure it saves the full
timestamped log to `ARTIFACT_DIR/container-startup.full.log`; `status` shows
the last 80 log lines.
The service `/healthz` becomes available only after AsyncLLM/model construction.
`/v1/models` and `/v1/chat/completions` are compatibility endpoints. Sampling
temperature and top-p remain fixed by the CIS service; incompatible client
values are rejected instead of silently changing the algorithm. `max_tokens`
may vary per request up to `MAX_COMPLETION_TOKENS`.

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

For another representative algorithm, keep the workload unchanged and set
`PRESS_CANDIDATE_COUNT`/`PRESS_ROLLOUT_COUNT` before `press.sh`. The benchmark
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

## Standard vLLM Bench workloads

`vllm bench` custom input is JSONL. Each row contains a distinct coding prompt;
`output_tokens` is optional because the wrapper also accepts
`BENCH_OUTPUT_LEN`:

```json
{"prompt":"Review this Python implementation for a concurrency bug: ...","output_tokens":512}
{"prompt":"Implement the missing method in this repository excerpt: ...","output_tokens":1024}
```

Place the file below `ARTIFACT_DIR`, then find the capacity curve:

```bash
cd deploy/dsv4_flash
bash benchmark_suite.sh "$ARTIFACT_DIR/vllm-bench/coding-prompts.jsonl"
```

This runs top-level CIS concurrency `2/4/8/16` with `request-rate=inf`. After the
saturation jobs/s is known, run steady Poisson and bursty traffic at roughly
50%, 70% and 85% of it:

```bash
bash vllm_bench.sh steady "$ARTIFACT_DIR/vllm-bench/coding-prompts.jsonl" 16 RATE
BENCH_BURSTINESS=0.3 \
  bash vllm_bench.sh bursty "$ARTIFACT_DIR/vllm-bench/coding-prompts.jsonl" 16 RATE
```

The wrapper intentionally uses non-streaming E2E latency. Conditional IS cannot
emit the finally selected answer until candidate/rollout/reward/resample has
finished, so ordinary token-stream TTFT/TPOT would be misleading. The vLLM result
provides top-level jobs/s and E2E percentiles; service traces provide internal
candidate, rollout, reward, KV, preemption and forward-token-slot metrics.

Use standard vLLM Bench for public/synthetic prompts. Use `press.sh` for frozen
MiniAgent calls because those must preserve the complete message/tool history.
The production-oriented matrix should include 4-16K interactive prompts,
16-64K agent turns and 64-128K repository-level calls when the validated context
cap permits them, with both repeated-prefix and cold-prefix traffic.

## Automatic deployment and workload matrix

`autotune_matrix.sh` compares ordinary AR, flat Conditional IS and the CIS
pressure/tree scheduler on the same persistent model engine and frozen prompt
files. It first performs successive deployment selection, then runs the final
load matrix and renders a standalone Chinese HTML report.

```bash
# Fast plumbing check: two deployment candidates, medium-context capacity only.
bash autotune_matrix.sh quick

# Recommended: MNS/MBT selection, 4K-32K contexts, 512/1024 outputs,
# saturation, steady 70%-of-capacity traffic and bursty traffic.
bash autotune_matrix.sh standard

# Wider MNS/MBT and concurrency boundaries. This can take many hours.
bash autotune_matrix.sh full
```

The default deployment candidates are paired instead of forming an expensive
Cartesian product:

```text
quick:    32/8192, 64/16384
standard: 32/8192, 64/16384, 128/32768
full:     32/8192, 64/16384, 128/32768, 256/65536
```

Override them without editing the script. An optional third field sets memory
utilization:

```bash
AUTOTUNE_ENGINE_PROFILES="32:8192:0.90 64:16384:0.94 128:32768:0.94" \
  bash autotune_matrix.sh standard
```

Use `AUTOTUNE_MAX_MODEL_LEN=131072` after the 64K suite is stable. Real custom
JSONL workloads can replace the generated prompts:

```bash
AUTOTUNE_MEDIUM_WORKLOAD=/path/to/frozen-medium.jsonl \
AUTOTUNE_LONG_WORKLOAD=/path/to/frozen-long.jsonl \
  bash autotune_matrix.sh standard
```

The report is written to:

```text
$ARTIFACT_DIR/autotune/<timestamp>-<suite>/autotune-report.html
```

That directory also contains JSON/CSV summaries, raw vLLM results, stdout,
pre/post service diagnostics, workload manifests, effective environments and
container logs. Synthetic prompts are for capacity selection; repeat the
winning configuration with frozen real MiniAgent calls before final claims.

The service exposes two benchmark-compatible routes:

```text
/v1/chat/completions         one complete Conditional IS job
/v1/direct/chat/completions  one ordinary AR generation on the same engine
```

Set `BENCH_API_MODE=direct|cis` when invoking `vllm_bench.sh` directly.
