# End-to-end smoke runbook

Install this repository and the pinned agent in the runtime environment:

```bash
python -m pip install -e '.[swebench]'
```

Set the model path and start one persistent TP2 service:

```bash
export CIS_MODEL_PATH=/path/to/Qwen3-Coder-30B-A3B-Instruct
conditional-is-serve \
  --config configs/swebench/conditional_is_smoke.toml \
  --host 0.0.0.0 \
  --port 8123
```

Run one SWE-bench Verified task using mini-SWE-agent's standard Docker runner
and the Conditional IS model overlay:

```bash
mini-extra swebench-single \
  -c swebench \
  -c configs/mini_swe_agent/conditional_is.yaml \
  --subset verified \
  --instance django__django-11099
```

Before any quality or performance sweep, verify:

- the Qwen chat template receives the bash tool schema;
- exactly the selected CIS completion reaches Docker;
- tool observations appear in the next query's messages;
- EOS, output-limit, timeout, malformed tool-call, and service errors are saved;
- `artifacts/traces/model_calls.jsonl` contains prompt lengths and CIS diagnostics.

The exact mini-SWE-agent CLI flags can change across releases. This repository
pins v2.4.6; use `mini-extra swebench-single --help` on the execution host to
confirm the local invocation before launching a batch.
