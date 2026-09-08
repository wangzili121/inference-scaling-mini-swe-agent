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
conditional-is-swebench \
  --instance-id django__django-11099 \
  --output artifacts/swebench/smoke-1
```

Before any quality or performance sweep, verify:

- the Qwen chat template receives the bash tool schema;
- exactly the selected CIS completion reaches Docker;
- tool observations appear in the next query's messages;
- EOS, output-limit, timeout, malformed tool-call, and service errors are saved;
- `artifacts/traces/model_calls.jsonl` contains prompt lengths and CIS diagnostics.

Then run the first three deterministic Verified tasks with the same service:

```bash
conditional-is-swebench --count 3 --output artifacts/swebench/smoke-3
```

The launcher rejects any mini-SWE-agent version other than v2.4.6, fixes the
Verified `test` split, checks the Conditional IS service and Docker daemon, and
records its resolved command in the output directory. Use `--dry-run` to inspect
that manifest without contacting the service or dataset.
