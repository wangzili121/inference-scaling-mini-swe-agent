#!/usr/bin/env python3
"""Render a standalone report for DSV4 direct/CIS/tree benchmark matrices."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


METRICS = (
    "completed",
    "failed",
    "duration",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_e2el_ms",
    "median_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "total_input_tokens",
    "total_output_tokens",
)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def backend_delta(run_dir: Path) -> dict[str, int | float]:
    before = load_json(run_dir / "service-diagnostics.before.json") or {}
    after = load_json(run_dir / "service-diagnostics.after.json") or {}
    left = before.get("backend", {})
    right = after.get("backend", {})
    if not isinstance(left, dict) or not isinstance(right, dict):
        return {}
    return {
        key: right[key] - left.get(key, 0)
        for key in right
        if isinstance(right[key], (int, float))
        and isinstance(left.get(key, 0), (int, float))
    }


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(root.rglob("run-metadata.json")):
        metadata = load_json(metadata_path)
        if metadata is None:
            continue
        run_dir = metadata_path.parent
        candidates = []
        for path in run_dir.glob("*.json"):
            if path.name in {
                "run-metadata.json",
                "service-diagnostics.before.json",
                "service-diagnostics.after.json",
            }:
                continue
            value = load_json(path)
            if value is not None and "request_throughput" in value:
                candidates.append((path, value))
        if not candidates:
            continue
        path, result = candidates[0]
        row = dict(metadata)
        row.update({metric: result.get(metric) for metric in METRICS})
        row["source"] = str(path.relative_to(root))
        delta = backend_delta(run_dir)
        for key in (
            "engine_requests",
            "generated_tokens",
            "prefill_tokens",
            "shared_prefill_tokens_saved",
            "generation_forward_token_slots",
            "num_preemptions",
            "maximum_in_flight_requests",
        ):
            row[f"backend_{key}"] = delta.get(key)
        rows.append(row)
    for path in sorted(root.rglob("failure.json")):
        value = load_json(path)
        if value is None:
            continue
        rows.append(
            {
                "label": path.parent.name,
                "workload_profile": "deployment",
                "arrival": "launch",
                "scheduler_variant": value.get("variant"),
                "max_num_seqs": value.get("mns"),
                "max_num_batched_tokens": value.get("mbt"),
                "gpu_memory_utilization": value.get("memory"),
                "completed": 0,
                "failed": 1,
                "source": str(path.relative_to(root)),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("workload_profile")),
            str(row.get("arrival")),
            int(row.get("max_concurrency") or 0),
            str(row.get("label")),
        )
    )
    return rows


def number(value: Any, digits: int = 2) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value:,.{digits}f}"


def percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:+.1f}%"


def ratio(a: Any, b: Any, *, lower_better: bool = False) -> float | None:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not b:
        return None
    return (b / a - 1) if lower_better else (a / b - 1)


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("workload_profile"),
            row.get("arrival"),
            row.get("max_concurrency"),
            row.get("output_tokens"),
            row.get("max_num_seqs"),
            row.get("max_num_batched_tokens"),
        )
        mode = "direct" if row.get("api_mode") == "direct" else str(
            row.get("scheduler_variant")
        )
        groups.setdefault(key, {})[mode] = row
    comparisons = []
    for key, modes in groups.items():
        baseline = modes.get("baseline")
        if baseline is None:
            continue
        item = {"key": key, "baseline": baseline, "direct": modes.get("direct")}
        item["tree"] = modes.get("pressure_tree")
        comparisons.append(item)
    return comparisons


def bar_chart(rows: list[dict[str, Any]], metric: str, title: str) -> str:
    usable = [row for row in rows if isinstance(row.get(metric), (int, float))]
    if not usable:
        return ""
    usable = usable[-24:]
    maximum = max(float(row[metric]) for row in usable) or 1.0
    bars = []
    for index, row in enumerate(usable):
        y = index * 29
        width = max(1.0, 520 * float(row[metric]) / maximum)
        label = f"{row.get('workload_profile')} · {row.get('label')} · c{row.get('max_concurrency')}"
        bars.append(
            f'<text x="0" y="{y + 15}" class="axis">{html.escape(label[:52])}</text>'
            f'<rect x="330" y="{y}" width="{width:.1f}" height="19" rx="2" />'
            f'<text x="{338 + width:.1f}" y="{y + 15}" class="value">'
            f'{html.escape(number(row[metric], 4 if metric == "request_throughput" else 1))}</text>'
        )
    height = len(usable) * 29 + 10
    return (
        f'<section><h2>{html.escape(title)}</h2><svg class="chart" viewBox="0 0 940 {height}" '
        f'role="img">{"".join(bars)}</svg></section>'
    )


def render(root: Path, rows: list[dict[str, Any]]) -> str:
    valid = [row for row in rows if not row.get("failed")]
    best = max(valid, key=lambda row: row.get("request_throughput") or 0, default=None)
    selected = load_json(root / "best-deployment.json") or {}
    comparisons = comparison_rows(rows)
    summary_cards = [
        ("有效实验", str(len(valid))),
        ("失败实验", str(len(rows) - len(valid))),
        (
            "最高完整请求吞吐",
            "-" if best is None else f"{number(best.get('request_throughput'), 4)} jobs/s",
        ),
        (
            "推荐部署",
            (
                "尚未选出"
                if not selected
                else f"MNS {selected.get('max_num_seqs')} / MBT {selected.get('max_num_batched_tokens')}"
            ),
        ),
    ]
    cards = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in summary_cards
    )
    table_rows = []
    for row in rows:
        status = "失败" if row.get("failed") else "有效"
        table_rows.append(
            "<tr>"
            f'<td><span class="status {"bad" if status == "失败" else "ok"}">{status}</span></td>'
            f"<td>{html.escape(str(row.get('workload_profile')))}</td>"
            f"<td>{html.escape(str(row.get('label')))}</td>"
            f"<td>{html.escape(str(row.get('arrival')))}</td>"
            f"<td>{row.get('max_num_seqs')}</td><td>{row.get('max_num_batched_tokens')}</td>"
            f"<td>{row.get('max_concurrency')}</td>"
            f"<td>{number(row.get('request_throughput'), 4)}</td>"
            f"<td>{number(row.get('mean_e2el_ms'), 0)}</td>"
            f"<td>{number(row.get('p95_e2el_ms'), 0)}</td>"
            f"<td>{number(row.get('backend_generation_forward_token_slots'), 0)}</td>"
            f"<td>{number(row.get('backend_num_preemptions'), 0)}</td>"
            f'<td><code>{html.escape(str(row.get("source")))}</code></td>'
            "</tr>"
        )
    comparison_html = []
    for item in comparisons:
        baseline = item["baseline"]
        direct = item["direct"]
        tree = item["tree"]
        direct_slowdown = None
        if direct:
            direct_slowdown = ratio(
                baseline.get("mean_e2el_ms"), direct.get("mean_e2el_ms")
            )
        tree_gain = None
        tree_p95 = None
        if tree:
            tree_gain = ratio(
                tree.get("request_throughput"), baseline.get("request_throughput")
            )
            tree_p95 = ratio(
                tree.get("p95_e2el_ms"), baseline.get("p95_e2el_ms"), lower_better=True
            )
        comparison_html.append(
            "<tr>"
            f"<td>{html.escape(str(item['key'][0]))}</td>"
            f"<td>{html.escape(str(item['key'][1]))}</td>"
            f"<td>{item['key'][2]}</td>"
            f"<td>{percent(direct_slowdown)}</td>"
            f"<td>{percent(tree_gain)}</td>"
            f"<td>{percent(tree_p95)}</td>"
            "</tr>"
        )
    warning = (
        "<p class=callout>报告中的 jobs/s 表示完整 API 请求吞吐。CIS 会在一次请求内部产生大量 "
        "candidate/rollout，必须结合 forward-token-slots 与普通 AR 对照阅读。合成 workload "
        "用于容量调优，最终结论仍应在冻结的真实编码轨迹上复验。</p>"
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DSV4 Conditional IS 自动调优报告</title>
<style>
:root{{--ink:#16191d;--muted:#68707a;--line:#d8dde3;--paper:#f5f7f9;--panel:#fff;--accent:#146c94;--good:#137333;--bad:#b3261e}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}}
header{{background:#17232d;color:#fff;padding:34px max(24px,calc((100vw - 1440px)/2))}} h1{{font-size:30px;margin:0 0 6px}} header p{{margin:0;color:#cbd7df}}
main{{max-width:1440px;margin:auto;padding:24px}} h2{{font-size:19px;margin:0 0 14px}} section{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:20px;margin:0 0 20px;overflow:auto}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px;margin-bottom:20px}} .card{{background:#fff;border:1px solid var(--line);border-top:3px solid var(--accent);padding:14px 16px;border-radius:4px}} .card span{{display:block;color:var(--muted)}} .card strong{{display:block;font-size:20px;margin-top:4px;overflow-wrap:anywhere}}
.callout{{border-left:4px solid #d98e04;background:#fff8e6;padding:12px 14px;margin:0 0 20px}}
table{{border-collapse:collapse;width:100%;white-space:nowrap}} th,td{{border-bottom:1px solid var(--line);padding:9px 10px;text-align:right}} th{{position:sticky;top:0;background:#eef2f5;color:#414850}} th:nth-child(-n+4),td:nth-child(-n+4),td:last-child{{text-align:left}}
.status{{font-weight:600}} .ok{{color:var(--good)}} .bad{{color:var(--bad)}} code{{font-size:12px;color:#4d5964}} .chart{{width:100%;min-width:900px}} .chart rect{{fill:var(--accent)}} .chart .axis{{font-size:11px;fill:#46515a}} .chart .value{{font-size:11px;font-weight:600;fill:#18232c}}
@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}} main{{padding:12px}}}}
</style></head><body>
<header><h1>DSV4 Conditional IS 自动调优报告</h1><p>普通 AR、原始 CIS 与树信息调度的同模型同 workload 对照</p></header>
<main><div class="cards">{cards}</div>{warning}
{bar_chart(rows, 'request_throughput', '完整请求吞吐 jobs/s')}
{bar_chart(rows, 'p95_e2el_ms', '端到端 P95 延迟 ms（越短越好）')}
<section><h2>模式对照</h2><table><thead><tr><th>workload</th><th>到达</th><th>并发</th><th>CIS mean 相对 AR</th><th>Tree 吞吐相对 CIS</th><th>Tree P95 改善</th></tr></thead><tbody>{''.join(comparison_html) or '<tr><td colspan=6>尚无完整三模式对照</td></tr>'}</tbody></table></section>
<section><h2>全部实验</h2><table><thead><tr><th>状态</th><th>workload</th><th>模式</th><th>到达</th><th>MNS</th><th>MBT</th><th>并发</th><th>jobs/s</th><th>mean ms</th><th>P95 ms</th><th>forward slots</th><th>preempt</th><th>原始结果</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = collect(args.root)
    if not rows:
        raise SystemExit(f"no benchmark runs found below {args.root}")
    output = args.output or args.root / "autotune-report.html"
    output.write_text(render(args.root, rows), encoding="utf-8")
    (args.root / "autotune-summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = sorted({key for row in rows for key in row})
    with (args.root / "autotune-summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
