from __future__ import annotations
from datetime import datetime, timezone
from html import escape
from .data import suggested_relationships


def _plan_summary(result: dict) -> str:
    nodes = result["metrics"].get("nodes", [])
    if not nodes:
        return ""
    lines = []
    for node in nodes:
        label = node["node"]
        if node.get("join_type"):
            label += f" ({node['join_type']})"
        if node.get("relation"):
            label += f" on {node['relation']}"
        lines.append("  " * node.get("depth", 0) + f"{label} · {node.get('rows', 0)} actual rows")
    return "<pre>" + escape("\n".join(lines)) + "</pre>"


def render_report(dataset: dict, results: list[dict], suggestion=None, ai_result=None, *,
                  workspace: dict[str, dict] | None = None, ai_baseline=None) -> str:
    sorted_results = sorted(results, key=lambda r: r["median_ms"])
    fastest = sorted_results[0]
    slowest = sorted_results[-1]
    speedup = (slowest["median_ms"] / fastest["median_ms"]) if fastest["median_ms"] > 0 else None
    rows = "".join(
        f"<tr><td>{escape(r['name'])}</td><td>{r['median_ms']:.3f}</td>"
        f"<td>{r['min_ms']:.3f}</td><td>{r['max_ms']:.3f}</td>"
        f"<td>{r['metrics']['planning_ms']:.3f}</td>"
        f"<td>{r['metrics']['buffer_hits']}</td><td>{r['metrics']['buffer_reads']}</td></tr>"
        for r in results
    )
    query_blocks = "".join(f"<h3>{escape(r['name'])}</h3><pre>{escape(r['sql'])}</pre>" + _plan_summary(r)
                           for r in results)
    ai_html = ""
    if suggestion is not None and suggestion.no_safe_optimization:
        ai_html = ("<h2>No safe optimization recommended</h2><p>" +
                   escape(suggestion.explanation) + "</p><p>" +
                   escape(suggestion.expected_mechanism) + "</p><p>Limitations: " +
                   escape(suggestion.cautions) + "</p>")
    elif suggestion is not None:
        ai_html = ("<h2>AI optimization proposal</h2><pre>" + escape(suggestion.optimized_sql) +
                   "</pre><p>" + escape(suggestion.explanation) + "</p><p>Expected mechanism: " +
                   escape(suggestion.expected_mechanism) + "</p><p>Limitations: " +
                   escape(suggestion.cautions) + "</p>")
        if ai_result is not None:
            ai_html += f"<p>Measured AI candidate median: {ai_result['median_ms']:.3f} ms</p>"
            ai_html += _plan_summary(ai_result)
            if ai_baseline is not None:
                ai_html += f"<p>Retested original median: {ai_baseline['median_ms']:.3f} ms</p>"
        else:
            ai_html += "<p>AI suggestion has not been benchmarked.</p>"
    label = f"{speedup:.2f}x" if speedup is not None else "undefined (zero timing)"
    friendly_name = dataset.get("friendly_name")
    table_html = f"<p>SQL table <code>{escape(friendly_name)}</code></p>" if friendly_name else ""
    dataset_html = (
        f"<p>Dataset: {escape(dataset['display_name'])} · {dataset['rows']:,} rows · "
        f"{len(dataset['columns'])} columns</p>{table_html}"
    )
    if workspace is not None:
        entries = "".join(
            f"<li><code>{escape(name)}</code> · {escape(item['display_name'])} · "
            f"{item['rows']:,} rows · {len(item['columns'])} columns</li>"
            for name, item in workspace.items()
        )
        dataset_html = f"<h2>Workspace tables</h2><ul>{entries}</ul>"
        relationships = suggested_relationships(workspace)
        if relationships:
            keys = "".join(
                f"<li><code>{escape(key['left_table'])}.{escape(key['column'])}</code> ↔ "
                f"<code>{escape(key['right_table'])}.{escape(key['column'])}</code> ({escape(key['type'])})</li>"
                for key in relationships
            )
            dataset_html += f"<h3>Possible join keys</h3><p>Matching names and types. Relationships are unverified.</p><ul>{keys}</ul>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>SQL Query Optimization Report</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:960px;margin:40px auto;padding:0 24px;color:#0f172a}}
h1{{color:#4338ca}}table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid #cbd5e1;text-align:left;padding:11px}}
th{{background:#f1f5f9}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#111827;color:#e2e8f0;padding:20px;border-radius:10px}}
.note{{background:#fefce8;padding:16px;border-left:4px solid #eab308}}.small{{color:#64748b}}
</style></head><body><h1>SQL Query Optimization Report</h1>
<p class="small">Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
{dataset_html}
<h2>Performance summary</h2><p>Fastest observed: <strong>{escape(fastest['name'])}</strong> — {fastest['median_ms']:.3f} ms median.</p>
<p>Fastest vs. slowest: <strong>{label}</strong> (using PostgreSQL execution time).</p>
<div class="note">Result equivalence: UNVERIFIED. Compare query results before adopting a faster rewrite.</div>
<h2>Comparison</h2><table><thead><tr><th>Query</th><th>Median ms</th><th>Min ms</th><th>Max ms</th><th>Planning ms</th><th>Buffer hits</th><th>Buffer reads</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Submitted SQL and execution plans</h2>{query_blocks}{ai_html}
<h2>Benchmark method</h2><p>Each query is measured through PostgreSQL EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON). Queries alternate execution order across repetitions, with a five-second statement timeout. Medians are used for ranking. Timing excludes browser rendering and network transmission.</p>
</body></html>"""
