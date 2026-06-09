"""HTML reporter — visual report for humans."""
from datetime import datetime
from pathlib import Path

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Eval Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
  h1 {{ color: #1a1a2e; }}
  .summary {{ display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }}
  .metric {{ background: white; border-radius: 8px; padding: 20px 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
  .metric .value {{ font-size: 2.5em; font-weight: bold; }}
  .metric .label {{ font-size: 0.9em; color: #666; margin-top: 4px; }}
  .pass {{ color: #27ae60; }}
  .fail {{ color: #e74c3c; }}
  .flaky {{ color: #f39c12; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  th {{ background: #1a1a2e; color: white; padding: 12px 16px; text-align: left; font-size: 0.85em; text-transform: uppercase; }}
  td {{ padding: 12px 16px; border-bottom: 1px solid #eee; font-size: 0.9em; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f9f9f9; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.78em; font-weight: bold; }}
  .badge-pass {{ background: #d5f5e3; color: #1e8449; }}
  .badge-fail {{ background: #fadbd8; color: #922b21; }}
  .badge-tag {{ background: #eaf2ff; color: #1a5276; margin-right: 4px; }}
  .variance {{ font-size: 0.82em; color: #555; margin-top: 4px; }}
  .stable {{ color: #27ae60; }}
  .flaky-v {{ color: #f39c12; }}
  .unstable {{ color: #e74c3c; }}
  .response {{ max-width: 400px; font-size: 0.85em; color: #444; word-break: break-word; }}
  .footer {{ margin-top: 30px; font-size: 0.8em; color: #999; text-align: center; }}
  .badge-tool {{ background: #e8d5f5; color: #6c3483; }}
  .badge-multi {{ background: #d5eaff; color: #1a5276; }}
  .tool-call {{ font-family: monospace; font-size: 0.82em; background: #f4f0fb; border-left: 3px solid #9b59b6; padding: 4px 8px; margin-top: 4px; border-radius: 3px; }}
  .turn-block {{ font-size: 0.82em; border-left: 3px solid #2980b9; padding: 4px 8px; margin-top: 4px; background: #f0f6ff; border-radius: 3px; }}
  .turn-label {{ color: #888; font-size: 0.78em; text-transform: uppercase; margin-bottom: 2px; }}
  .turn-pass {{ color: #27ae60; font-size: 0.75em; }}
  .turn-fail {{ color: #e74c3c; font-size: 0.75em; }}
</style>
</head>
<body>
<h1>🧪 LLM Eval Report</h1>
<p style="color:#666">Generated: {timestamp} | Provider: {provider}</p>

<div class="summary">
  <div class="metric"><div class="value">{total}</div><div class="label">Total Cases</div></div>
  <div class="metric"><div class="value pass">{passed}</div><div class="label">Passed</div></div>
  <div class="metric"><div class="value fail">{failed}</div><div class="label">Failed</div></div>
  <div class="metric"><div class="value {rate_class}">{pass_rate}</div><div class="label">Pass Rate</div></div>
</div>

<table>
<thead>
  <tr>
    <th>ID</th>
    <th>Description</th>
    <th>Type</th>
    <th>Result</th>
    <th>Score</th>
    <th>Response / Tool Calls / Turns</th>
    <th>Variance</th>
    <th>Latency</th>
    <th>Tokens (p/c)</th>
    <th>Tags</th>
  </tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<div class="footer">llm-eval-harness · github.com/pramathesh/llm-eval-harness</div>
</body>
</html>"""

_ROW = """<tr>
  <td><code>{case_id}</code></td>
  <td>{description}</td>
  <td>{type_badge}</td>
  <td><span class="badge {badge_class}">{result}</span></td>
  <td>{score}</td>
  <td class="response">{response}</td>
  <td class="variance">{variance}</td>
  <td>{latency} ms</td>
  <td style="font-size:0.82em;color:#555">{tokens}</td>
  <td>{tags}</td>
</tr>"""


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_response_cell(r) -> str:
    """Render the response/tool-call/turns cell based on result type."""
    case_type = getattr(r, "case_type", "standard")

    if case_type == "tool_call":
        turns = getattr(r, "turns", [])
        if turns and turns[0].tool_calls:
            tc = turns[0].tool_calls[0]
            import json
            args_str = json.dumps(tc.get("arguments", {}), ensure_ascii=False)
            if len(args_str) > 80:
                args_str = args_str[:80] + "…"
            return (
                f'<div class="tool-call">🔧 <strong>{_esc(tc["name"])}</strong>'
                f'({_esc(args_str)})</div>'
            )
        preview = _esc(r.response[:150])
        return f'<em style="color:#888">No tool called</em><br><span>{preview}</span>'

    if case_type == "multi_turn":
        turns = getattr(r, "turns", [])
        parts = [f'<div style="font-size:0.82em;color:#555">💬 {len(turns)} turns</div>']
        for t in turns:
            verdict = ""
            if t.scorer_result is not None:
                verdict = ('<span class="turn-pass">✓</span>' if t.scorer_result.passed
                           else '<span class="turn-fail">✗</span>')
            preview = _esc(t.response[:80]) + ("…" if len(t.response) > 80 else "")
            parts.append(
                f'<div class="turn-block">'
                f'<div class="turn-label">Turn {t.turn_index + 1} {verdict}</div>'
                f'{preview}</div>'
            )
        return "".join(parts)

    # Standard single-turn response
    preview = _esc(r.response[:200])
    if len(r.response) > 200:
        preview += "…"
    return preview


def _type_badge(r) -> str:
    case_type = getattr(r, "case_type", "standard")
    if case_type == "tool_call":
        return '<span class="badge badge-tool">tool call</span>'
    if case_type == "multi_turn":
        turns = getattr(r, "turns", [])
        return f'<span class="badge badge-multi">multi-turn ({len(turns)})</span>'
    return '<span style="color:#aaa;font-size:0.8em">standard</span>'


class HTMLReporter:

    def write(self, results: list, path: str = "eval_report.html",
              provider: str = "") -> str:
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        failed = total - passed
        pass_rate_pct = (passed / total * 100) if total else 0
        rate_class = "pass" if pass_rate_pct >= 80 else ("flaky" if pass_rate_pct >= 50 else "fail")

        rows = []
        for r in results:
            variance_html = ""
            if r.variance_report:
                v = r.variance_report
                vc = {"STABLE": "stable", "FLAKY": "flaky-v", "UNSTABLE": "unstable"}.get(v.verdict, "")
                variance_html = (
                    f'<span class="{vc}">{v.verdict}</span><br>'
                    f'pass_rate={v.pass_rate:.0%} | p50={v.latency_p50:.0f}ms'
                )

            tags_html = "".join(f'<span class="badge badge-tag">{t}</span>' for t in r.tags)
            tokens_html = (f"{r.prompt_tokens}/{r.completion_tokens}"
                           if r.prompt_tokens or r.completion_tokens else "—")
            rows.append(_ROW.format(
                case_id=r.case_id,
                description=r.description,
                type_badge=_type_badge(r),
                badge_class="badge-pass" if r.passed else "badge-fail",
                result="PASS" if r.passed else "FAIL",
                score=f"{r.scorer_result.score:.2f}" if r.scorer_result else "—",
                response=_render_response_cell(r),
                variance=variance_html or "—",
                latency=round(r.latency_ms),
                tokens=tokens_html,
                tags=tags_html,
            ))

        html = _TEMPLATE.format(
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            provider=provider or "unknown",
            total=total,
            passed=passed,
            failed=failed,
            pass_rate=f"{pass_rate_pct:.0f}%",
            rate_class=rate_class,
            rows="\n".join(rows),
        )
        Path(path).write_text(html)
        return path
