"""HTML reporter — visual report for humans."""
from datetime import datetime
from pathlib import Path
from evals.runner import EvalResult

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
    <th>Result</th>
    <th>Score</th>
    <th>Response Preview</th>
    <th>Variance</th>
    <th>Latency</th>
    <th>Tags</th>
  </tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<div class="footer">llm-eval-harness · Phase 1 · github.com/pramathesh/llm-eval-harness</div>
</body>
</html>"""

_ROW = """<tr>
  <td><code>{case_id}</code></td>
  <td>{description}</td>
  <td><span class="badge {badge_class}">{result}</span></td>
  <td>{score}</td>
  <td class="response">{response}</td>
  <td class="variance">{variance}</td>
  <td>{latency} ms</td>
  <td>{tags}</td>
</tr>"""


class HTMLReporter:

    def write(self, results: list[EvalResult], path: str = "eval_report.html",
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
            rows.append(_ROW.format(
                case_id=r.case_id,
                description=r.description,
                badge_class="badge-pass" if r.passed else "badge-fail",
                result="PASS" if r.passed else "FAIL",
                score=f"{r.scorer_result.score:.2f}" if r.scorer_result else "—",
                response=r.response[:200].replace("<", "&lt;").replace(">", "&gt;") + ("..." if len(r.response) > 200 else ""),
                variance=variance_html or "—",
                latency=round(r.latency_ms),
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
