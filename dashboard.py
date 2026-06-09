"""
Streamlit dashboard for LLM Eval Harness.

Usage:
    streamlit run dashboard.py
    streamlit run dashboard.py -- --report path/to/eval_results.json

Shows:
  - Summary metrics (pass rate, totals, provider)
  - Pass rate by tag
  - Score distribution
  - Latency per case
  - Score drift section (if eval_baseline.json exists)
  - Full results table
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="LLM Eval Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Sidebar — report picker                                              #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.title("🧪 LLM Eval Harness")
    st.markdown("---")

    # Allow --report flag from CLI args
    default_report = "eval_results.json"
    if len(sys.argv) > 2 and sys.argv[1] == "--report":
        default_report = sys.argv[2]

    report_path = st.text_input("Report path", value=default_report)
    baseline_path = st.text_input("Baseline path (optional)", value="eval_baseline.json")

    st.markdown("---")
    st.markdown("**Run evals:**")
    st.code("python3 run_evals.py", language="bash")
    st.markdown("**Save baseline:**")
    st.code("python3 run_evals.py --save-baseline", language="bash")
    st.markdown("**Check drift:**")
    st.code("python3 detect_drift.py", language="bash")

# ------------------------------------------------------------------ #
# Load data                                                            #
# ------------------------------------------------------------------ #

if not Path(report_path).exists():
    st.error(f"Report not found: `{report_path}`")
    st.info("Run `python3 run_evals.py` to generate a report, then refresh.")
    st.stop()

with open(report_path) as f:
    data = json.load(f)

results = data.get("results", [])
summary = data.get("summary", {})
provider = data.get("provider", "unknown")
run_ts = data.get("run_timestamp", "unknown")

if not results:
    st.error("No results found in the report.")
    st.stop()

# ------------------------------------------------------------------ #
# Header                                                               #
# ------------------------------------------------------------------ #

st.title("🧪 LLM Eval Report")
st.caption(f"Provider: **{provider}** | Run: `{run_ts}` | File: `{report_path}`")

# ------------------------------------------------------------------ #
# Summary metrics                                                      #
# ------------------------------------------------------------------ #

total = summary.get("total", len(results))
passed = summary.get("passed", sum(1 for r in results if r.get("passed")))
failed = summary.get("failed", total - passed)
pass_rate_raw = summary.get("pass_rate", passed / total if total else 0)
# Handle both float and "83%" string forms
if isinstance(pass_rate_raw, str):
    pass_rate_pct = pass_rate_raw
else:
    pass_rate_pct = f"{pass_rate_raw:.0%}"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cases", total)
col2.metric("Passed", passed, delta=None)
col3.metric("Failed", failed, delta=None)
col4.metric("Pass Rate", pass_rate_pct)

st.markdown("---")

# ------------------------------------------------------------------ #
# Build DataFrame                                                      #
# ------------------------------------------------------------------ #

rows = []
for r in results:
    tags = r.get("tags", [])
    variance = r.get("variance")
    v_verdict = variance.get("verdict", "") if variance else ""
    case_type = r.get("case_type", "standard")
    rows.append({
        "ID": r.get("case_id", ""),
        "Description": r.get("description", ""),
        "Type": case_type,
        "Result": "PASS" if r.get("passed") else "FAIL",
        "Score": r.get("score"),
        "Latency (ms)": r.get("latency_ms"),
        "Variance": v_verdict,
        "Tags": ", ".join(tags),
        "Tokens (p/c)": f"{r.get('prompt_tokens',0)}/{r.get('completion_tokens',0)}",
    })

df = pd.DataFrame(rows)

# ------------------------------------------------------------------ #
# Charts row                                                           #
# ------------------------------------------------------------------ #

chart_col1, chart_col2, chart_col3 = st.columns(3)

# Pass rate by tag
with chart_col1:
    st.subheader("Pass Rate by Tag")
    tag_data: dict[str, list] = {}
    for r in results:
        for tag in r.get("tags", []):
            tag_data.setdefault(tag, []).append(1 if r.get("passed") else 0)
    if tag_data:
        tag_df = pd.DataFrame({
            "Tag": list(tag_data.keys()),
            "Pass Rate": [sum(v) / len(v) for v in tag_data.values()],
        }).set_index("Tag").sort_values("Pass Rate", ascending=True)
        st.bar_chart(tag_df)
    else:
        st.info("No tags defined in test cases.")

# Score distribution
with chart_col2:
    st.subheader("Score Distribution")
    scores = [r.get("score") for r in results if r.get("score") is not None]
    if scores:
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
        counts = [0] * 5
        for s in scores:
            for i, b in enumerate(bins[:-1]):
                if b <= s < bins[i + 1]:
                    counts[i] += 1
                    break
        score_df = pd.DataFrame({"Score Range": labels, "Cases": counts}).set_index("Score Range")
        st.bar_chart(score_df)
    else:
        st.info("No score data available.")

# Latency per case
with chart_col3:
    st.subheader("Latency (ms)")
    latencies = [(r.get("case_id", ""), r.get("latency_ms", 0)) for r in results]
    if latencies:
        lat_df = pd.DataFrame(latencies, columns=["Case", "Latency (ms)"]).set_index("Case")
        st.bar_chart(lat_df)

st.markdown("---")

# ------------------------------------------------------------------ #
# Drift section (only if baseline exists)                              #
# ------------------------------------------------------------------ #

if Path(baseline_path).exists():
    st.subheader("📊 Score Drift vs Baseline")
    try:
        from evals.drift_detector import DriftDetector
        drift = DriftDetector().compare(baseline_path, report_path)

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        reg_delta = f"-{len(drift.regressions)}" if drift.regressions else "0"
        imp_delta = f"+{len(drift.improvements)}" if drift.improvements else "0"
        d_col1.metric("Regressions", len(drift.regressions),
                       delta=reg_delta if drift.regressions else None,
                       delta_color="inverse")
        d_col2.metric("Improvements", len(drift.improvements),
                       delta=imp_delta if drift.improvements else None)
        d_col3.metric("Score Drops", len(drift.score_drops))
        d_col4.metric("Score Gains", len(drift.score_gains))

        if drift.regressions:
            st.error("**Regressions detected (PASS → FAIL):**")
            for c in drift.regressions:
                st.markdown(f"- `{c.case_id}` — {c.description}")

        if drift.improvements:
            st.success("**Improvements detected (FAIL → PASS):**")
            for c in drift.improvements:
                st.markdown(f"- `{c.case_id}` — {c.description}")

        if drift.score_drops:
            st.warning("**Score drops (still passing, score fell):**")
            drop_rows = [
                {"Case": c.case_id, "Baseline": c.baseline_score,
                 "Current": c.current_score, "Delta": c.score_delta}
                for c in drift.score_drops
            ]
            st.dataframe(pd.DataFrame(drop_rows), use_container_width=True)

        if not drift.has_changes:
            st.success("✓ No drift detected — all cases stable vs baseline.")

    except Exception as exc:
        st.warning(f"Could not compute drift: {exc}")

    st.markdown("---")

# ------------------------------------------------------------------ #
# Results table                                                        #
# ------------------------------------------------------------------ #

st.subheader("Results")

filter_col1, filter_col2 = st.columns([1, 3])
with filter_col1:
    show_filter = st.selectbox("Filter", ["All", "PASS only", "FAIL only"])
with filter_col2:
    type_filter = st.multiselect(
        "Case type",
        options=df["Type"].unique().tolist(),
        default=df["Type"].unique().tolist(),
    )

filtered = df[df["Type"].isin(type_filter)]
if show_filter == "PASS only":
    filtered = filtered[filtered["Result"] == "PASS"]
elif show_filter == "FAIL only":
    filtered = filtered[filtered["Result"] == "FAIL"]

def _color_result(val):
    if val == "PASS":
        return "color: #27ae60; font-weight: bold"
    if val == "FAIL":
        return "color: #e74c3c; font-weight: bold"
    return ""

styled = filtered.style.map(_color_result, subset=["Result"])
st.dataframe(styled, use_container_width=True, height=500)

# ------------------------------------------------------------------ #
# Footer                                                               #
# ------------------------------------------------------------------ #

st.markdown("---")
st.caption("llm-eval-harness · github.com/pramathesh/llm-eval-harness")
