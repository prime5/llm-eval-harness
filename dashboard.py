"""
Streamlit dashboard for LLM Eval Harness.

Usage:
    streamlit run dashboard.py
    streamlit run dashboard.py -- --report path/to/eval_results.json

Tabs:
  - Core Evals (Phases 1–4): pass rate, score distribution, drift, results table
  - RAG Eval (Phase 5): precision, faithfulness, sufficiency, adversarial metrics
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
# Sidebar                                                              #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.title("🧪 LLM Eval Harness")
    st.markdown("---")

    default_report = "eval_results.json"
    if len(sys.argv) > 2 and sys.argv[1] == "--report":
        default_report = sys.argv[2]

    report_path   = st.text_input("Core eval report", value=default_report)
    baseline_path = st.text_input("Core baseline (optional)", value="eval_baseline.json")
    rag_report    = st.text_input("RAG eval report", value="phase5/data/rag_eval_results.json")
    rag_baseline  = st.text_input("RAG baseline (optional)", value="phase5/data/rag_baseline.json")

    st.markdown("---")
    st.markdown("**Run core evals:**")
    st.code("python3 run_evals.py", language="bash")
    st.markdown("**Run RAG evals:**")
    st.code("python3 -m phase5.eval_suite", language="bash")
    st.markdown("**Check drift:**")
    st.code("python3 detect_drift.py", language="bash")
    st.code("python3 phase5/rag_drift.py", language="bash")

# ------------------------------------------------------------------ #
# Tabs                                                                 #
# ------------------------------------------------------------------ #

tab_main, tab_rag = st.tabs(["📋 Core Evals (Phases 1–4)", "🔍 RAG Eval (Phase 5)"])

# ================================================================== #
# TAB 1 — Core Evals                                                  #
# ================================================================== #

with tab_main:

    if not Path(report_path).exists():
        st.error(f"Report not found: `{report_path}`")
        st.info("Run `python3 run_evals.py` to generate a report, then refresh.")
        st.stop()

    with open(report_path) as f:
        data = json.load(f)

    results  = data.get("results", [])
    summary  = data.get("summary", {})
    provider = data.get("provider", "unknown")
    run_ts   = data.get("run_timestamp", "unknown")

    if not results:
        st.error("No results found in the report.")
        st.stop()

    # Header
    st.title("🧪 LLM Eval Report")
    st.caption(f"Provider: **{provider}** | Run: `{run_ts}` | File: `{report_path}`")

    # Summary metrics
    total        = summary.get("total", len(results))
    passed       = summary.get("passed", sum(1 for r in results if r.get("passed")))
    failed       = summary.get("failed", total - passed)
    pass_rate_raw = summary.get("pass_rate", passed / total if total else 0)
    pass_rate_pct = pass_rate_raw if isinstance(pass_rate_raw, str) else f"{pass_rate_raw:.0%}"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cases", total)
    col2.metric("Passed",      passed)
    col3.metric("Failed",      failed)
    col4.metric("Pass Rate",   pass_rate_pct)
    st.markdown("---")

    # Build DataFrame
    rows = []
    for r in results:
        tags      = r.get("tags", [])
        variance  = r.get("variance")
        v_verdict = variance.get("verdict", "") if variance else ""
        rows.append({
            "ID":           r.get("case_id", ""),
            "Description":  r.get("description", ""),
            "Type":         r.get("case_type", "standard"),
            "Result":       "PASS" if r.get("passed") else "FAIL",
            "Score":        r.get("score"),
            "Latency (ms)": r.get("latency_ms"),
            "Variance":     v_verdict,
            "Tags":         ", ".join(tags),
            "Tokens (p/c)": f"{r.get('prompt_tokens',0)}/{r.get('completion_tokens',0)}",
        })
    df = pd.DataFrame(rows)

    # Charts
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        st.subheader("Pass Rate by Tag")
        tag_data: dict[str, list] = {}
        for r in results:
            for tag in r.get("tags", []):
                tag_data.setdefault(tag, []).append(1 if r.get("passed") else 0)
        if tag_data:
            tag_df = pd.DataFrame({
                "Tag":       list(tag_data.keys()),
                "Pass Rate": [sum(v) / len(v) for v in tag_data.values()],
            }).set_index("Tag").sort_values("Pass Rate", ascending=True)
            st.bar_chart(tag_df)
        else:
            st.info("No tags defined in test cases.")

    with chart_col2:
        st.subheader("Score Distribution")
        scores = [r.get("score") for r in results if r.get("score") is not None]
        if scores:
            bins   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
            labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
            counts = [0] * 5
            for s in scores:
                for i, b in enumerate(bins[:-1]):
                    if b <= s < bins[i + 1]:
                        counts[i] += 1
                        break
            st.bar_chart(
                pd.DataFrame({"Score Range": labels, "Cases": counts}).set_index("Score Range")
            )
        else:
            st.info("No score data available.")

    with chart_col3:
        st.subheader("Latency (ms)")
        latencies = [(r.get("case_id", ""), r.get("latency_ms", 0)) for r in results]
        if latencies:
            st.bar_chart(
                pd.DataFrame(latencies, columns=["Case", "Latency (ms)"]).set_index("Case")
            )

    st.markdown("---")

    # Drift section
    if Path(baseline_path).exists():
        st.subheader("📊 Score Drift vs Baseline")
        try:
            from evals.drift_detector import DriftDetector
            drift = DriftDetector().compare(baseline_path, report_path)

            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            d_col1.metric("Regressions",  len(drift.regressions),
                          delta=f"-{len(drift.regressions)}" if drift.regressions else None,
                          delta_color="inverse")
            d_col2.metric("Improvements", len(drift.improvements),
                          delta=f"+{len(drift.improvements)}" if drift.improvements else None)
            d_col3.metric("Score Drops",  len(drift.score_drops))
            d_col4.metric("Score Gains",  len(drift.score_gains))

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
                st.dataframe(pd.DataFrame([
                    {"Case": c.case_id, "Baseline": c.baseline_score,
                     "Current": c.current_score, "Delta": c.score_delta}
                    for c in drift.score_drops
                ]), use_container_width=True)
            if not drift.has_changes:
                st.success("✓ No drift detected — all cases stable vs baseline.")

        except Exception as exc:
            st.warning(f"Could not compute drift: {exc}")
        st.markdown("---")

    # Results table
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
        if val == "PASS": return "color: #27ae60; font-weight: bold"
        if val == "FAIL": return "color: #e74c3c; font-weight: bold"
        return ""

    st.dataframe(
        filtered.style.map(_color_result, subset=["Result"]),
        use_container_width=True, height=500,
    )

    st.markdown("---")
    st.caption("llm-eval-harness · github.com/pramathesh/llm-eval-harness")


# ================================================================== #
# TAB 2 — Phase 5: RAG Eval                                          #
# ================================================================== #

with tab_rag:

    st.title("🔍 RAG Eval — Phase 5")
    st.caption("Retrieval precision · Answer faithfulness · Context sufficiency · Adversarial retrieval")

    # ── Load RAG results ──────────────────────────────────────────────
    if not Path(rag_report).exists():
        st.warning(f"No RAG eval results found at `{rag_report}`.")
        st.info("Run the eval suite first:")
        st.code("python3 -m phase5.eval_suite", language="bash")
        st.stop()

    with open(rag_report) as f:
        rag = json.load(f)

    st.caption(f"Run: `{rag.get('run_at', 'unknown')}` | File: `{rag_report}`")

    # ── Summary metric cards ──────────────────────────────────────────
    overall = "✅ PASSED" if rag.get("passed") else "❌ FAILED"
    st.subheader(f"Overall: {overall}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Retrieval Precision",   f"{rag.get('precision_pass_rate', 0):.0%}",   help="% of queries where top-1 chunk matched expected source")
    m2.metric("Answer Faithfulness",   f"{rag.get('faithfulness_avg_score', 0):.2f}", help="LLM-judge score 0–1 (1 = fully grounded, no hallucination)")
    m3.metric("Context Sufficiency",   f"{rag.get('sufficiency_pass_rate', 0):.0%}",  help="% of queries correctly answered or refused based on context")
    m4.metric("Adversarial Retrieval", f"{rag.get('adversarial_pass_rate', 0):.0%}",  help="% of misleading queries that still retrieved the correct source")

    st.markdown("---")

    # ── RAG drift vs baseline ─────────────────────────────────────────
    if Path(rag_baseline).exists():
        st.subheader("📊 RAG Drift vs Baseline")
        try:
            with open(rag_baseline) as f:
                base = json.load(f)

            METRICS = [
                ("precision_pass_rate",    "Retrieval Precision"),
                ("faithfulness_avg_score", "Answer Faithfulness"),
                ("sufficiency_pass_rate",  "Context Sufficiency"),
                ("adversarial_pass_rate",  "Adversarial Retrieval"),
            ]
            drift_rows = []
            any_regression = False
            for key, label in METRICS:
                b_val  = base.get(key, 0.0)
                c_val  = rag.get(key, 0.0)
                delta  = c_val - b_val
                reg    = delta < -0.05
                if reg:
                    any_regression = True
                drift_rows.append({
                    "Metric":   label,
                    "Baseline": f"{b_val:.3f}",
                    "Current":  f"{c_val:.3f}",
                    "Δ":        f"{delta:+.3f}",
                    "Status":   "⚠ REGRESSION" if reg else ("↑ improved" if delta > 0.01 else "stable"),
                })

            if any_regression:
                st.error("Regression detected in one or more RAG metrics.")
            else:
                st.success("✓ No RAG regression vs baseline.")
            st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)

        except Exception as exc:
            st.warning(f"Could not compute RAG drift: {exc}")
        st.markdown("---")

    # ── Per-dimension detail tables ───────────────────────────────────
    detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs([
        "Retrieval Precision", "Faithfulness", "Sufficiency", "Adversarial",
    ])

    def result_badge(passed: bool) -> str:
        return "✓" if passed else "✗"

    with detail_tab1:
        rows = [
            {
                "":       result_badge(r["passed"]),
                "Query":  r["query"],
                "Expected": r["expected_source"],
                "Got":    r["actual_top1"],
                "Distance": f"{r['distance']:.4f}",
            }
            for r in rag.get("precision_results", [])
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with detail_tab2:
        rows = [
            {
                "":      result_badge(r["passed"]),
                "Query": r["query"],
                "Score": f"{r['score']:.2f}",
                "Unsupported claims": "; ".join(r.get("unsupported_claims", [])) or "none",
            }
            for r in rag.get("faithfulness_results", [])
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with detail_tab3:
        rows = [
            {
                "":         result_badge(r["passed"]),
                "Query":    r["query"],
                "Expected": "answer" if r["expected_answer"] else "refuse",
                "Got":      "answered" if r["got_answer"] else "refused",
                "Response": r.get("chain_response", "")[:120],
            }
            for r in rag.get("sufficiency_results", [])
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with detail_tab4:
        rows = [
            {
                "":            result_badge(r["passed"]),
                "Query":       r["query"],
                "Expected":    r["correct_source"],
                "Got":         r["actual_top1"],
                "Distance":    f"{r['distance']:.4f}",
                "Description": r["description"],
            }
            for r in rag.get("adversarial_results", [])
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.caption("llm-eval-harness · Phase 5 RAG Evaluation · github.com/pramathesh/llm-eval-harness")
