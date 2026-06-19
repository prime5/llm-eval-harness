"""
Phase 5 — RAG Drift Detector

Compares two rag_eval_results.json files and flags regressions
in any of the four RAG eval dimensions.

Usage:
    python3 phase5/rag_drift.py
    python3 phase5/rag_drift.py --baseline phase5/data/rag_baseline.json \
                                 --current  phase5/data/rag_eval_results.json
    python3 phase5/rag_drift.py --fail-on-regression  # exit code 1 if regression

CI usage (see eval_ci.yml):
    python3 phase5/rag_drift.py --fail-on-regression --output phase5/data/rag_drift.json
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Minimum drop in a metric that counts as a regression
REGRESSION_THRESHOLD = 0.05   # 5 percentage points


@dataclass
class MetricDelta:
    metric:    str
    baseline:  float
    current:   float
    delta:     float           # current - baseline (negative = regression)
    regressed: bool


@dataclass
class RAGDriftReport:
    baseline_run: str
    current_run:  str
    deltas:       list[MetricDelta] = field(default_factory=list)
    regressions:  list[MetricDelta] = field(default_factory=list)
    has_regression: bool = False


METRICS = [
    ("precision_pass_rate",    "Retrieval Precision"),
    ("faithfulness_avg_score", "Answer Faithfulness"),
    ("sufficiency_pass_rate",  "Context Sufficiency"),
    ("adversarial_pass_rate",  "Adversarial Retrieval"),
]


def load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def compare(baseline: dict, current: dict) -> RAGDriftReport:
    report = RAGDriftReport(
        baseline_run=baseline.get("run_at", "unknown"),
        current_run=current.get("run_at", "unknown"),
    )
    for key, label in METRICS:
        b_val = baseline.get(key, 0.0)
        c_val = current.get(key, 0.0)
        delta = c_val - b_val
        regressed = delta < -REGRESSION_THRESHOLD
        d = MetricDelta(
            metric=label, baseline=b_val, current=c_val,
            delta=delta, regressed=regressed,
        )
        report.deltas.append(d)
        if regressed:
            report.regressions.append(d)

    report.has_regression = len(report.regressions) > 0
    return report


def print_report(report: RAGDriftReport) -> None:
    status = "✗ REGRESSION DETECTED" if report.has_regression else "✓ No regression"
    print(f"\n── RAG Drift Report — {status} {'─' * 20}")
    print(f"  Baseline : {report.baseline_run}")
    print(f"  Current  : {report.current_run}\n")
    for d in report.deltas:
        arrow = "▼" if d.regressed else ("▲" if d.delta > 0.01 else "─")
        flag  = " ← REGRESSION" if d.regressed else ""
        print(f"  {arrow} {d.metric:<28} {d.baseline:.3f} → {d.current:.3f}"
              f"  (Δ {d.delta:+.3f}){flag}")
    print()


def main():
    parser = argparse.ArgumentParser(description="RAG eval drift detector")
    parser.add_argument("--baseline", default="phase5/data/rag_baseline.json")
    parser.add_argument("--current",  default="phase5/data/rag_eval_results.json")
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--output", help="Write drift report JSON to this path")
    args = parser.parse_args()

    if not Path(args.baseline).exists():
        print(f"No baseline found at {args.baseline} — skipping drift check (first run)")
        sys.exit(0)

    baseline = load(args.baseline)
    current  = load(args.current)
    report   = compare(baseline, current)
    print_report(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Drift report written to {args.output}")

    if args.fail_on_regression and report.has_regression:
        print("CI gate: failing build due to RAG regression.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
