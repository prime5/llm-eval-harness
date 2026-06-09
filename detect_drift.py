#!/usr/bin/env python3
"""
Detect score drift between two eval runs.

Usage:
    # Compare baseline vs latest run
    python detect_drift.py

    # Explicit paths
    python detect_drift.py --baseline eval_baseline.json --current eval_results.json

    # Tighter threshold (flag changes > 5 points)
    python detect_drift.py --threshold 0.05

    # Post to Slack on regression
    python detect_drift.py --slack-webhook https://hooks.slack.com/services/...

    # Exit code 1 if regressions found (for CI gate)
    python detect_drift.py --fail-on-regression

    # Write drift report as JSON
    python detect_drift.py --output drift_report.json
"""
import argparse
import json
import sys
from pathlib import Path

from colorama import Fore, Style, init as colorama_init

from evals.drift_detector import DriftDetector

colorama_init()


def parse_args():
    p = argparse.ArgumentParser(description="LLM Eval Harness — Drift Detector")
    p.add_argument("--baseline", default="eval_baseline.json",
                   help="Path to baseline JSON report (default: eval_baseline.json)")
    p.add_argument("--current", default="eval_results.json",
                   help="Path to current JSON report (default: eval_results.json)")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Min absolute score delta to flag as drop/gain (default: 0.1)")
    p.add_argument("--slack-webhook", metavar="URL",
                   help="Slack incoming webhook URL — posts alert if regressions found")
    p.add_argument("--fail-on-regression", action="store_true",
                   help="Exit code 1 if any regressions detected (for CI gate)")
    p.add_argument("--output", metavar="PATH",
                   help="Write drift report as JSON to this file")
    return p.parse_args()


def post_slack(webhook_url: str, report) -> None:
    """Post a regression alert to Slack via incoming webhook."""
    import urllib.request

    reg_lines = "\n".join(f"• `{c.case_id}`: {c.description[:60]}" for c in report.regressions)
    text = (
        f":rotating_light: *LLM Eval Regression Detected*\n"
        f"Provider: `{report.provider}`\n"
        f"Baseline: `{report.baseline_timestamp}`\n"
        f"Current:  `{report.current_timestamp}`\n\n"
        f"*{len(report.regressions)} regression(s):*\n{reg_lines}"
    )
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                print(f"{Fore.YELLOW}⚠ Slack webhook returned HTTP {resp.status}{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}✓ Slack alert posted.{Style.RESET_ALL}")
    except Exception as exc:
        print(f"{Fore.RED}✗ Slack webhook failed: {exc}{Style.RESET_ALL}")


def main():
    args = parse_args()

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)

    if not baseline_path.exists():
        print(f"{Fore.RED}✗ Baseline not found: {args.baseline}{Style.RESET_ALL}")
        print(f"  Run: python run_evals.py --save-baseline  to create one.")
        sys.exit(1)

    if not current_path.exists():
        print(f"{Fore.RED}✗ Current report not found: {args.current}{Style.RESET_ALL}")
        print(f"  Run: python run_evals.py  to generate one.")
        sys.exit(1)

    detector = DriftDetector(threshold=args.threshold)
    report = detector.compare(str(baseline_path), str(current_path))
    report.print_summary()

    if args.output:
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"📊 Drift report written: {args.output}")

    if report.regressions and args.slack_webhook:
        post_slack(args.slack_webhook, report)

    if args.fail_on_regression and report.has_regressions:
        print(f"{Fore.RED}✗ Failing CI — {len(report.regressions)} regression(s) detected.{Style.RESET_ALL}")
        sys.exit(1)

    if not report.has_regressions:
        print(f"{Fore.GREEN}✓ No regressions. Baseline holds.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
