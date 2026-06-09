#!/usr/bin/env python3
"""
Main entry point for the LLM Eval Harness.

Usage:
  # Run all test suites against OpenAI
  python run_evals.py

  # Run specific suite against Anthropic
  python run_evals.py --provider anthropic --suite test_cases/basic_qa.yaml

  # Run multiple suites, output JSON only
  python run_evals.py --suite test_cases/basic_qa.yaml test_cases/instruction_following.yaml --format json

  # Fail CI if pass rate below threshold
  python run_evals.py --min-pass-rate 0.8
"""
import argparse
import sys
from pathlib import Path

from colorama import Fore, Style, init as colorama_init
from tabulate import tabulate

from config.settings import DEFAULT_PROVIDER
from providers import get_provider
from evals.runner import EvalRunner
from evals.agent_runner import AgentRunner
from reporters.html_reporter import HTMLReporter
from reporters.json_reporter import JSONReporter

colorama_init()

DEFAULT_SUITES = [
    "test_cases/basic_qa.yaml",
    "test_cases/instruction_following.yaml",
    "test_cases/complex_explanations.yaml",
]

ADVERSARIAL_SUITES = [
    "test_cases/prompt_injection.yaml",
    "test_cases/hallucination.yaml",
    "test_cases/safety_compliance.yaml",
    "test_cases/jailbreak.yaml",
]

AGENT_SUITES = [
    "test_cases/tool_calling.yaml",
    "test_cases/multi_turn.yaml",
]


def parse_args():
    p = argparse.ArgumentParser(description="LLM Eval Harness")
    p.add_argument("--provider", default=DEFAULT_PROVIDER,
                   choices=["openai", "anthropic"], help="LLM provider to use")
    p.add_argument("--suite", nargs="+", default=DEFAULT_SUITES,
                   help="YAML test suite file(s)")
    p.add_argument("--format", choices=["html", "json", "both"], default="both",
                   help="Report output format")
    p.add_argument("--output-dir", default=".", help="Directory for report files")
    p.add_argument("--adversarial", action="store_true",
                   help="Run Phase 2 adversarial suites instead of default suites")
    p.add_argument("--agent", action="store_true",
                   help="Run Phase 3 agent suites (tool calling + multi-turn)")
    p.add_argument("--all", action="store_true", dest="run_all",
                   help="Run ALL suites — Phase 1 + Phase 2 + Phase 3")
    p.add_argument("--min-pass-rate", type=float, default=0.0,
                   help="Exit code 1 if overall pass rate below this (0.0–1.0). Useful for CI.")
    p.add_argument("--variance-runs", type=int, default=None,
                   help="Override variance run count from .env")
    p.add_argument("--save-baseline", action="store_true",
                   help="Save current results as eval_baseline.json after the run")
    p.add_argument("--check-drift", action="store_true",
                   help="Compare results against eval_baseline.json after the run; exit 1 on regression")
    return p.parse_args()


def print_summary(results, provider_name: str):
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    rate = passed / total if total else 0

    color = Fore.GREEN if rate >= 0.8 else (Fore.YELLOW if rate >= 0.5 else Fore.RED)
    print(f"\n{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
    print(f"Provider : {provider_name}")
    print(f"Total    : {total}")
    print(f"Passed   : {Fore.GREEN}{passed}{Style.RESET_ALL}")
    print(f"Failed   : {Fore.RED}{total - passed}{Style.RESET_ALL}")
    print(f"Pass Rate: {color}{rate:.0%}{Style.RESET_ALL}")
    print(f"{'='*60}\n")

    rows = []
    for r in results:
        status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if r.passed else f"{Fore.RED}FAIL{Style.RESET_ALL}"
        variance = ""
        if r.variance_report:
            v = r.variance_report
            vc = Fore.GREEN if v.verdict == "STABLE" else (Fore.YELLOW if v.verdict == "FLAKY" else Fore.RED)
            variance = f"{vc}{v.verdict}{Style.RESET_ALL} ({v.pass_rate:.0%})"
        rows.append([
            r.case_id,
            r.description[:45] + "..." if len(r.description) > 45 else r.description,
            status,
            f"{r.scorer_result.score:.2f}" if r.scorer_result else "—",
            variance or "—",
            f"{r.latency_ms:.0f}ms",
        ])

    print(tabulate(rows, headers=["ID", "Description", "Result", "Score", "Variance", "Latency"],
                   tablefmt="simple"))
    print()


def main():
    args = parse_args()

    print(f"\n{Style.BRIGHT}🧪 LLM Eval Harness{Style.RESET_ALL}")
    print(f"Provider: {args.provider}")

    provider = get_provider(args.provider)

    from config.settings import VARIANCE_RUNS
    variance_runs = args.variance_runs or VARIANCE_RUNS
    runner = EvalRunner(provider, variance_runs=variance_runs)
    agent_runner = AgentRunner(provider)

    # Determine which suites to run and with which runner
    if args.run_all:
        standard_suites = DEFAULT_SUITES + ADVERSARIAL_SUITES
        agent_suites = AGENT_SUITES
    elif args.adversarial:
        standard_suites = ADVERSARIAL_SUITES
        agent_suites = []
    elif args.agent:
        standard_suites = []
        agent_suites = AGENT_SUITES
    else:
        standard_suites = args.suite
        agent_suites = []

    all_results = []

    for suite_path in standard_suites:
        if not Path(suite_path).exists():
            print(f"{Fore.YELLOW}⚠ Suite not found: {suite_path}{Style.RESET_ALL}")
            continue
        print(f"\nRunning suite: {suite_path}")
        results = runner.run_file(suite_path)
        all_results.extend(results)
        for r in results:
            icon = "✅" if r.passed else "❌"
            print(f"  {icon} [{r.case_id}] {r.description[:60]}")

    for suite_path in agent_suites:
        if not Path(suite_path).exists():
            print(f"{Fore.YELLOW}⚠ Suite not found: {suite_path}{Style.RESET_ALL}")
            continue
        print(f"\nRunning agent suite: {suite_path}")
        results = agent_runner.run_file(suite_path)
        all_results.extend(results)
        for r in results:
            icon = "✅" if r.passed else "❌"
            case_type_label = f"[{r.case_type}]" if hasattr(r, "case_type") else ""
            print(f"  {icon} [{r.case_id}]{case_type_label} {r.description[:55]}")

    if not all_results:
        print(f"{Fore.RED}No results — check suite paths.{Style.RESET_ALL}")
        sys.exit(1)

    print_summary(all_results, args.provider)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.format in ("html", "both"):
        path = HTMLReporter().write(all_results, str(out / "eval_report.html"), provider=args.provider)
        print(f"📄 HTML report: {path}")

    json_path = str(out / "eval_results.json")
    if args.format in ("json", "both"):
        path = JSONReporter().write(all_results, json_path, provider=args.provider)
        print(f"📊 JSON report: {path}")

    # CI gate
    passed = sum(1 for r in all_results if r.passed)
    rate = passed / len(all_results)
    if rate < args.min_pass_rate:
        print(f"\n{Fore.RED}✗ Pass rate {rate:.0%} below required {args.min_pass_rate:.0%}. Failing CI.{Style.RESET_ALL}")
        sys.exit(1)

    # Baseline management
    if args.save_baseline:
        import shutil
        baseline_path = out / "eval_baseline.json"
        shutil.copy(json_path, str(baseline_path))
        print(f"📌 Baseline saved: {baseline_path}")

    # Drift detection
    if args.check_drift:
        baseline_path = out / "eval_baseline.json"
        if baseline_path.exists():
            from evals.drift_detector import DriftDetector
            drift_report = DriftDetector().compare(str(baseline_path), json_path)
            drift_report.print_summary()
            if drift_report.has_regressions:
                print(f"{Fore.RED}✗ Regressions detected — failing.{Style.RESET_ALL}")
                sys.exit(1)
        else:
            print(f"{Fore.YELLOW}⚠ No baseline found at {baseline_path}. Run with --save-baseline first.{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}✓ Done.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
