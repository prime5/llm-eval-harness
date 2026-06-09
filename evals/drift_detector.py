"""
Score drift detection — compare two eval runs and surface regressions.

The core insight: a harness that only tells you today's pass rate is a
snapshot. One that tells you which cases regressed since the last run is
a continuous quality system.

Usage:
    detector = DriftDetector(threshold=0.1)
    report = detector.compare("eval_baseline.json", "eval_results.json")
    report.print_summary()
    if report.has_regressions:
        sys.exit(1)

Change types:
    regression   — case went from PASS to FAIL (most severe)
    improvement  — case went from FAIL to PASS (positive signal)
    score_drop   — still passing but score fell more than threshold
    score_gain   — still passing but score rose more than threshold
    stable       — no meaningful change
    new          — in current run but not in baseline
    removed      — in baseline but not in current run
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CaseDrift:
    case_id: str
    description: str
    change_type: str          # regression | improvement | score_drop | score_gain | stable | new | removed
    baseline_passed: Optional[bool]
    current_passed: Optional[bool]
    baseline_score: Optional[float]
    current_score: Optional[float]

    @property
    def score_delta(self) -> Optional[float]:
        if self.baseline_score is not None and self.current_score is not None:
            return round(self.current_score - self.baseline_score, 4)
        return None

    def one_line(self) -> str:
        delta = f" (Δ{self.score_delta:+.2f})" if self.score_delta is not None else ""
        b = "PASS" if self.baseline_passed else "FAIL"
        c = "PASS" if self.current_passed else "FAIL"
        return f"[{self.case_id}] {self.description[:50]} | {b} → {c}{delta}"


@dataclass
class DriftReport:
    baseline_timestamp: str
    current_timestamp: str
    provider: str
    threshold: float

    regressions: list[CaseDrift] = field(default_factory=list)
    improvements: list[CaseDrift] = field(default_factory=list)
    score_drops: list[CaseDrift] = field(default_factory=list)
    score_gains: list[CaseDrift] = field(default_factory=list)
    stable: list[CaseDrift] = field(default_factory=list)
    new_cases: list[CaseDrift] = field(default_factory=list)
    removed_cases: list[CaseDrift] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)

    @property
    def has_changes(self) -> bool:
        return bool(
            self.regressions or self.improvements or
            self.score_drops or self.score_gains or
            self.new_cases or self.removed_cases
        )

    @property
    def total_compared(self) -> int:
        return (len(self.regressions) + len(self.improvements) +
                len(self.score_drops) + len(self.score_gains) + len(self.stable))

    def to_dict(self) -> dict:
        def case_list(cases):
            return [
                {
                    "case_id": c.case_id,
                    "description": c.description,
                    "change_type": c.change_type,
                    "baseline_passed": c.baseline_passed,
                    "current_passed": c.current_passed,
                    "baseline_score": c.baseline_score,
                    "current_score": c.current_score,
                    "score_delta": c.score_delta,
                }
                for c in cases
            ]
        return {
            "baseline_timestamp": self.baseline_timestamp,
            "current_timestamp": self.current_timestamp,
            "provider": self.provider,
            "threshold": self.threshold,
            "summary": {
                "total_compared": self.total_compared,
                "regressions": len(self.regressions),
                "improvements": len(self.improvements),
                "score_drops": len(self.score_drops),
                "score_gains": len(self.score_gains),
                "new_cases": len(self.new_cases),
                "removed_cases": len(self.removed_cases),
            },
            "regressions": case_list(self.regressions),
            "improvements": case_list(self.improvements),
            "score_drops": case_list(self.score_drops),
            "score_gains": case_list(self.score_gains),
            "new_cases": case_list(self.new_cases),
            "removed_cases": case_list(self.removed_cases),
        }

    def print_summary(self, use_color: bool = True) -> None:
        try:
            from colorama import Fore, Style
        except ImportError:
            use_color = False

        def red(s):   return f"{Fore.RED}{s}{Style.RESET_ALL}" if use_color else s
        def green(s): return f"{Fore.GREEN}{s}{Style.RESET_ALL}" if use_color else s
        def yellow(s): return f"{Fore.YELLOW}{s}{Style.RESET_ALL}" if use_color else s
        def bold(s):  return f"{Style.BRIGHT}{s}{Style.RESET_ALL}" if use_color else s

        print(f"\n{bold('='*60)}")
        print(bold("Score Drift Report"))
        print(f"Baseline : {self.baseline_timestamp}")
        print(f"Current  : {self.current_timestamp}")
        print(f"Provider : {self.provider}")
        print(f"{'='*60}")
        print(f"Compared : {self.total_compared} cases | threshold={self.threshold}")
        print(f"  {red(f'Regressions : {len(self.regressions)}')}")
        print(f"  {green(f'Improvements: {len(self.improvements)}')}")
        print(f"  {yellow(f'Score drops : {len(self.score_drops)}')}")
        print(f"  {green(f'Score gains : {len(self.score_gains)}')}")
        if self.new_cases:
            print(f"  New cases   : {len(self.new_cases)}")
        if self.removed_cases:
            print(f"  Removed     : {len(self.removed_cases)}")
        print(f"{'='*60}")

        if self.regressions:
            print(f"\n{red('REGRESSIONS (PASS → FAIL):')}")
            for c in self.regressions:
                print(f"  ❌ {c.one_line()}")

        if self.improvements:
            print(f"\n{green('IMPROVEMENTS (FAIL → PASS):')}")
            for c in self.improvements:
                print(f"  ✅ {c.one_line()}")

        if self.score_drops:
            print(f"\n{yellow('SCORE DROPS (still passing, score fell):')}")
            for c in self.score_drops:
                print(f"  ⬇  {c.one_line()}")

        if self.score_gains:
            print(f"\n{green('SCORE GAINS (still passing, score rose):')}")
            for c in self.score_gains:
                print(f"  ⬆  {c.one_line()}")

        if not self.has_changes:
            print(f"\n{green('✓ No drift detected — all cases stable.')}")

        print()


class DriftDetector:
    """
    Compare two JSON eval reports and classify per-case drift.

    threshold: minimum absolute score change to classify as a drop/gain
               (default 0.1 = 10 percentage points)
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def compare(self, baseline_path: str, current_path: str) -> DriftReport:
        baseline = self._load(baseline_path)
        current = self._load(current_path)

        baseline_cases = {r["case_id"]: r for r in baseline.get("results", [])}
        current_cases  = {r["case_id"]: r for r in current.get("results", [])}

        all_ids = set(baseline_cases) | set(current_cases)

        report = DriftReport(
            baseline_timestamp=baseline.get("run_timestamp", "unknown"),
            current_timestamp=current.get("run_timestamp", "unknown"),
            provider=current.get("provider", baseline.get("provider", "unknown")),
            threshold=self.threshold,
        )

        for case_id in sorted(all_ids):
            b = baseline_cases.get(case_id)
            c = current_cases.get(case_id)

            if b is None:
                report.new_cases.append(CaseDrift(
                    case_id=case_id,
                    description=c.get("description", ""),
                    change_type="new",
                    baseline_passed=None,
                    current_passed=c.get("passed"),
                    baseline_score=None,
                    current_score=c.get("score"),
                ))
                continue

            if c is None:
                report.removed_cases.append(CaseDrift(
                    case_id=case_id,
                    description=b.get("description", ""),
                    change_type="removed",
                    baseline_passed=b.get("passed"),
                    current_passed=None,
                    baseline_score=b.get("score"),
                    current_score=None,
                ))
                continue

            b_pass = b.get("passed", False)
            c_pass = c.get("passed", False)
            b_score = b.get("score") or 0.0
            c_score = c.get("score") or 0.0
            delta = c_score - b_score
            desc = c.get("description") or b.get("description", "")

            drift = CaseDrift(
                case_id=case_id,
                description=desc,
                change_type="",        # set below
                baseline_passed=b_pass,
                current_passed=c_pass,
                baseline_score=b_score,
                current_score=c_score,
            )

            if b_pass and not c_pass:
                drift.change_type = "regression"
                report.regressions.append(drift)
            elif not b_pass and c_pass:
                drift.change_type = "improvement"
                report.improvements.append(drift)
            elif b_pass and c_pass and delta < -self.threshold:
                drift.change_type = "score_drop"
                report.score_drops.append(drift)
            elif b_pass and c_pass and delta > self.threshold:
                drift.change_type = "score_gain"
                report.score_gains.append(drift)
            else:
                drift.change_type = "stable"
                report.stable.append(drift)

        return report

    @staticmethod
    def _load(path: str) -> dict:
        with open(path) as f:
            return json.load(f)
