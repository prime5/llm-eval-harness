"""
Core eval runner.

Loads test cases from YAML, drives the provider, applies scorers,
and returns structured EvalResult objects ready for reporting.

YAML test case schema:
  id: unique string
  description: human readable
  prompt: the user message
  system_prompt: (optional) system message
  scorers:
    - type: contains_keywords
      keywords: [python, list]
      require_all: true
    - type: length_check
      min_words: 5
      max_words: 200
    - type: no_forbidden_content
      forbidden: [sorry, I cannot]
  variance_runs: 3   # optional, overrides global setting
  tags: [smoke, factual]
"""
from __future__ import annotations
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from config.settings import VARIANCE_RUNS
from providers.base import CompletionRequest
from .scorer import Scorer, ScorerResult
from .variance import VarianceAnalyzer, VarianceReport


@dataclass
class EvalResult:
    case_id: str
    description: str
    prompt: str
    response: str
    scorer_result: ScorerResult
    variance_report: Optional[VarianceReport]
    latency_ms: float
    tags: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.scorer_result.passed if self.scorer_result else False

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "description": self.description,
            "passed": self.passed,
            "score": round(self.scorer_result.score, 3) if self.scorer_result else None,
            "scorer_explanation": self.scorer_result.explanation if self.scorer_result else None,
            "response_preview": self.response[:200] + "..." if len(self.response) > 200 else self.response,
            "latency_ms": round(self.latency_ms, 1),
            "variance": self.variance_report.summary() if self.variance_report else None,
            "tags": self.tags,
            "error": self.error,
        }


class EvalRunner:
    """
    Orchestrates loading test cases, running them against a provider,
    and collecting results.
    """

    def __init__(self, provider, variance_runs: int = VARIANCE_RUNS):
        self._provider = provider
        self._variance_runs = variance_runs

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run_file(self, path: str | Path) -> list[EvalResult]:
        """Load a YAML test suite and run all cases."""
        cases = self._load_yaml(path)
        return [self._run_case(c) for c in cases]

    def run_suite(self, suite: list[dict]) -> list[EvalResult]:
        """Run test cases provided as a list of dicts (for programmatic use)."""
        return [self._run_case(c) for c in suite]

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _load_yaml(self, path: str | Path) -> list[dict]:
        with open(path) as f:
            data = yaml.safe_load(f)
        # Support both a bare list and a dict with a 'cases' key
        if isinstance(data, list):
            return data
        return data.get("cases", [])

    def _run_case(self, case: dict) -> EvalResult:
        case_id = case.get("id", "unknown")
        description = case.get("description", "")
        prompt = case.get("prompt", "")
        system_prompt = case.get("system_prompt")
        tags = case.get("tags", [])
        variance_runs = case.get("variance_runs", self._variance_runs)

        request = CompletionRequest(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        try:
            # Primary run
            start = time.time()
            response = self._provider.complete(request)
            latency_ms = (time.time() - start) * 1000

            # Build scorer from YAML spec
            scorer_fn = self._build_scorer(case.get("scorers", []))
            primary_result = scorer_fn(response.text)

            # Variance analysis (only if variance_runs > 1)
            variance_report = None
            if variance_runs > 1:
                analyzer = VarianceAnalyzer(self._provider, runs=variance_runs)
                variance_report = analyzer.analyze(request, scorer_fn, case_id=case_id)

            return EvalResult(
                case_id=case_id,
                description=description,
                prompt=prompt,
                response=response.text,
                scorer_result=primary_result,
                variance_report=variance_report,
                latency_ms=latency_ms,
                tags=tags,
            )

        except Exception as exc:
            return EvalResult(
                case_id=case_id,
                description=description,
                prompt=prompt,
                response="",
                scorer_result=ScorerResult(score=0.0, passed=False,
                                           strategy="error", explanation=str(exc)),
                variance_report=None,
                latency_ms=0.0,
                tags=tags,
                error=str(exc),
            )

    def _build_scorer(self, scorer_specs: list[dict]):
        """
        Convert YAML scorer specs into a callable that scores a response.
        Multiple scorers are aggregated with 'mean' strategy by default.
        """
        if not scorer_specs:
            # Default: just check response is non-empty
            return lambda text: Scorer.length_check(text, min_words=1)

        def scorer_fn(text: str) -> ScorerResult:
            results = []
            for spec in scorer_specs:
                stype = spec.get("type", "")
                agg = spec.get("aggregate", "mean")

                if stype == "exact_match":
                    results.append(Scorer.exact_match(
                        text,
                        expected=spec["expected"],
                        case_sensitive=spec.get("case_sensitive", False),
                        threshold=spec.get("threshold", 1.0),
                    ))
                elif stype == "contains_keywords":
                    results.append(Scorer.contains_keywords(
                        text,
                        keywords=spec["keywords"],
                        require_all=spec.get("require_all", True),
                        threshold=spec.get("threshold", 0.5),
                    ))
                elif stype == "regex_match":
                    results.append(Scorer.regex_match(
                        text,
                        pattern=spec["pattern"],
                        threshold=spec.get("threshold", 1.0),
                    ))
                elif stype == "length_check":
                    results.append(Scorer.length_check(
                        text,
                        min_words=spec.get("min_words", 0),
                        max_words=spec.get("max_words", 10_000),
                        threshold=spec.get("threshold", 1.0),
                    ))
                elif stype == "no_forbidden_content":
                    results.append(Scorer.no_forbidden_content(
                        text,
                        forbidden=spec["forbidden"],
                        threshold=spec.get("threshold", 1.0),
                    ))

            if len(results) == 1:
                return results[0]
            return Scorer.aggregate(results, strategy=spec.get("aggregate", "mean"))

        return scorer_fn
