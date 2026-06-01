"""
Variance / determinism analysis.

Runs the same prompt N times and measures how consistent
the model's responses are. Critical for LLM testing because
models are non-deterministic — a single run tells you nothing
about reliability.

Metrics produced:
  - pass_rate       : fraction of runs that passed scoring
  - pass_at_k       : did at least 1 of k runs pass? (OpenAI's pass@k)
  - response_variance: how different are the raw response texts?
  - latency_stats   : p50 / p95 / p99 latency across runs
"""
from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from providers.base import CompletionRequest, CompletionResponse
    from .scorer import ScorerResult


@dataclass
class VarianceReport:
    case_id: str
    runs: int
    responses: list[str]
    scorer_results: list["ScorerResult"]
    latencies_ms: list[float]

    # Computed after init
    pass_rate: float = 0.0
    pass_at_k: bool = False
    unique_response_ratio: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    verdict: str = ""

    def __post_init__(self):
        if self.scorer_results:
            passed = sum(1 for r in self.scorer_results if r.passed)
            self.pass_rate = passed / len(self.scorer_results)
            self.pass_at_k = self.pass_rate > 0

        if self.responses:
            unique = len(set(r.strip().lower() for r in self.responses))
            self.unique_response_ratio = unique / len(self.responses)

        if self.latencies_ms:
            sorted_lat = sorted(self.latencies_ms)
            n = len(sorted_lat)
            self.latency_p50 = sorted_lat[int(n * 0.50)]
            self.latency_p95 = sorted_lat[min(int(n * 0.95), n - 1)]
            self.latency_p99 = sorted_lat[min(int(n * 0.99), n - 1)]

        # Overall verdict
        if self.pass_rate >= 0.8:
            self.verdict = "STABLE"
        elif self.pass_rate >= 0.5:
            self.verdict = "FLAKY"
        else:
            self.verdict = "UNSTABLE"

    def summary(self) -> dict:
        return {
            "case_id": self.case_id,
            "runs": self.runs,
            "pass_rate": f"{self.pass_rate:.0%}",
            "pass_at_k": self.pass_at_k,
            "unique_response_ratio": f"{self.unique_response_ratio:.0%}",
            "latency_p50_ms": round(self.latency_p50, 1),
            "latency_p95_ms": round(self.latency_p95, 1),
            "verdict": self.verdict,
        }


class VarianceAnalyzer:
    """
    Runs a prompt N times and aggregates results.

    Usage:
        analyzer = VarianceAnalyzer(provider, runs=5)
        report = analyzer.analyze(request, scorer_fn, case_id="q1")
    """

    def __init__(self, provider, runs: int = 5):
        self._provider = provider
        self._runs = runs

    def analyze(
        self,
        request: "CompletionRequest",
        scorer_fn,
        case_id: str = "unknown",
    ) -> VarianceReport:
        """
        Execute the request N times, score each response,
        return a VarianceReport.

        scorer_fn: callable(response_text) -> ScorerResult
        """
        from providers.base import CompletionRequest as CR

        responses = []
        scorer_results = []
        latencies = []

        for _ in range(self._runs):
            resp = self._provider.complete(request)
            responses.append(resp.text)
            latencies.append(resp.latency_ms)
            scorer_results.append(scorer_fn(resp.text))

        return VarianceReport(
            case_id=case_id,
            runs=self._runs,
            responses=responses,
            scorer_results=scorer_results,
            latencies_ms=latencies,
        )
