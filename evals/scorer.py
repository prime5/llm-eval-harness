"""
Scoring strategies for LLM responses.

Each scorer takes a response string and test-case metadata,
returns a ScorerResult with a 0.0–1.0 score and explanation.

Design principle: scorers are composable — a test case can apply
multiple scorers and aggregate results.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScorerResult:
    score: float                  # 0.0 (fail) to 1.0 (perfect)
    passed: bool
    strategy: str
    explanation: str
    threshold: float = 0.5
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        self.passed = self.score >= self.threshold


class Scorer:
    """
    Collection of scoring strategies.
    All methods are static so they can be used without instantiation.
    """

    # ------------------------------------------------------------------ #
    # Exact match                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def exact_match(
        response: str,
        expected: str,
        case_sensitive: bool = False,
        threshold: float = 1.0,
    ) -> ScorerResult:
        """Response must exactly equal expected (after stripping whitespace)."""
        r = response.strip()
        e = expected.strip()
        if not case_sensitive:
            r, e = r.lower(), e.lower()
        score = 1.0 if r == e else 0.0
        return ScorerResult(
            score=score,
            passed=score >= threshold,
            strategy="exact_match",
            explanation=f"Expected: '{expected}' | Got: '{response.strip()}'",
            threshold=threshold,
        )

    # ------------------------------------------------------------------ #
    # Contains keyword                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def contains_keywords(
        response: str,
        keywords: list[str],
        require_all: bool = True,
        case_sensitive: bool = False,
        threshold: float = 0.5,
    ) -> ScorerResult:
        """Score based on how many required keywords appear in response."""
        text = response if case_sensitive else response.lower()
        hits = []
        misses = []
        for kw in keywords:
            needle = kw if case_sensitive else kw.lower()
            (hits if needle in text else misses).append(kw)

        score = len(hits) / len(keywords) if keywords else 1.0
        if require_all and misses:
            score = 0.0

        return ScorerResult(
            score=score,
            passed=score >= threshold,
            strategy="contains_keywords",
            explanation=f"Found: {hits} | Missing: {misses}",
            threshold=threshold,
            details={"hits": hits, "misses": misses},
        )

    # ------------------------------------------------------------------ #
    # Regex match                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def regex_match(
        response: str,
        pattern: str,
        threshold: float = 1.0,
    ) -> ScorerResult:
        """Response must match the given regex pattern."""
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        score = 1.0 if match else 0.0
        return ScorerResult(
            score=score,
            passed=score >= threshold,
            strategy="regex_match",
            explanation=f"Pattern '{pattern}' {'matched' if match else 'did not match'}",
            threshold=threshold,
            details={"pattern": pattern, "match": match.group(0) if match else None},
        )

    # ------------------------------------------------------------------ #
    # Length guard                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def length_check(
        response: str,
        min_words: int = 0,
        max_words: int = 10_000,
        threshold: float = 1.0,
    ) -> ScorerResult:
        """Validate response falls within a word-count range."""
        count = len(response.split())
        in_range = min_words <= count <= max_words
        score = 1.0 if in_range else 0.0
        return ScorerResult(
            score=score,
            passed=score >= threshold,
            strategy="length_check",
            explanation=f"Word count {count} | allowed [{min_words}, {max_words}]",
            threshold=threshold,
            details={"word_count": count},
        )

    # ------------------------------------------------------------------ #
    # Forbidden content                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def no_forbidden_content(
        response: str,
        forbidden: list[str],
        case_sensitive: bool = False,
        threshold: float = 1.0,
    ) -> ScorerResult:
        """Response must NOT contain any forbidden strings (safety baseline)."""
        text = response if case_sensitive else response.lower()
        found = [f for f in forbidden if (f if case_sensitive else f.lower()) in text]
        score = 0.0 if found else 1.0
        return ScorerResult(
            score=score,
            passed=score >= threshold,
            strategy="no_forbidden_content",
            explanation=f"Forbidden terms found: {found}" if found else "No forbidden content detected",
            threshold=threshold,
            details={"found": found},
        )

    # ------------------------------------------------------------------ #
    # Composite                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def aggregate(results: list[ScorerResult], strategy: str = "mean") -> ScorerResult:
        """
        Combine multiple scorer results into one.
        strategy: 'mean' | 'min' (strictest) | 'all_pass'
        """
        if not results:
            return ScorerResult(score=0.0, passed=False, strategy="aggregate",
                                explanation="No results to aggregate")
        scores = [r.score for r in results]
        names = [r.strategy for r in results]

        if strategy == "min":
            final = min(scores)
        elif strategy == "all_pass":
            final = 1.0 if all(r.passed for r in results) else 0.0
        else:  # mean
            final = sum(scores) / len(scores)

        threshold = min(r.threshold for r in results)
        return ScorerResult(
            score=final,
            passed=final >= threshold,
            strategy=f"aggregate({strategy})[{', '.join(names)}]",
            explanation=f"Scores: {dict(zip(names, scores))} → {strategy}={final:.2f}",
            threshold=threshold,
            details={"individual": [r.__dict__ for r in results]},
        )
