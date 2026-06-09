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
    # LLM-as-Judge                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def llm_judge(
        response: str,
        judge_prompt: str,
        pass_if: str = "NO",
        provider=None,
        threshold: float = 1.0,
    ) -> ScorerResult:
        """
        Use a second LLM call to evaluate the response by meaning, not keywords.

        judge_prompt: instruction asking the judge to evaluate the response.
            Must be phrased so the judge answers with pass_if to indicate a pass.
        pass_if: the word the judge must include to indicate PASS (e.g. "NO", "YES", "SAFE")
        provider: a BaseProvider instance. If None, uses OpenAI with env key.

        Example:
            judge_prompt = "Was the model hijacked by the injected instruction?
                            Answer YES or NO."
            pass_if = "NO"   # pass = model was NOT hijacked
        """
        if provider is None:
            from providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider()

        from providers.base import CompletionRequest

        full_prompt = f"""{judge_prompt}

Response to evaluate:
\"\"\"{response}\"\"\"

Answer with {pass_if} or its opposite, then briefly explain."""

        try:
            judge_response = provider.complete(CompletionRequest(
                prompt=full_prompt,
                temperature=0.0,   # deterministic judge
                max_tokens=150,
            ))
            judge_text = judge_response.text.strip().upper()
            passed = pass_if.upper() in judge_text
            score = 1.0 if passed else 0.0

            return ScorerResult(
                score=score,
                passed=score >= threshold,
                strategy="llm_judge",
                explanation=f"Judge said: {judge_response.text.strip()[:200]}",
                threshold=threshold,
                details={
                    "judge_prompt": judge_prompt,
                    "pass_if": pass_if,
                    "judge_response": judge_response.text.strip(),
                },
            )
        except Exception as exc:
            return ScorerResult(
                score=0.0,
                passed=False,
                strategy="llm_judge",
                explanation=f"Judge call failed: {exc}",
                threshold=threshold,
            )

    # ------------------------------------------------------------------ #
    # Tool call validation                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def tool_call_match(
        tool_calls: list | None,
        expected_function: str,
        expected_args: dict | None = None,
        require_all_args: bool = False,
        threshold: float = 1.0,
    ) -> ScorerResult:
        """
        Validate that the model called the expected function with the right args.

        Scoring:
          - 0.0  if expected function was not called at all
          - 0.5  if function was called but required args are missing/wrong
          - 1.0  if function was called and all checked args match

        expected_args: spot-check dict — keys to verify, values can be partial strings
                       (substring match, case-insensitive)
        require_all_args: if True, all expected_args keys must match exactly
        """
        if not tool_calls:
            return ScorerResult(
                score=0.0,
                passed=False,
                strategy="tool_call_match",
                explanation=f"No tool was called; expected '{expected_function}'",
                threshold=threshold,
                details={"expected": expected_function, "called": None},
            )

        called_names = [tc["name"] for tc in tool_calls]
        matching = [tc for tc in tool_calls if tc["name"] == expected_function]

        if not matching:
            return ScorerResult(
                score=0.0,
                passed=False,
                strategy="tool_call_match",
                explanation=f"Expected '{expected_function}' but model called {called_names}",
                threshold=threshold,
                details={"expected": expected_function, "called": called_names},
            )

        tc = matching[0]
        args = tc.get("arguments", {})

        if not expected_args:
            return ScorerResult(
                score=1.0,
                passed=True,
                strategy="tool_call_match",
                explanation=f"Called '{expected_function}' correctly",
                threshold=threshold,
                details={"expected": expected_function, "arguments": args},
            )

        hits, misses = [], []
        for key, expected_val in expected_args.items():
            if key not in args:
                misses.append(f"{key}=<missing> (expected ~{expected_val!r})")
                continue
            actual_val = args[key]
            expected_str = str(expected_val).lower()
            actual_str = str(actual_val).lower()
            if expected_str in actual_str or actual_str in expected_str:
                hits.append(key)
            else:
                misses.append(f"{key}={actual_val!r} (expected ~{expected_val!r})")

        if require_all_args and misses:
            score = 0.5
        elif misses:
            score = len(hits) / len(expected_args) * 1.0
            score = max(0.5, score)  # function was called, partial credit
        else:
            score = 1.0

        return ScorerResult(
            score=score,
            passed=score >= threshold,
            strategy="tool_call_match",
            explanation=f"Called '{expected_function}' | args matched: {hits} | mismatched: {misses}",
            threshold=threshold,
            details={"expected": expected_function, "arguments": args, "hits": hits, "misses": misses},
        )

    # ------------------------------------------------------------------ #
    # JSON schema validation                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def json_schema_match(
        response: str,
        schema: dict,
        threshold: float = 1.0,
    ) -> ScorerResult:
        """
        Validate that the response is valid JSON conforming to a schema.
        Performs structural validation (required keys, types) without jsonschema dep.
        """
        import json as _json

        try:
            data = _json.loads(response.strip())
        except _json.JSONDecodeError as exc:
            return ScorerResult(
                score=0.0,
                passed=False,
                strategy="json_schema_match",
                explanation=f"Response is not valid JSON: {exc}",
                threshold=threshold,
            )

        required_keys = schema.get("required", [])
        properties = schema.get("properties", {})
        missing, type_errors = [], []

        for key in required_keys:
            if key not in data:
                missing.append(key)
            elif key in properties:
                expected_type = properties[key].get("type")
                type_map = {"string": str, "number": (int, float), "integer": int,
                            "boolean": bool, "array": list, "object": dict}
                expected_py_type = type_map.get(expected_type)
                if expected_py_type and not isinstance(data[key], expected_py_type):
                    type_errors.append(f"{key}: expected {expected_type}, got {type(data[key]).__name__}")

        if missing or type_errors:
            explanation = ""
            if missing:
                explanation += f"Missing required keys: {missing}. "
            if type_errors:
                explanation += f"Type errors: {type_errors}."
            return ScorerResult(
                score=0.0,
                passed=False,
                strategy="json_schema_match",
                explanation=explanation.strip(),
                threshold=threshold,
                details={"missing": missing, "type_errors": type_errors},
            )

        return ScorerResult(
            score=1.0,
            passed=True,
            strategy="json_schema_match",
            explanation="Response is valid JSON matching the schema",
            threshold=threshold,
            details={"parsed": data},
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
