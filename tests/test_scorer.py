"""
Unit tests for the Scorer module.
These run offline — no LLM API calls needed.
pytest tests/test_scorer.py
"""
import pytest
from evals.scorer import Scorer


class TestExactMatch:
    def test_exact_match_passes(self):
        r = Scorer.exact_match("Paris", "Paris")
        assert r.passed
        assert r.score == 1.0

    def test_exact_match_fails(self):
        r = Scorer.exact_match("London", "Paris")
        assert not r.passed
        assert r.score == 0.0

    def test_case_insensitive_default(self):
        r = Scorer.exact_match("paris", "Paris")
        assert r.passed

    def test_case_sensitive_fails(self):
        r = Scorer.exact_match("paris", "Paris", case_sensitive=True)
        assert not r.passed


class TestContainsKeywords:
    def test_all_keywords_present(self):
        r = Scorer.contains_keywords("Python is a programming language", ["Python", "language"])
        assert r.passed
        assert r.score == 1.0

    def test_missing_keyword_with_require_all(self):
        r = Scorer.contains_keywords("Python is great", ["Python", "Java"], require_all=True)
        assert not r.passed
        assert r.score == 0.0

    def test_partial_match_without_require_all(self):
        r = Scorer.contains_keywords("Python is great", ["Python", "Java"],
                                      require_all=False, threshold=0.4)
        assert r.passed
        assert r.score == 0.5

    def test_case_insensitive_match(self):
        r = Scorer.contains_keywords("PARIS is beautiful", ["paris"])
        assert r.passed


class TestRegexMatch:
    def test_regex_matches(self):
        r = Scorer.regex_match('{"name": "Alice"}', r'\{.*"name".*\}')
        assert r.passed

    def test_regex_no_match(self):
        r = Scorer.regex_match("Hello world", r'\d+')
        assert not r.passed

    def test_regex_number_extraction(self):
        r = Scorer.regex_match("The answer is 42.", r'\b42\b')
        assert r.passed


class TestLengthCheck:
    def test_within_range(self):
        r = Scorer.length_check("one two three four five", min_words=3, max_words=10)
        assert r.passed

    def test_too_short(self):
        r = Scorer.length_check("hi", min_words=5)
        assert not r.passed

    def test_too_long(self):
        r = Scorer.length_check(" ".join(["word"] * 50), max_words=10)
        assert not r.passed


class TestNoForbiddenContent:
    def test_clean_response(self):
        r = Scorer.no_forbidden_content("Sure, here is your answer.", ["I cannot", "I'm unable"])
        assert r.passed

    def test_forbidden_found(self):
        r = Scorer.no_forbidden_content("I cannot help with that.", ["I cannot"])
        assert not r.passed
        assert "I cannot" in r.details["found"]


class TestAggregate:
    def test_mean_aggregation(self):
        r1 = Scorer.exact_match("Paris", "Paris")      # score 1.0
        r2 = Scorer.exact_match("London", "Paris")     # score 0.0
        agg = Scorer.aggregate([r1, r2], strategy="mean")
        assert agg.score == pytest.approx(0.5)

    def test_min_aggregation(self):
        r1 = Scorer.exact_match("Paris", "Paris")
        r2 = Scorer.length_check("hello world", min_words=1)
        agg = Scorer.aggregate([r1, r2], strategy="min")
        assert agg.score == 1.0

    def test_all_pass_aggregation(self):
        r1 = Scorer.exact_match("Paris", "Paris")
        r2 = Scorer.exact_match("London", "Paris")
        agg = Scorer.aggregate([r1, r2], strategy="all_pass")
        assert not agg.passed

    def test_empty_results(self):
        agg = Scorer.aggregate([])
        assert not agg.passed
