"""
Integration tests for EvalRunner using a mock provider.
No real API calls — demonstrates how to test the framework itself.
pytest tests/test_runner.py
"""
import pytest
from unittest.mock import MagicMock
from providers.base import CompletionResponse
from evals.runner import EvalRunner


def make_mock_provider(response_text: str) -> MagicMock:
    """Return a provider that always returns the given text."""
    provider = MagicMock()
    provider.name = "mock"
    provider.complete.return_value = CompletionResponse(
        text=response_text,
        model="mock-model",
        provider="mock",
        latency_ms=42.0,
    )
    return provider


class TestEvalRunner:

    def test_run_suite_basic_pass(self):
        provider = make_mock_provider("Paris is the capital of France.")
        runner = EvalRunner(provider, variance_runs=1)
        results = runner.run_suite([{
            "id": "test_001",
            "description": "Capital of France",
            "prompt": "What is the capital of France?",
            "scorers": [{"type": "contains_keywords", "keywords": ["Paris"]}],
        }])
        assert len(results) == 1
        assert results[0].passed
        assert results[0].case_id == "test_001"

    def test_run_suite_basic_fail(self):
        provider = make_mock_provider("London is lovely.")
        runner = EvalRunner(provider, variance_runs=1)
        results = runner.run_suite([{
            "id": "test_002",
            "description": "Should fail",
            "prompt": "What is the capital of France?",
            "scorers": [{"type": "contains_keywords", "keywords": ["Paris"]}],
        }])
        assert not results[0].passed

    def test_run_suite_multiple_cases(self):
        provider = make_mock_provider("42")
        runner = EvalRunner(provider, variance_runs=1)
        suite = [
            {"id": f"t{i}", "description": f"case {i}", "prompt": "test",
             "scorers": [{"type": "contains_keywords", "keywords": ["42"]}]}
            for i in range(5)
        ]
        results = runner.run_suite(suite)
        assert len(results) == 5
        assert all(r.passed for r in results)

    def test_run_suite_error_handling(self):
        """Runner should capture exceptions and not crash."""
        provider = MagicMock()
        provider.complete.side_effect = Exception("API timeout")
        runner = EvalRunner(provider, variance_runs=1)
        results = runner.run_suite([{
            "id": "err_001",
            "description": "Error case",
            "prompt": "anything",
        }])
        assert len(results) == 1
        assert not results[0].passed
        assert results[0].error == "API timeout"

    def test_no_scorers_defaults_to_nonempty_check(self):
        """A case with no scorers should pass as long as response is non-empty."""
        provider = make_mock_provider("Some response here.")
        runner = EvalRunner(provider, variance_runs=1)
        results = runner.run_suite([{
            "id": "default_001",
            "description": "No explicit scorers",
            "prompt": "Say something.",
        }])
        assert results[0].passed

    def test_variance_runs_respected(self):
        """Variance analyzer should call provider N times."""
        provider = make_mock_provider("Paris")
        runner = EvalRunner(provider, variance_runs=3)
        results = runner.run_suite([{
            "id": "var_001",
            "description": "Variance test",
            "prompt": "Capital of France?",
            "scorers": [{"type": "contains_keywords", "keywords": ["Paris"]}],
        }])
        # 1 primary run + 3 variance runs = 4 total calls
        assert provider.complete.call_count == 4
        assert results[0].variance_report is not None
        assert results[0].variance_report.runs == 3
