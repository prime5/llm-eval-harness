"""
Offline unit tests for Phase 3: AgentRunner, tool_call_match scorer, json_schema_match scorer.
No API calls — uses a MockProvider that returns scripted responses.
"""
import pytest
from evals.agent_runner import AgentRunner, AgentEvalResult, TurnResult, _normalise_tool
from evals.scorer import Scorer, ScorerResult
from providers.base import CompletionRequest, CompletionResponse


# ------------------------------------------------------------------ #
# Mock provider                                                        #
# ------------------------------------------------------------------ #

class MockProvider:
    """Returns scripted responses in order; loops if exhausted."""

    def __init__(self, responses: list):
        self._responses = responses
        self._index = 0

    @property
    def name(self):
        return "mock"

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        resp = self._responses[self._index % len(self._responses)]
        self._index += 1

        if isinstance(resp, dict):
            return CompletionResponse(
                text=resp.get("text", ""),
                model="mock-model",
                provider="mock",
                prompt_tokens=10,
                completion_tokens=5,
                latency_ms=1.0,
                tool_calls=resp.get("tool_calls"),
            )
        return CompletionResponse(
            text=resp,
            model="mock-model",
            provider="mock",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
        )


# ------------------------------------------------------------------ #
# Scorer: tool_call_match                                              #
# ------------------------------------------------------------------ #

class TestToolCallMatch:

    def test_passes_when_correct_function_called_no_args(self):
        tool_calls = [{"id": "1", "name": "get_weather", "arguments": {"location": "SF"}}]
        result = Scorer.tool_call_match(tool_calls, expected_function="get_weather")
        assert result.passed
        assert result.score == 1.0

    def test_fails_when_no_tool_called(self):
        result = Scorer.tool_call_match(None, expected_function="get_weather")
        assert not result.passed
        assert result.score == 0.0

    def test_fails_when_wrong_function_called(self):
        tool_calls = [{"id": "1", "name": "search_web", "arguments": {"query": "weather SF"}}]
        result = Scorer.tool_call_match(tool_calls, expected_function="get_weather")
        assert not result.passed
        assert result.score == 0.0

    def test_passes_with_exact_arg_match(self):
        tool_calls = [{"id": "1", "name": "get_weather", "arguments": {"location": "San Francisco"}}]
        result = Scorer.tool_call_match(
            tool_calls,
            expected_function="get_weather",
            expected_args={"location": "San Francisco"},
        )
        assert result.passed
        assert result.score == 1.0

    def test_passes_with_partial_substring_arg_match(self):
        # "San Francisco" contains "Francisco"
        tool_calls = [{"id": "1", "name": "get_weather", "arguments": {"location": "San Francisco, CA"}}]
        result = Scorer.tool_call_match(
            tool_calls,
            expected_function="get_weather",
            expected_args={"location": "San Francisco"},
        )
        assert result.passed

    def test_partial_score_when_some_args_missing(self):
        tool_calls = [{"id": "1", "name": "create_event", "arguments": {"title": "Meeting"}}]
        result = Scorer.tool_call_match(
            tool_calls,
            expected_function="create_event",
            expected_args={"title": "Meeting", "duration_minutes": "60"},
            require_all_args=False,
        )
        # Function was called, title matched, duration missing → partial score but >= 0.5
        assert result.score >= 0.5

    def test_fails_with_require_all_args_when_arg_missing(self):
        tool_calls = [{"id": "1", "name": "create_event", "arguments": {"title": "Meeting"}}]
        result = Scorer.tool_call_match(
            tool_calls,
            expected_function="create_event",
            expected_args={"title": "Meeting", "duration_minutes": "60"},
            require_all_args=True,
            threshold=1.0,
        )
        assert not result.passed

    def test_empty_tool_calls_list_fails(self):
        result = Scorer.tool_call_match([], expected_function="get_weather")
        assert not result.passed


# ------------------------------------------------------------------ #
# Scorer: json_schema_match                                            #
# ------------------------------------------------------------------ #

class TestJsonSchemaMatch:

    def test_valid_json_with_required_keys(self):
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        result = Scorer.json_schema_match('{"name": "Alice", "age": 30}', schema)
        assert result.passed
        assert result.score == 1.0

    def test_invalid_json_fails(self):
        result = Scorer.json_schema_match("not json at all", {"required": ["name"]})
        assert not result.passed
        assert "not valid JSON" in result.explanation

    def test_missing_required_key_fails(self):
        schema = {"required": ["name", "age"], "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        result = Scorer.json_schema_match('{"name": "Alice"}', schema)
        assert not result.passed
        assert "age" in result.explanation

    def test_wrong_type_fails(self):
        schema = {
            "required": ["count"],
            "properties": {"count": {"type": "integer"}},
        }
        result = Scorer.json_schema_match('{"count": "not-a-number"}', schema)
        assert not result.passed

    def test_no_required_keys_passes_valid_json(self):
        result = Scorer.json_schema_match('{"anything": true}', {})
        assert result.passed


# ------------------------------------------------------------------ #
# AgentRunner: tool call cases                                         #
# ------------------------------------------------------------------ #

class TestAgentRunnerToolCall:

    def _make_runner(self, mock_responses):
        provider = MockProvider(mock_responses)
        return AgentRunner(provider)

    def test_tool_call_case_passes_when_function_called(self):
        mock_resp = {
            "text": "",
            "tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": {"location": "San Francisco"}}],
        }
        runner = self._make_runner([mock_resp])
        case = {
            "id": "tc_test_001",
            "type": "tool_call",
            "description": "Test weather call",
            "prompt": "What is the weather in San Francisco?",
            "tools": [{"name": "get_weather", "description": "Get weather", "parameters": {
                "type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]
            }}],
            "scorers": [{"type": "tool_call_match", "expected_function": "get_weather",
                         "expected_args": {"location": "San Francisco"}}],
        }
        result = runner._run_case(case)
        assert isinstance(result, AgentEvalResult)
        assert result.passed
        assert result.case_type == "tool_call"
        assert len(result.turns) == 1

    def test_tool_call_case_fails_when_no_function_called(self):
        runner = self._make_runner([{"text": "The weather is nice today.", "tool_calls": None}])
        case = {
            "id": "tc_test_002",
            "type": "tool_call",
            "description": "Should call tool but doesn't",
            "prompt": "What is the weather in London?",
            "tools": [{"name": "get_weather", "description": "Get weather", "parameters": {
                "type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]
            }}],
            "scorers": [{"type": "tool_call_match", "expected_function": "get_weather"}],
        }
        result = runner._run_case(case)
        assert not result.passed

    def test_no_tool_call_scorer_passes_when_no_function_called(self):
        runner = self._make_runner([{"text": "Paris is the capital of France.", "tool_calls": None}])
        case = {
            "id": "tc_test_003",
            "type": "tool_call",
            "description": "Direct answer without tool",
            "prompt": "What is the capital of France?",
            "tools": [{"name": "search_web", "description": "Search", "parameters": {
                "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]
            }}],
            "scorers": [
                {"type": "no_tool_call"},
                {"type": "contains_keywords", "keywords": ["Paris"], "require_all": True},
            ],
        }
        result = runner._run_case(case)
        assert result.passed

    def test_error_handling_returns_failed_result(self):
        class BrokenProvider:
            name = "broken"
            def complete(self, req):
                raise RuntimeError("API unavailable")

        runner = AgentRunner(BrokenProvider())
        case = {"id": "err_001", "type": "tool_call", "description": "Error case",
                "prompt": "Hello", "scorers": []}
        result = runner._run_case(case)
        assert not result.passed
        assert result.error is not None


# ------------------------------------------------------------------ #
# AgentRunner: multi-turn cases                                        #
# ------------------------------------------------------------------ #

class TestAgentRunnerMultiTurn:

    def _make_runner(self, mock_responses):
        provider = MockProvider(mock_responses)
        return AgentRunner(provider)

    def test_multi_turn_builds_message_history(self):
        captured_requests = []

        class CapturingProvider:
            name = "capturing"
            def complete(self, request: CompletionRequest) -> CompletionResponse:
                captured_requests.append(request)
                return CompletionResponse(
                    text="Acknowledged.", model="mock", provider="mock",
                    prompt_tokens=5, completion_tokens=3, latency_ms=1.0,
                )

        runner = AgentRunner(CapturingProvider())
        case = {
            "id": "mt_hist_001",
            "type": "multi_turn",
            "description": "History building",
            "turns": [
                {"user": "Turn 1 message.", "scorers": []},
                {"user": "Turn 2 message.", "scorers": []},
            ],
        }
        runner._run_case(case)
        # Turn 2 request should include all prior messages in history
        assert len(captured_requests) == 2
        turn2_messages = captured_requests[1].messages
        assert turn2_messages is not None
        user_contents = [m["content"] for m in turn2_messages if m["role"] == "user"]
        assert "Turn 1 message." in user_contents
        assert "Turn 2 message." in user_contents

    def test_multi_turn_all_turns_must_pass(self):
        # Turn 1 passes (contains "Alice"), turn 2 fails (missing "software")
        runner = self._make_runner(["My name is Alice, I'm an engineer.", "I like cooking."])
        case = {
            "id": "mt_test_001",
            "type": "multi_turn",
            "description": "All turns must pass",
            "turns": [
                {"user": "What is my name?",
                 "scorers": [{"type": "contains_keywords", "keywords": ["Alice"], "require_all": True}]},
                {"user": "What do I do?",
                 "scorers": [{"type": "contains_keywords", "keywords": ["software"], "require_all": True}]},
            ],
        }
        result = runner._run_case(case)
        # min aggregation — turn 2 fails, so overall fails
        assert not result.passed
        assert result.turns[0].scorer_result.passed
        assert not result.turns[1].scorer_result.passed

    def test_multi_turn_passes_when_all_turns_pass(self):
        runner = self._make_runner(["My name is Alice.", "I am a software engineer."])
        case = {
            "id": "mt_test_002",
            "type": "multi_turn",
            "description": "All turns pass",
            "turns": [
                {"user": "What is my name?",
                 "scorers": [{"type": "contains_keywords", "keywords": ["Alice"], "require_all": True}]},
                {"user": "What do I do?",
                 "scorers": [{"type": "contains_keywords", "keywords": ["engineer"], "require_all": True}]},
            ],
        }
        result = runner._run_case(case)
        assert result.passed
        assert len(result.turns) == 2

    def test_multi_turn_no_scorers_passes_by_default(self):
        runner = self._make_runner(["Hello!", "Sure!"])
        case = {
            "id": "mt_test_003",
            "type": "multi_turn",
            "description": "No scorers = trivially passes",
            "turns": [
                {"user": "Hi.", "scorers": []},
                {"user": "OK.", "scorers": []},
            ],
        }
        result = runner._run_case(case)
        assert result.passed

    def test_agent_eval_result_properties(self):
        runner = self._make_runner(["Response 1.", "Response 2."])
        case = {
            "id": "mt_prop_001",
            "type": "multi_turn",
            "description": "Property shim test",
            "tags": ["smoke"],
            "turns": [
                {"user": "First question.", "scorers": []},
                {"user": "Second question.", "scorers": []},
            ],
        }
        result = runner._run_case(case)
        # Duck-type shim properties must be accessible
        assert result.prompt == "First question."
        assert result.response == "Response 2."
        assert result.latency_ms >= 0
        assert result.variance_report is None
        assert result.tags == ["smoke"]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

class TestNormaliseTool:

    def test_shorthand_converted_to_openai_format(self):
        tool = {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        }
        result = _normalise_tool(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"

    def test_already_openai_format_passes_through(self):
        tool = {
            "type": "function",
            "function": {"name": "search_web", "description": "...", "parameters": {}},
        }
        result = _normalise_tool(tool)
        assert result == tool
