"""
Agent runner for multi-turn conversation and tool call eval cases.

Extends the single-turn EvalRunner with two new case types:

  type: tool_call   — sends a prompt with function definitions, validates
                      that the model called the right function with correct args.

  type: multi_turn  — drives a conversation across multiple user turns,
                      maintaining message history and scoring each assistant reply.

YAML schema for tool_call:
  - id: tc_001
    type: tool_call
    description: Weather query triggers get_weather
    prompt: "What's the weather in San Francisco?"
    tools:
      - name: get_weather
        description: "Get current weather for a city"
        parameters:
          type: object
          properties:
            location: {type: string}
          required: [location]
    scorers:
      - type: tool_call_match
        expected_function: get_weather
        expected_args:
          location: San Francisco

YAML schema for multi_turn:
  - id: mt_001
    type: multi_turn
    description: Name retention across turns
    system_prompt: "You are a helpful assistant."
    turns:
      - user: "My name is Alice."
        scorers: []
      - user: "What is my name?"
        scorers:
          - type: contains_keywords
            keywords: [Alice]
"""
from __future__ import annotations

import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from providers.base import CompletionRequest
from .scorer import Scorer, ScorerResult


@dataclass
class TurnResult:
    turn_index: int
    user_message: str
    response: str
    tool_calls: Optional[list]
    scorer_result: Optional[ScorerResult]
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class AgentEvalResult:
    """
    Result of an agent eval case (tool_call or multi_turn).
    Duck-types EvalResult's key surface so CLI and reporters work transparently.
    """
    case_id: str
    description: str
    case_type: str                        # "tool_call" | "multi_turn"
    turns: list[TurnResult]
    scorer_result: ScorerResult
    tags: list[str] = field(default_factory=list)
    error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # EvalResult compatibility shim                                        #
    # ------------------------------------------------------------------ #

    @property
    def passed(self) -> bool:
        return self.scorer_result.passed if self.scorer_result else False

    @property
    def prompt(self) -> str:
        return self.turns[0].user_message if self.turns else ""

    @property
    def response(self) -> str:
        return self.turns[-1].response if self.turns else ""

    @property
    def latency_ms(self) -> float:
        return sum(t.latency_ms for t in self.turns)

    @property
    def prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.turns)

    @property
    def completion_tokens(self) -> int:
        return sum(t.completion_tokens for t in self.turns)

    @property
    def variance_report(self):
        return None

    def to_dict(self) -> dict:
        turn_dicts = []
        for t in self.turns:
            d = {
                "turn": t.turn_index + 1,
                "user": t.user_message,
                "response": t.response[:200] + "..." if len(t.response) > 200 else t.response,
                "latency_ms": round(t.latency_ms, 1),
            }
            if t.tool_calls:
                d["tool_calls"] = t.tool_calls
            if t.scorer_result:
                d["passed"] = t.scorer_result.passed
                d["score"] = round(t.scorer_result.score, 3)
                d["scorer_explanation"] = t.scorer_result.explanation
            turn_dicts.append(d)

        return {
            "case_id": self.case_id,
            "description": self.description,
            "case_type": self.case_type,
            "passed": self.passed,
            "score": round(self.scorer_result.score, 3) if self.scorer_result else None,
            "scorer_explanation": self.scorer_result.explanation if self.scorer_result else None,
            "turns": turn_dicts,
            "total_latency_ms": round(self.latency_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tags": self.tags,
            "error": self.error,
        }


class AgentRunner:
    """
    Runs tool_call and multi_turn eval cases against a provider.

    Usage:
        runner = AgentRunner(provider)
        results = runner.run_file("test_cases/tool_calling.yaml")
    """

    def __init__(self, provider):
        self._provider = provider

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run_file(self, path: str | Path) -> list[AgentEvalResult]:
        cases = self._load_yaml(path)
        return [self._run_case(c) for c in cases]

    # ------------------------------------------------------------------ #
    # Routing                                                              #
    # ------------------------------------------------------------------ #

    def _load_yaml(self, path: str | Path) -> list[dict]:
        with open(path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            return data
        return data.get("cases", [])

    def _run_case(self, case: dict) -> AgentEvalResult:
        case_type = case.get("type", "tool_call")
        try:
            if case_type == "multi_turn":
                return self._run_multi_turn(case)
            else:
                return self._run_tool_call(case)
        except Exception as exc:
            return AgentEvalResult(
                case_id=case.get("id", "unknown"),
                description=case.get("description", ""),
                case_type=case_type,
                turns=[],
                scorer_result=ScorerResult(
                    score=0.0, passed=False, strategy="error",
                    explanation=str(exc),
                ),
                tags=case.get("tags", []),
                error=str(exc),
            )

    # ------------------------------------------------------------------ #
    # Tool call case                                                       #
    # ------------------------------------------------------------------ #

    def _run_tool_call(self, case: dict) -> AgentEvalResult:
        case_id = case.get("id", "unknown")
        description = case.get("description", "")
        prompt = case.get("prompt", "")
        system_prompt = case.get("system_prompt")
        tags = case.get("tags", [])

        # Normalise tool definitions to OpenAI format
        raw_tools = case.get("tools", [])
        tools = [_normalise_tool(t) for t in raw_tools] if raw_tools else None

        request = CompletionRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            tools=tools,
            max_tokens=512,
        )

        start = time.time()
        response = self._provider.complete(request)
        latency_ms = (time.time() - start) * 1000

        scorer_specs = case.get("scorers", [])
        scorer_result = self._score_tool_call(
            response.text, response.tool_calls, scorer_specs
        )

        turn = TurnResult(
            turn_index=0,
            user_message=prompt,
            response=response.text,
            tool_calls=response.tool_calls,
            scorer_result=scorer_result,
            latency_ms=latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

        return AgentEvalResult(
            case_id=case_id,
            description=description,
            case_type="tool_call",
            turns=[turn],
            scorer_result=scorer_result,
            tags=tags,
        )

    def _score_tool_call(
        self, text: str, tool_calls: list | None, scorer_specs: list[dict]
    ) -> ScorerResult:
        if not scorer_specs:
            # No scorers defined: pass if any tool was called
            if tool_calls:
                return ScorerResult(
                    score=1.0, passed=True, strategy="tool_call_default",
                    explanation=f"Model called: {[tc['name'] for tc in tool_calls]}",
                )
            return ScorerResult(
                score=0.0, passed=False, strategy="tool_call_default",
                explanation="No tool was called and no scorers defined",
            )

        results = []
        for spec in scorer_specs:
            stype = spec.get("type", "")
            if stype == "tool_call_match":
                results.append(Scorer.tool_call_match(
                    tool_calls,
                    expected_function=spec["expected_function"],
                    expected_args=spec.get("expected_args"),
                    require_all_args=spec.get("require_all_args", False),
                    threshold=spec.get("threshold", 1.0),
                ))
            elif stype == "no_tool_call":
                # Passes if the model did NOT call any tool
                called = bool(tool_calls)
                results.append(ScorerResult(
                    score=0.0 if called else 1.0,
                    passed=not called,
                    strategy="no_tool_call",
                    explanation=f"Tool called: {called}",
                    threshold=spec.get("threshold", 1.0),
                ))
            elif stype == "contains_keywords":
                results.append(Scorer.contains_keywords(
                    text,
                    keywords=spec["keywords"],
                    require_all=spec.get("require_all", True),
                    threshold=spec.get("threshold", 0.5),
                ))
            elif stype == "llm_judge":
                results.append(Scorer.llm_judge(
                    text,
                    judge_prompt=spec["judge_prompt"],
                    pass_if=spec.get("pass_if", "YES"),
                    provider=self._provider,
                    threshold=spec.get("threshold", 1.0),
                ))

        if len(results) == 1:
            return results[0]
        return Scorer.aggregate(results, strategy="min")

    # ------------------------------------------------------------------ #
    # Multi-turn case                                                      #
    # ------------------------------------------------------------------ #

    def _run_multi_turn(self, case: dict) -> AgentEvalResult:
        case_id = case.get("id", "unknown")
        description = case.get("description", "")
        system_prompt = case.get("system_prompt")
        tags = case.get("tags", [])
        turn_specs = case.get("turns", [])

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        turn_results: list[TurnResult] = []
        all_scorer_results: list[ScorerResult] = []

        for i, turn_spec in enumerate(turn_specs):
            user_msg = turn_spec.get("user", "")
            messages.append({"role": "user", "content": user_msg})

            request = CompletionRequest(
                prompt=user_msg,
                messages=list(messages),  # snapshot of history
                max_tokens=512,
            )

            start = time.time()
            response = self._provider.complete(request)
            latency_ms = (time.time() - start) * 1000

            # Append assistant reply to history for next turn
            messages.append({"role": "assistant", "content": response.text})

            # Score this turn
            scorer_specs = turn_spec.get("scorers", [])
            if scorer_specs:
                turn_scorer = self._build_turn_scorer(scorer_specs)
                turn_result_score = turn_scorer(response.text)
                all_scorer_results.append(turn_result_score)
            else:
                turn_result_score = None

            turn_results.append(TurnResult(
                turn_index=i,
                user_message=user_msg,
                response=response.text,
                tool_calls=response.tool_calls,
                scorer_result=turn_result_score,
                latency_ms=latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            ))

        # Aggregate across all scored turns (min strategy: every turn must pass)
        if all_scorer_results:
            aggregate = Scorer.aggregate(all_scorer_results, strategy="min")
        else:
            aggregate = ScorerResult(
                score=1.0, passed=True, strategy="multi_turn_no_scorers",
                explanation="No scorers defined on any turn",
            )

        return AgentEvalResult(
            case_id=case_id,
            description=description,
            case_type="multi_turn",
            turns=turn_results,
            scorer_result=aggregate,
            tags=tags,
        )

    def _build_turn_scorer(self, scorer_specs: list[dict]):
        """Build a scorer callable for a single conversation turn."""
        def scorer_fn(text: str) -> ScorerResult:
            results = []
            for spec in scorer_specs:
                stype = spec.get("type", "")
                if stype == "exact_match":
                    results.append(Scorer.exact_match(
                        text, expected=spec["expected"],
                        case_sensitive=spec.get("case_sensitive", False),
                        threshold=spec.get("threshold", 1.0),
                    ))
                elif stype == "contains_keywords":
                    results.append(Scorer.contains_keywords(
                        text, keywords=spec["keywords"],
                        require_all=spec.get("require_all", True),
                        threshold=spec.get("threshold", 0.5),
                    ))
                elif stype == "regex_match":
                    results.append(Scorer.regex_match(
                        text, pattern=spec["pattern"],
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
                        text, forbidden=spec["forbidden"],
                        threshold=spec.get("threshold", 1.0),
                    ))
                elif stype == "llm_judge":
                    results.append(Scorer.llm_judge(
                        text, judge_prompt=spec["judge_prompt"],
                        pass_if=spec.get("pass_if", "YES"),
                        provider=self._provider,
                        threshold=spec.get("threshold", 1.0),
                    ))
            if not results:
                return ScorerResult(score=1.0, passed=True, strategy="no_op",
                                    explanation="No scorers for this turn")
            if len(results) == 1:
                return results[0]
            return Scorer.aggregate(results, strategy=spec.get("aggregate", "mean"))

        return scorer_fn


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _normalise_tool(tool: dict) -> dict:
    """
    Accept both shorthand YAML format and full OpenAI format.

    Shorthand (what test authors write):
      name: get_weather
      description: "..."
      parameters: {...}

    Full OpenAI format (passed through unchanged):
      type: function
      function:
        name: ...
    """
    if "type" in tool and tool["type"] == "function":
        return tool
    # Shorthand → OpenAI format
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }
