"""Tests for the rubric judge (P2). All offline with mocked providers."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from providers.base import CompletionResponse
from evals.rubric import load_rubric
from evals.rubric_judge import RubricJudge

RUBRIC = (
    Path(__file__).resolve().parent.parent / "rubrics" / "conversation_v1.yaml"
)

CONVERSATION = {
    "id": "t1",
    "system_prompt": "You are a helpful assistant.",
    "turns": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "Are you sure?"},
        {"role": "assistant", "content": "Yes, 2+2=4."},
    ],
}

GOOD_JSON = (
    '{"reasons": {'
    '"correctness": "Turn 2 answers 4, correct; turn 4 confirms.", '
    '"instruction_adherence": "No special instructions; direct answers in turns 2 and 4.", '
    '"context_consistency": "Turn 4 is consistent with turn 2."}, '
    '"scores": {"correctness": 2, "instruction_adherence": 2, "context_consistency": 2}}'
)


def make_provider(*texts):
    """Mock provider returning each text in turn."""
    provider = MagicMock()
    provider.complete.side_effect = [
        CompletionResponse(text=t, model="mock", provider="mock", latency_ms=1.0)
        for t in texts
    ]
    return provider


@pytest.fixture()
def rubric():
    return load_rubric(RUBRIC)


def test_valid_json(rubric):
    judge = RubricJudge(make_provider(GOOD_JSON), rubric, model="judge-1")
    result = judge.judge(CONVERSATION)
    assert result.ok
    assert result.error is None
    assert result.scores == {
        "correctness": 2, "instruction_adherence": 2, "context_consistency": 2,
    }
    assert set(result.reasons) == set(result.scores)
    assert all(result.reasons.values())


def test_fenced_json(rubric):
    fenced = "```json\n" + GOOD_JSON + "\n```"
    judge = RubricJudge(make_provider(fenced), rubric, model="judge-1")
    result = judge.judge(CONVERSATION)
    assert result.ok
    assert result.scores["correctness"] == 2


def test_garbage_then_valid_retries_once(rubric):
    provider = make_provider("I cannot decide, sorry.", GOOD_JSON)
    judge = RubricJudge(provider, rubric, model="judge-1")
    result = judge.judge(CONVERSATION)
    assert result.ok
    assert result.scores["correctness"] == 2
    assert provider.complete.call_count == 2


def test_garbage_twice_records_error(rubric):
    provider = make_provider("nope", "still nope")
    judge = RubricJudge(provider, rubric, model="judge-1")
    result = judge.judge(CONVERSATION)
    assert not result.ok
    assert result.error == "unparseable"
    assert result.scores is None
    assert result.reasons is None
    assert provider.complete.call_count == 2
    assert result.raw == "still nope"


def _bad_then_good(rubric, bad_json):
    provider = make_provider(bad_json, GOOD_JSON)
    judge = RubricJudge(provider, rubric, model="judge-1")
    result = judge.judge(CONVERSATION)
    assert result.ok, "valid retry should succeed"
    assert provider.complete.call_count == 2, "invalid first reply must be retried"
    return result


def test_score_outside_scale_rejected(rubric):
    bad = GOOD_JSON.replace('"correctness": 2', '"correctness": 3')
    result = _bad_then_good(rubric, bad)
    assert result.scores["correctness"] == 2


def test_missing_dimension_rejected(rubric):
    import json

    bad = json.dumps({
        "reasons": {
            "correctness": "Turn 2 answers 4, correct.",
            "instruction_adherence": "Direct answers in turns 2 and 4.",
        },
        "scores": {"correctness": 2, "instruction_adherence": 2},
    })
    result = _bad_then_good(rubric, bad)
    assert set(result.scores) == {
        "correctness", "instruction_adherence", "context_consistency",
    }


def test_empty_reason_rejected(rubric):
    bad = GOOD_JSON.replace(
        '"correctness": "Turn 2 answers 4, correct; turn 4 confirms."',
        '"correctness": "  "',
    )
    result = _bad_then_good(rubric, bad)
    assert result.reasons["correctness"].strip()


def test_provider_receives_temperature_and_model(rubric):
    provider = make_provider(GOOD_JSON)
    RubricJudge(provider, rubric, model="judge-1").judge(CONVERSATION)
    request = provider.complete.call_args[0][0]
    assert request.temperature == 0.0
    assert request.model == "judge-1"


def test_result_carries_rubric_version_and_sha(rubric):
    judge = RubricJudge(make_provider(GOOD_JSON), rubric, model="judge-1")
    result = judge.judge(CONVERSATION)
    assert result.rubric_version == rubric.version == "conversation_v1"
    assert result.rubric_sha == rubric.sha256
    assert result.model == "judge-1"


def test_accepts_bare_turn_list(rubric):
    judge = RubricJudge(make_provider(GOOD_JSON), rubric, model="judge-1")
    result = judge.judge(CONVERSATION["turns"])
    assert result.ok
