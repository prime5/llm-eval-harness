"""
Phase 1 — offline tests for the planner.

No API key and no network: a stub provider replays canned responses, so the
parts worth testing (JSON extraction, metadata stamping, the repair loop,
refusing an invalid plan) are tested deterministically.
"""
import copy
import json
from pathlib import Path

import pytest

from providers.base import BaseProvider, CompletionRequest, CompletionResponse
from sdet.planner import PROMPT_VERSION, build_system_prompt, extract_json, generate_plan
from sdet.validate_plan import prd_sha256

FIXTURES = Path(__file__).parent.parent / "sdet" / "fixtures"


class StubProvider(BaseProvider):
    """Replays scripted responses; records the messages it was sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    @property
    def name(self) -> str:
        return "stub"

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request.messages)
        text = self._responses.pop(0)
        return CompletionResponse(text=text, model="stub-model", provider="stub")


@pytest.fixture(scope="module")
def prd_text():
    return (FIXTURES / "flowgate_prd.md").read_text()


@pytest.fixture(scope="module")
def env_text():
    return (FIXTURES / "flowgate_test_env.md").read_text()


@pytest.fixture
def golden():
    plan = json.loads((FIXTURES / "flowgate_plan.example.json").read_text())
    plan.pop("prd", None)
    plan.pop("generator", None)
    return plan


def run(provider, prd_text, env_text, **kw):
    return generate_plan(
        provider,
        prd_text=prd_text,
        env_text=env_text,
        prd_path="sdet/fixtures/flowgate_prd.md",
        sut="flowgate",
        plan_id="test-plan",
        run_id="deadbeef",
        **kw,
    )


# --------------------------------------------------------------------------- #
# JSON extraction                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here is the plan:\n{"a": 1}\nLet me know if you want changes.',
    ],
)
def test_extract_json_survives_wrapping(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_raises_without_an_object():
    with pytest.raises(ValueError):
        extract_json("I'd be happy to help with that!")


# --------------------------------------------------------------------------- #
# Happy path                                                                   #
# --------------------------------------------------------------------------- #
def test_valid_plan_accepted_on_first_attempt(golden, prd_text, env_text):
    provider = StubProvider([json.dumps(golden)])
    result = run(provider, prd_text, env_text)
    assert result.report.ok, result.report.to_dict()
    assert result.attempts == 1
    assert result.report.coverage["coverage"] == 1.0


def test_metadata_is_stamped_not_taken_from_the_model(golden, prd_text, env_text):
    lying = copy.deepcopy(golden)
    lying["prd"] = {"path": "nope.md", "sha256": "0" * 64}
    lying["generator"] = {"kind": "human", "generated_at": "1999-01-01T00:00:00Z"}
    result = run(StubProvider([json.dumps(lying)]), prd_text, env_text)
    assert result.plan["prd"]["sha256"] == prd_sha256(prd_text)
    assert result.plan["prd"]["path"] == "sdet/fixtures/flowgate_prd.md"
    assert result.plan["generator"]["kind"] == "llm"
    assert result.plan["generator"]["model"] == "stub:stub-model"
    assert result.plan["generator"]["prompt_version"] == PROMPT_VERSION
    assert result.plan["generator"]["run_id"] == "deadbeef"
    assert result.report.ok


def test_schema_is_included_in_the_system_prompt():
    assert '"TestPlan"' in build_system_prompt()
    assert "source_quote" in build_system_prompt()


# --------------------------------------------------------------------------- #
# Repair loop                                                                  #
# --------------------------------------------------------------------------- #
def test_fabricated_quote_is_repaired_on_second_attempt(golden, prd_text, env_text):
    broken = copy.deepcopy(golden)
    broken["requirements"][0]["source_quote"] = "GET /healthz returns 204 and an empty body."
    provider = StubProvider([json.dumps(broken), json.dumps(golden)])
    result = run(provider, prd_text, env_text, max_repairs=1)

    assert result.report.ok
    assert result.attempts == 2
    assert result.plan["generator"]["repair_attempts"] == 1

    repair_message = provider.calls[1][-1]["content"]
    assert "S4" in repair_message
    assert "REQ-HLT-001" in repair_message


def test_unparseable_response_is_retried(golden, prd_text, env_text):
    provider = StubProvider(["I can't produce that.", json.dumps(golden)])
    result = run(provider, prd_text, env_text, max_repairs=1)
    assert result.report.ok
    assert "PARSE" in provider.calls[1][-1]["content"]


def test_still_invalid_after_repairs_is_reported_not_returned_clean(golden, prd_text, env_text):
    broken = copy.deepcopy(golden)
    broken["test_cases"] = [t for t in broken["test_cases"] if "REQ-HLT-001" not in t["requirement_ids"]]
    provider = StubProvider([json.dumps(broken), json.dumps(broken)])
    result = run(provider, prd_text, env_text, max_repairs=1)

    assert not result.report.ok
    assert "S3" in {i.code for i in result.report.errors}
    assert result.attempts == 2
    assert len(result.raw_responses) == 2


def test_max_repairs_zero_makes_one_call(golden, prd_text, env_text):
    provider = StubProvider([json.dumps(golden)])
    result = run(provider, prd_text, env_text, max_repairs=0)
    assert result.attempts == 1
    assert len(provider.calls) == 1
