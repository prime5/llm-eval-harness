"""
Phase 0 — offline tests for the test-plan schema and validator.

Each mutation test breaks the golden plan in one specific way and asserts
the validator reports exactly that failure class. If a check stops firing,
the matching test fails.
"""
import copy
import json
from pathlib import Path

import pytest

from sdet.validate_plan import validate_plan

FIXTURES = Path(__file__).parent.parent / "sdet" / "fixtures"


@pytest.fixture(scope="module")
def prd_text():
    return (FIXTURES / "flowgate_prd.md").read_text()


@pytest.fixture
def plan():
    return json.loads((FIXTURES / "flowgate_plan.example.json").read_text())


def codes(report):
    return {i.code for i in report.errors}


def warning_codes(report):
    return {i.code for i in report.warnings}


# --------------------------------------------------------------------------- #
# Golden plan                                                                  #
# --------------------------------------------------------------------------- #
def test_golden_plan_is_valid(plan, prd_text):
    report = validate_plan(plan, prd_text)
    assert report.ok, report.to_dict()
    assert report.warnings == []


def test_golden_plan_coverage(plan, prd_text):
    cov = validate_plan(plan, prd_text).coverage
    assert cov["requirements"] == 14
    assert cov["testable"] == 13
    assert cov["covered"] == 13
    assert cov["coverage"] == 1.0


# --------------------------------------------------------------------------- #
# Structural layer (JSON Schema)                                               #
# --------------------------------------------------------------------------- #
def test_test_case_without_requirement_is_rejected(plan):
    plan["test_cases"][0]["requirement_ids"] = []
    assert "SCHEMA" in codes(validate_plan(plan))


def test_untestable_requirement_needs_reason(plan):
    untestable = next(r for r in plan["requirements"] if not r["testable"])
    del untestable["untestable_reason"]
    assert "SCHEMA" in codes(validate_plan(plan))


def test_stochastic_test_needs_repetitions(plan):
    tc = next(t for t in plan["test_cases"] if t["determinism"] == "stochastic")
    del tc["stochastic"]
    assert "SCHEMA" in codes(validate_plan(plan))


def test_llm_generator_must_record_model_and_prompt(plan):
    plan["generator"] = {"kind": "llm", "generated_at": "2026-09-21T00:00:00Z"}
    assert "SCHEMA" in codes(validate_plan(plan))


def test_count_assertion_needs_filter(plan):
    tc = next(t for t in plan["test_cases"] if t["id"] == "TC-002")
    del tc["assertions"][0]["filter"]
    assert "SCHEMA" in codes(validate_plan(plan))


def test_unknown_assertion_subject_is_rejected(plan):
    plan["test_cases"][0]["assertions"][0]["subject"] = "vibes"
    assert "SCHEMA" in codes(validate_plan(plan))


def test_request_step_needs_request(plan):
    del plan["test_cases"][0]["steps"][0]["request"]
    assert "SCHEMA" in codes(validate_plan(plan))


# --------------------------------------------------------------------------- #
# Semantic layer                                                               #
# --------------------------------------------------------------------------- #
def test_s1_duplicate_test_case_id(plan):
    plan["test_cases"].append(copy.deepcopy(plan["test_cases"][0]))
    assert "S1" in codes(validate_plan(plan))


def test_s2_dangling_requirement_reference(plan):
    plan["test_cases"][0]["requirement_ids"] = ["REQ-XXX-999"]
    assert "S2" in codes(validate_plan(plan))


def test_s3_uncovered_testable_requirement(plan):
    plan["test_cases"] = [t for t in plan["test_cases"] if "REQ-HLT-001" not in t["requirement_ids"]]
    report = validate_plan(plan)
    assert "S3" in codes(report)
    assert report.coverage == {} or report.coverage["coverage"] < 1.0


def test_s4_fabricated_quote(plan, prd_text):
    plan["requirements"][0]["source_quote"] = "GET /healthz returns 204 with an empty body."
    assert "S4" in codes(validate_plan(plan, prd_text))


def test_s4_tolerates_whitespace_differences(plan, prd_text):
    q = plan["requirements"][4]["source_quote"]
    plan["requirements"][4]["source_quote"] = q.replace(" ", "\n  ")
    assert validate_plan(plan, prd_text).ok


def test_s5_stale_plan_is_a_warning_not_an_error(plan, prd_text):
    report = validate_plan(plan, prd_text + "\nNew requirement added later.\n")
    assert report.ok
    assert "S5" in warning_codes(report)


def test_s6_test_covering_only_untestable_requirement(plan):
    plan["test_cases"][0]["requirement_ids"] = ["REQ-ORD-001"]
    # TC-001 was the only test for REQ-HLT-001, so S3 fires too.
    report = validate_plan(plan)
    assert "S6" in warning_codes(report)
    assert "S3" in codes(report)


# --------------------------------------------------------------------------- #
# v0.2: SUT configuration                                                      #
# --------------------------------------------------------------------------- #
def test_sut_config_requires_env(plan):
    tc = next(t for t in plan["test_cases"] if "sut_config" in t)
    tc["sut_config"] = {"rationale": "no env block at all, which is not a configuration"}
    assert "SCHEMA" in codes(validate_plan(plan))


def test_sut_config_env_values_must_be_strings(plan):
    tc = next(t for t in plan["test_cases"] if "sut_config" in t)
    tc["sut_config"]["env"]["MAX_IN_FLIGHT"] = 2
    assert "SCHEMA" in codes(validate_plan(plan))


def test_breaker_tests_declare_the_config_that_makes_them_possible(plan):
    """The breaker is only drivable with a lowered threshold; a plan that claims
    to test it without saying so is making a claim it cannot keep."""
    for tc in plan["test_cases"]:
        if any(r.startswith("REQ-CB-") for r in tc["requirement_ids"]):
            assert tc.get("sut_config", {}).get("env", {}).get("BREAKER_THRESHOLD"), tc["id"]
