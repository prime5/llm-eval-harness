"""
Phase 1 — PRD to test plan.

The planner asks a model for a plan, then refuses to accept one that does not
validate. Validation errors are fed back for a bounded number of repair
attempts; a plan that still fails is returned with its report so the failure is
visible rather than silently downstream.

Two inputs, deliberately separate:
  PRD           product truth. Requirement quotes must come from here.
  env contract  what the test runner may assume about the SUT (environments,
                configuration knobs). Never a source of requirements.

Metadata the model is not trusted with — the PRD hash, model name, prompt
version, timestamp — is filled in by this module after parsing.

Usage:
    python -m sdet.planner \
        --prd sdet/fixtures/flowgate_prd.md \
        --env-doc sdet/fixtures/flowgate_test_env.md \
        --sut flowgate \
        --out sdet/out/flowgate_plan.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from providers.base import BaseProvider, CompletionRequest

from sdet.validate_plan import Issue, PlanReport, load_schema, prd_sha256, validate_plan

PROMPT_VERSION = "plan-v1"

SYSTEM_PROMPT = """You are a senior test engineer. You read a product requirements document and produce a test plan as JSON.

Rules:
1. Output JSON only. No prose, no markdown fences.
2. Every requirement you list must carry a `source_quote` copied WORD FOR WORD from the PRD. Never paraphrase a quote. Never quote the environment contract.
3. Cover the PRD. Every requirement that can be tested needs at least one test case.
4. If a requirement cannot be tested in any available environment, set `testable` to false and give an `untestable_reason`. Do not invent a test you cannot actually run.
5. Where the PRD is ambiguous or silent on something a test needs, add an `open_question` instead of guessing. Asking is better than inventing.
6. Assertions must be specific. An assertion that would pass whatever the system did is worthless.
7. Omit the `prd` and `generator` fields; the tool fills them in.

Conform to this JSON Schema:

{schema}
"""

USER_PROMPT = """# Product requirements document

{prd}

---

# Test environment contract

{env_doc}

---

Produce the test plan for system under test "{sut}" as JSON. plan_id: "{plan_id}".
"""

REPAIR_PROMPT = """The plan you produced failed validation.

{errors}

Notes on the checks:
- SCHEMA: structural; the message names the offending field path.
- S1: duplicate ID.
- S2: a test case or question references a requirement ID that is not in the plan.
- S3: a requirement marked testable has no test case covering it.
- S4: the `source_quote` is not present word for word in the PRD. Copy the exact sentence, or drop the requirement.

Return the corrected plan as JSON only. Fix the listed problems and change nothing else."""


@dataclass
class PlannerResult:
    plan: dict
    report: PlanReport
    attempts: int
    raw_responses: list


def build_user_prompt(prd_text: str, env_text: str, sut: str, plan_id: str) -> str:
    return USER_PROMPT.format(prd=prd_text.strip(), env_doc=env_text.strip(), sut=sut, plan_id=plan_id)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(schema=json.dumps(load_schema(), indent=2))


def extract_json(text: str) -> dict:
    """Models wrap JSON in fences or prose more often than they admit."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in response")
    return json.loads(text[start : end + 1])


def stamp_metadata(
    plan: dict,
    *,
    prd_path: str,
    prd_text: str,
    model: str,
    provider_name: str,
    plan_id: str,
    run_id: str,
    repair_attempts: int,
) -> dict:
    """Fill the fields the model is not trusted to report about itself."""
    plan["schema_version"] = "0.2"
    plan.setdefault("plan_id", plan_id)
    plan["prd"] = {"path": prd_path, "sha256": prd_sha256(prd_text)}
    plan["generator"] = {
        "kind": "llm",
        "model": f"{provider_name}:{model}",
        "prompt_version": PROMPT_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repair_attempts": repair_attempts,
    }
    return plan


def format_errors(report: PlanReport) -> str:
    return "\n".join(f"- [{i.code}] {i.path}: {i.message}" for i in report.errors)


def generate_plan(
    provider: BaseProvider,
    *,
    prd_text: str,
    env_text: str,
    prd_path: str,
    sut: str,
    plan_id: str,
    model: str = "",
    temperature: float = 0.0,
    max_tokens: int = 8000,
    max_repairs: int = 1,
    run_id: Optional[str] = None,
) -> PlannerResult:
    run_id = run_id or uuid.uuid4().hex[:12]
    system_prompt = build_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(prd_text, env_text, sut, plan_id)},
    ]

    raw_responses: list = []
    report = PlanReport()
    plan: dict = {}

    for attempt in range(max_repairs + 1):
        response = provider.complete(
            CompletionRequest(
                prompt="",
                messages=messages,
                model=model or None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        raw_responses.append(response.text)

        try:
            plan = extract_json(response.text)
        except (ValueError, json.JSONDecodeError) as exc:
            report = PlanReport()
            report.errors.append(Issue("PARSE", f"response was not usable JSON: {exc}"))
            messages += [
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": REPAIR_PROMPT.format(errors=format_errors(report))},
            ]
            continue

        plan = stamp_metadata(
            plan,
            prd_path=prd_path,
            prd_text=prd_text,
            model=model or response.model,
            provider_name=provider.name,
            plan_id=plan_id,
            run_id=run_id,
            repair_attempts=attempt,
        )
        report = validate_plan(plan, prd_text)
        if report.ok:
            return PlannerResult(plan, report, attempt + 1, raw_responses)

        messages += [
            {"role": "assistant", "content": json.dumps(plan)},
            {"role": "user", "content": REPAIR_PROMPT.format(errors=format_errors(report))},
        ]

    return PlannerResult(plan, report, max_repairs + 1, raw_responses)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a test plan from a PRD.")
    ap.add_argument("--prd", type=Path, required=True)
    ap.add_argument("--env-doc", type=Path, required=True)
    ap.add_argument("--sut", default="flowgate")
    ap.add_argument("--plan-id", default=None, help="default: <sut>-<run id>")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default="", help="default: provider's DEFAULT_MODEL")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-repairs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("sdet/out/plan.json"))
    args = ap.parse_args(argv)

    from providers import get_provider

    prd_text = args.prd.read_text()
    env_text = args.env_doc.read_text()
    run_id = uuid.uuid4().hex[:12]
    plan_id = args.plan_id or f"{args.sut}-{run_id}"

    result = generate_plan(
        get_provider(args.provider),
        prd_text=prd_text,
        env_text=env_text,
        prd_path=str(args.prd),
        sut=args.sut,
        plan_id=plan_id,
        model=args.model,
        temperature=args.temperature,
        max_repairs=args.max_repairs,
        run_id=run_id,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result.plan, indent=2) + "\n")
    raw_path = args.out.with_suffix(".raw.txt")
    raw_path.write_text("\n\n===== attempt boundary =====\n\n".join(result.raw_responses))

    print(json.dumps(
        {
            "plan": str(args.out),
            "raw": str(raw_path),
            "attempts": result.attempts,
            "valid": result.report.ok,
            "errors": [i.__dict__ for i in result.report.errors],
            "warnings": [i.__dict__ for i in result.report.warnings],
            "coverage": result.report.coverage,
        },
        indent=2,
    ))
    return 0 if result.report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
