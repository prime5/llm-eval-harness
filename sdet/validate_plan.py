"""
Phase 0 — validate a test plan before anything downstream consumes it.

Two layers, cheapest first:

  1. Structural: JSON Schema (sdet/schema/test_plan.schema.json)
  2. Semantic:   cross-references JSON Schema cannot express
       S1  duplicate IDs                                              error
       S2  test case / question points at a missing requirement       error
       S3  testable requirement with no test case (coverage gap)      error
       S4  requirement quote not found verbatim in the PRD            error
       S5  plan's PRD hash doesn't match the PRD supplied (stale)     warning
       S6  test case covers only untestable requirements              warning

Semantic checks only run when the structure is valid; they assume the shape.

Usage:
    python -m sdet.validate_plan sdet/fixtures/flowgate_plan.example.json \
        --prd sdet/fixtures/flowgate_prd.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from jsonschema import Draft7Validator

SCHEMA_PATH = Path(__file__).parent / "schema" / "test_plan.schema.json"


@dataclass
class Issue:
    code: str
    message: str
    path: str = ""


@dataclass
class PlanReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    coverage: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": [asdict(i) for i in self.errors],
            "warnings": [asdict(i) for i in self.warnings],
            "coverage": self.coverage,
        }


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def prd_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_plan(plan: dict, prd_text: Optional[str] = None) -> PlanReport:
    report = PlanReport()

    # ---- Layer 1: structure ------------------------------------------------
    validator = Draft7Validator(load_schema())
    for err in sorted(validator.iter_errors(plan), key=lambda e: [str(p) for p in e.path]):
        report.errors.append(
            Issue("SCHEMA", err.message, "/".join(str(p) for p in err.path))
        )
    if report.errors:
        return report

    reqs = plan["requirements"]
    tcs = plan["test_cases"]
    qs = plan.get("open_questions", [])

    # ---- S1: duplicate IDs -------------------------------------------------
    for kind, items in (("requirement", reqs), ("test case", tcs), ("open question", qs)):
        seen: set = set()
        for item in items:
            if item["id"] in seen:
                report.errors.append(Issue("S1", f"duplicate {kind} id {item['id']}", item["id"]))
            seen.add(item["id"])

    req_by_id = {r["id"]: r for r in reqs}

    # ---- S2: dangling references -------------------------------------------
    for tc in tcs:
        for rid in tc["requirement_ids"]:
            if rid not in req_by_id:
                report.errors.append(Issue("S2", f"{tc['id']} references unknown {rid}", tc["id"]))
    for q in qs:
        for rid in q.get("requirement_ids", []):
            if rid not in req_by_id:
                report.errors.append(Issue("S2", f"{q['id']} references unknown {rid}", q["id"]))

    # ---- S3 / S6: coverage -------------------------------------------------
    covered = {rid for tc in tcs for rid in tc["requirement_ids"] if rid in req_by_id}
    testable = [r for r in reqs if r["testable"]]
    for r in testable:
        if r["id"] not in covered:
            report.errors.append(Issue("S3", f"testable requirement {r['id']} has no test case", r["id"]))
    for tc in tcs:
        refs = [req_by_id[rid] for rid in tc["requirement_ids"] if rid in req_by_id]
        if refs and not any(r["testable"] for r in refs):
            report.warnings.append(
                Issue("S6", f"{tc['id']} only covers requirements marked untestable", tc["id"])
            )

    # ---- S4 / S5: grounding in the PRD -------------------------------------
    if prd_text is not None:
        norm_prd = _norm(prd_text)
        for r in reqs:
            if _norm(r["source_quote"]) not in norm_prd:
                report.errors.append(
                    Issue("S4", f"{r['id']} quote not found verbatim in PRD", r["id"])
                )
        if plan["prd"]["sha256"] != prd_sha256(prd_text):
            report.warnings.append(
                Issue("S5", "plan was generated from a different PRD version", "prd/sha256")
            )

    covered_testable = sum(1 for r in testable if r["id"] in covered)
    report.coverage = {
        "requirements": len(reqs),
        "testable": len(testable),
        "covered": covered_testable,
        "coverage": round(covered_testable / len(testable), 3) if testable else None,
        "test_cases": len(tcs),
        "open_questions": len(qs),
    }
    return report


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a PRD-derived test plan.")
    ap.add_argument("plan", type=Path)
    ap.add_argument("--prd", type=Path, help="PRD file; enables quote-grounding (S4) and staleness (S5) checks")
    args = ap.parse_args(argv)

    plan = json.loads(args.plan.read_text())
    prd_text = args.prd.read_text() if args.prd else None
    report = validate_plan(plan, prd_text)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
