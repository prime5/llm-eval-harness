# sdet — PRD → test plan → tests

Agentic SDET workflow built on this harness. System under test: [flowgate](https://github.com/prime5/flowgate).

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Test-plan schema, validator, golden flowgate plan | ✅ |
| 1 | `plan`: PRD → schema-valid plan via LLM | |
| 2 | Grade the planner: golden set, rubric judge, variance | |
| 3 | `codegen`: plan → pytest, gated on parse/collect | |
| 4 | CI: generated suite runs against the deployed SUT | |
| 5 | Triage agent: classify failures, propose fixes as PRs | |

## Why a plan schema (and not PRD → code directly)

The plan is the contract between stages. It is diffable, reviewable by a human, and gives codegen a stable target. It also makes the planner gradeable with no LLM in the loop.

## Design decisions

- **Every requirement carries a verbatim PRD quote.** The validator checks it against the PRD (S4). A fabricated requirement fails deterministically, with no judge call.
- **Coverage is computed, not claimed.** Every testable requirement needs at least one test (S3). Requirements can be marked `testable: false` only with a reason. The planner has to say *why* it can't test something, not skip it silently.
- **Open questions are first-class.** A planner that flags an ambiguity ("5 failures: consecutive or windowed?") scores better than one that invents an answer.
- **Assertions are structured** (`subject` / `operator` / `expected` / `filter`). Codegen is mechanical, and the Phase 5 fix agent can be checked for *weakening* an assertion rather than fixing the cause.
- **Stochastic tests declare their tolerance.** Anything touching flowgate's 2% simulated error rate must set `repetitions` and `min_pass_rate`. That applies the harness's variance analysis to generated tests.

## Run

```bash
pytest tests/test_plan_schema.py -v
python -m sdet.validate_plan sdet/fixtures/flowgate_plan.example.json --prd sdet/fixtures/flowgate_prd.md
```

Requires `jsonschema` (Draft 7).
