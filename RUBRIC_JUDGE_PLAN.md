# Rubric judge and golden dataset: phase plan

Written 2026-10-10. Self-contained: anyone (you, or any assistant) can pick this up
from the repo alone, without the conversation that produced it.

## Resume protocol (read this first)

1. Run `git log --oneline | head` and `pytest -q`. Everything offline must pass before you start.
2. Find the first unchecked phase in the status list below. Do **only that phase**.
3. Do its steps, run its **Done when** checks, commit with its message, tick the box here.
4. Never skip ahead. Phases are ordered because each one's output is the next one's input.
5. Hand-off prompt for any assistant: "Read RUBRIC_JUDGE_PLAN.md. Do the first unchecked
   phase only. Run its Done-when checks. Commit with its message. Do not write human labels."

## Status

- [ ] P0  Fix the existing binary judge (confirmed bug), baseline tests
- [ ] P1  Rubric spec v1 and loader
- [ ] P2  Rubric judge (JSON verdicts with reasons)
- [ ] P3  Golden conversations and validator, with a fixed dev/test split
- [ ] P4  Labeling tool and your blind labels (human only)
- [ ] P5  Agreement metrics (Cohen's kappa, weighted)
- [ ] P6  Judge-vs-human run and report
- [ ] P7  Improve the rubric from dev disagreements, report the held-out split once
- [ ] P8  Later: judge self-consistency, NDCG on retrieval, GEPA prompt optimization

Each phase is sized for one 30-45 minute session and ends in one commit.

## Where the repo is today (verified 2026-10-10)

- `evals/scorer.py`: `ScorerResult(score, passed, strategy, explanation, threshold, details)`.
  `Scorer.llm_judge(response, judge_prompt, pass_if="NO", provider=None, threshold=1.0)`
  asks a second model to answer `pass_if` or its opposite, at temperature 0.0, max_tokens 150.
- **Confirmed bug:** it passes when `pass_if.upper()` appears *anywhere* in the judge's
  text. With `pass_if="NO"`, a judge that answers "YES. The model followed the injected
  instruction. NOTE: this is a clear failure." is scored as a pass, because "NOTE"
  contains "NO". "I cannot tell; this is not clear." also passes. Reproduced with a mock
  provider on 2026-10-10. Similar substring hazards exist for "YES" (e.g. "yesterday").
- Defaults disagree: `evals/runner.py` line ~223 defaults `pass_if` to "NO";
  `evals/agent_runner.py` lines ~297 and ~417 default it to "YES".
- 14 `llm_judge` uses in test cases: hallucination 4, jailbreak 4, prompt_injection 3,
  safety_compliance 3. Results recorded before P0 used the buggy logic.
- `providers/base.py`: `CompletionRequest(prompt, system_prompt, model, temperature, max_tokens, tools, messages)`,
  `CompletionResponse(text, model, provider, ...)`, `BaseProvider.complete()`.
- `evals/variance.py` already measures repeated-run consistency.
- `reporters/slack.py` (added 2026-10-10): `post(text)` to an incoming webhook, never raises, never
  echoes the webhook URL, validates the host; `format_drift_alert(report)`; used by `detect_drift.py`.
  Offline tests in `tests/test_slack_reporter.py`. Posting to a real channel verified by the owner (a test
  message arrived in the eval-reports channel).
- Tests are offline with mocked providers (`tests/test_adversarial.py` has
  `make_mock_provider(judge_text)`). Keep every new unit test offline.
- Multi-turn cases live in `test_cases/multi_turn.yaml` (`turns: [{user, scorers}]`).
- `requirements.txt` has `pyyaml` and `jsonschema`. It does not have scikit-learn
  (use it only as an optional check in tests).
- Uncommitted at this date: modified `.github/workflows/eval_ci.yml`, `.gitignore`,
  `README.md`, `requirements.txt`; untracked `.claude/`, `.dockerignore`, `Dockerfile`,
  `TRAJECTORY_EVAL_SPEC.md`, `phase6/`, `sdet/`. Do not commit `.claude/`.

## What your ServiceNow friend's list maps to

| Friend's item | Phase |
|---|---|
| 1. Rubric (scorecard), 0-2 scoring | P1, P2 |
| 2. Golden data sets | P3, P4 |
| 3. Annotate with a reason for each score | P2 (judge reasons), P4 (your reasons) |
| 4. Test kappa | P5, P6 |
| 4. GEPA / agentic context engineering (DSPy) | P8, gated on P6/P7 |
| 5. NDCG | P8, applied to retrieval ranking (see Appendix D) |
| "Auto research": improve the prompt, not the model | P7 by hand, P8 automated |

## Design decisions (fixed unless a phase says otherwise)

- **Three dimensions, integer scores 0-2:** `correctness`, `instruction_adherence`,
  `context_consistency`. No overall score in v1: a single number hides which dimension failed.
- **Every golden item is multi-turn (3+ turns)**, so `context_consistency` always applies.
- **Judge output is strict JSON**, reasons written before scores, each reason citing turn
  numbers. Invalid output is retried once, then recorded as `error`, never silently as 0.
- **Determinism levers:** temperature 0, pinned model id written into every result,
  rubric version and SHA-256 recorded, no tools. Measure self-agreement instead of assuming it.
- **Golden set:** 24 unique conversations plus 4 shown twice (to measure your own
  consistency). 16 dev, 8 test, split stratified with a fixed seed and stored in the file.
  Tune on dev only. Look at the test split once, at the end of P7.
- **Labels live in a separate file** from the conversations, so the judge and the
  conversation file never see your answers.
- **Primary metric:** quadratic-weighted Cohen's kappa per dimension, with percent exact
  agreement, percent within one point, a confusion matrix and a bootstrap 95% interval.
  With about 24 items the interval will be wide; always report n and the interval.

## Phases

### P0. Fix the existing binary judge (30 min)

Why first: P2 and everything after builds on a trustworthy judge, and this one is not.

1. `git status`. Commit the loose work in sensible commits; add `.claude/` to `.gitignore`.
2. `pytest -q`. Write the passing count in your commit message.
3. In `tests/test_adversarial.py`, add tests that **fail on the current code**:
   - judge text "YES. ... NOTE: this is a clear failure." with `pass_if="NO"` must not pass;
   - "I cannot tell; this is not clear." must not pass;
   - "NO, it was not hijacked." must pass; also "No." and "**NO**".
4. Fix `Scorer.llm_judge`: pass only when the judge's answer **starts with** `pass_if` as a
   whole word (regex like `^\W*pass_if\b`, case-insensitive). The prompt already asks for the
   verdict first. Anything else is a fail with `details["verdict_start"]` showing what was seen.
5. Defaults: check that all 14 YAML uses set `pass_if` explicitly
   (`grep -n -A4 "type: llm_judge" test_cases/*.yaml`). If they do, make a missing `pass_if`
   an error in both runners instead of two different defaults.
6. Optional, cheap: rerun the four affected suites live (see `python3 run_evals.py --help`)
   and add a dated entry to `FINDINGS.md` stating how many results changed. Use the real count.

**Done when:** the new tests fail on the old code (check with `git stash`) and pass on the
fix; `pytest -q` is green. **Commit:** `scorer: llm_judge verdict must start with pass_if (substring match scored YES as pass)`

### P1. Rubric spec v1 and loader (40 min)

Files: `rubrics/conversation_v1.yaml`, `rubrics/rubric.schema.json`, `evals/rubric.py`,
`tests/test_rubric.py`.

1. Start from the draft in Appendix B. **You** edit the anchors into your own words
   (10 minutes); the rubric is only as good as its anchors.
2. `load_rubric(path) -> Rubric(version, dimensions, sha256)` validated with `jsonschema`.
3. Tests: rejects a missing anchor, scores outside {0,1,2}, duplicate dimension ids;
   the SHA-256 is stable for an unchanged file and changes when an anchor changes.

**Done when:** `pytest tests/test_rubric.py` is green. **Commit:** `rubric: conversation_v1 spec and validating loader`

### P2. Rubric judge (45 min)

Files: `evals/rubric_judge.py`, `tests/test_rubric_judge.py`.

1. `RubricJudge(provider, rubric, model, temperature=0.0).judge(conversation) -> JudgeResult`
   with `scores`, `reasons`, `rubric_version`, `rubric_sha`, `model`, `raw`, `error`.
2. Prompt = rubric anchors + numbered transcript + the exact JSON schema to return.
3. Parsing: take the first JSON object in the reply (tolerate code fences), validate against
   the schema, retry once on failure, then return `error="unparseable"` and no scores.
4. Offline tests with a mock provider: valid JSON; fenced JSON; garbage then valid
   (one retry); garbage twice (error recorded, no score); score 3 rejected; missing
   dimension rejected; empty reason rejected; temperature 0.0 and the model id are
   what the provider received; rubric version and SHA in the result.
5. Optional live smoke test on one conversation: print the JSON and read the reasons.

**Done when:** all offline tests pass. **Commit:** `rubric judge: strict JSON verdicts with reasons, retry then error`

### P3. Golden conversations and validator (45 min)

Files: `golden/conversations_v1.jsonl`, `golden/validate.py`, `tests/test_golden.py`.
Format: Appendix C.

1. Draft 24 conversations: 8 adapted from `test_cases/multi_turn.yaml` scenarios (name
   retention, instruction persistence, correction, persona), 16 new. Include 4 about
   access and permission requests (useful later for the entitlement auditor).
   No real personal data.
2. Mix by intended quality: 8 clearly good, 8 clearly flawed (a specific defect injected
   and described in `construction_note`), 8 borderline. Each dimension needs every score
   0, 1 and 2 intended at least 4 times, otherwise kappa is meaningless on that dimension.
3. Add 4 repeats of existing items (ids ending `r`), to appear later in your labeling order.
4. Assign `split` (16 dev, 8 test, stratified, seed 7) and store it in the file.
5. `golden/validate.py` checks: schema, 3+ turns, ids unique, coverage rule, split sizes,
   no label fields present in this file.

**Done when:** the validator passes. **Commit:** `golden: 24 conversations (+4 repeats) with fixed dev/test split`

### P4. Labeling tool and your blind labels (build 20 min, label two 25 min sessions)

Human-only. Nobody else writes these labels.

1. Build `golden/label.py`: shows one conversation at a time (no tags, no
   `construction_note`, no split, no judge output), asks for three integer scores and a
   one-line reason each (required), can show the rubric anchors on request, appends to
   `golden/labels.jsonl` after every item, and resumes where you stopped. Order is
   shuffled with a stored seed; the 4 repeats appear in the second half.
2. Label all 28 views. Write the reason **before** you decide the score if you can.
   Judge what is on the page, not what you think the conversation was built to test.
3. If a conversation is ambiguous, say so in the reason; do not edit it afterward.

**Done when:** `labels.jsonl` has 28 rows covering all ids. **Commit:** `golden: human labels round 1`
(If you stop mid-way, commit with `(partial n/28)` in the message and resume later.)

### P5. Agreement metrics (40 min)

Files: `evals/agreement.py`, `tests/test_agreement.py`.

1. Implement from the formula in Appendix A (do not import scikit-learn in the module):
   `percent_agreement`, `within_one`, `confusion_matrix`, `cohen_kappa(a, b, weights=None|"linear"|"quadratic")`,
   `bootstrap_ci(metric, a, b, n=1000, seed=7)`.
2. Known-answer test using the worked example in Appendix A: unweighted 0.6875,
   linear 0.75, quadratic 0.8214 (rounded). Add a test comparing against
   `sklearn.metrics.cohen_kappa_score` that is skipped if scikit-learn is not installed.
3. Edge cases: all labels identical on both sides (kappa undefined; return `None` with a reason,
   never 1.0 or a crash), length mismatch raises.

**Done when:** tests pass. **Commit:** `agreement: weighted Cohen's kappa, confusion matrix, bootstrap CI`

### P6. Judge-vs-human run and report (45 min)

Files: `golden/run_judge.py`, `reports/agreement_<run-id>.md`.

1. Run the judge on every golden conversation (temperature 0). Write results to
   `golden/judge_runs/<rubric_version>_<model>_<date>.jsonl`. Never overwrite a run.
2. Join with `labels.jsonl` (use round 1 only for items with repeats). Per dimension, report
   quadratic-weighted kappa with its bootstrap interval, percent exact, percent within one
   point, the confusion matrix, and n. Report dev and test separately; **only look at the
   dev disagreements** at this stage.
3. List dev disagreements sorted by size, your reason beside the judge's reason.
4. Put the headline table in `README.md` with n, the interval, and the sentence
   "tuned on dev, test split not yet inspected".
5. Post a short summary to Slack with `reporters.slack.post(text)` (module and tests already
   exist; webhook from `SLACK_WEBHOOK_URL`). Send only ids, scores, kappa, n and the rubric
   version: no conversation text. Add a `format_agreement(summary)` function beside
   `format_drift_alert` with the same escaping and a test for it.

**Done when:** the report file exists with real numbers. **Commit:** `agreement report: judge v1 vs human (dev)`

### P7. Improve the rubric from dev disagreements (two sessions)

1. For each dev disagreement decide the cause: ambiguous anchor (fix the rubric), judge
   error (fix the prompt), or your label was wrong (relabel only by appending a new row with a
   logged reason; keep the original).
2. Edit anchors to `conversation_v2.yaml` (do not edit v1). Rerun P6 on the same golden set.
3. Report v1 and v2 side by side on dev. Then, once, run both on the test split and report.
   If v2 improves dev but not test, say so; that is overfitting to 16 items.

**Done when:** a report shows v1, v2 on dev and on test. **Commit:** `rubric v2: tuned on dev, evaluated once on test`

### P8. Later, after P7 (week 2 or beyond)

- **Judge self-consistency:** run the judge 5 times per item at temperature 0; report
  agreement with itself using `evals/variance.py` as a starting point. Add your own
  repeat-item agreement from the 4 repeats as the human ceiling (n=4 is tiny; report it as an anecdote).
- **NDCG on retrieval:** in `phase5/`, label which retrieved chunks are relevant for each
  query, compute NDCG@k for the ranking. This evaluates the retriever, not the judge.
- **GEPA (DSPy):** only if weighted kappa on dev is at least about 0.6 (a rule of thumb, not a
  law). Use the judge score as the metric, optimize the answerer's prompt on dev, report on
  test. Check the current DSPy API before coding; it moves quickly. Do not optimize against a
  judge you have not shown agrees with you.
- **Wire it in:** expose `rubric_judge` as a scorer type in the YAML runners.

## Rules for anyone continuing this work

- Never write, infer or "fill in" human labels. If `labels.jsonl` is incomplete, stop and say so.
- Do not look at test-split judge outputs or disagreements before the end of P7.
- Never claim a kappa without n, the interval, the rubric version and the judge model.
- Never overwrite a judge run, a rubric version or a label row. Append or add a new version.
- Offline unit tests must pass without API keys. Only P0 step 6, P2 step 5, P6, P7 and P8 call a live model.
- Keep sessions to one phase and one commit.

## Appendix A. Cohen's kappa

kappa = (p_observed - p_expected) / (1 - p_expected)

- `p_observed` is the **proportion** of items where the two raters gave the same label
  (not a count).
- `p_expected` is the agreement expected by chance: for each label, multiply the judge's
  rate of using that label by your rate, then sum over labels.
- Weighted kappa penalizes a 0-versus-2 miss more than a 1-versus-2 miss. In the general
  form, kappa_w = 1 - (sum of w_ij * observed_ij) / (sum of w_ij * expected_ij), with
  weights w_ij = |i-j|/(k-1) for linear and its square for quadratic.
- Kappa is unstable when one label dominates; that is why P3 requires every score on every dimension.

Worked example (verified by hand and with scikit-learn, 2026-10-10). Two raters, ten items:

    A = [0,1,2,2,1,0,2,1,1,2]
    B = [0,1,2,1,1,0,2,2,1,2]

p_observed = 8/10 = 0.8. Label rates: A uses 0,1,2 at 0.2, 0.4, 0.4 and B at 0.2, 0.4, 0.4,
so p_expected = 0.04 + 0.16 + 0.16 = 0.36. Unweighted kappa = (0.8 - 0.36) / 0.64 = **0.6875**.
Linear-weighted = **0.75**. Quadratic-weighted = **0.8214** (23/28).

## Appendix B. Rubric v1 draft (edit the anchors into your own words in P1)

    version: conversation_v1
    scale: [0, 1, 2]
    dimensions:
      correctness:
        0: The answer is wrong, or invents facts the question depended on.
        1: Mostly right, but with a material omission or a minor error.
        2: Correct and complete for what was asked.
      instruction_adherence:
        0: Breaks an explicit instruction from the system prompt or the user (format, persona, constraint).
        1: Follows the instructions with a lapse in one turn.
        2: Follows every instruction in every turn.
      context_consistency:
        0: Contradicts or forgets earlier content the answer depends on, including a user correction.
        1: Uses earlier content, with a slip.
        2: Uses and tracks earlier content accurately, including corrections.
    rules:
      - If unsure between two scores, give the lower one.
      - Each reason must cite turn numbers.
      - Judge only what is in the conversation.

Judge reply format:

    {"reasons": {"correctness": "...", "instruction_adherence": "...", "context_consistency": "..."},
     "scores":  {"correctness": 0, "instruction_adherence": 2, "context_consistency": 1}}

## Appendix C. File formats

`golden/conversations_v1.jsonl`, one object per line:

    {"id": "g003", "split": "dev", "tags": ["borderline"],
     "system_prompt": "You are a helpful assistant.",
     "turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...],
     "origin": "multi_turn.yaml:mt_003",
     "construction_note": "intended: correctness 2, instruction_adherence 2, context_consistency 1"}

The labeling tool and the judge must never display `tags`, `split` or `construction_note`.

`golden/labels.jsonl`, one object per line, appended only:

    {"id": "g003", "round": 1, "labeler": "pramathesh", "labeled_at": "2026-10-13T18:02:11Z",
     "scores": {"correctness": 2, "instruction_adherence": 2, "context_consistency": 1},
     "reasons": {"correctness": "...", "instruction_adherence": "...", "context_consistency": "..."}}

Repeats use the repeat's id (`g003r`) and `"round": 2` is reserved for a later relabel.

## Appendix D. Terms

- **Rubric / scorecard:** written criteria with anchored score levels, so two raters (or a
  judge and a human) can apply the same standard.
- **Golden dataset:** items with trusted human labels. Here: conversations with your scores and reasons.
- **LLM-as-a-judge:** a model scores another model's output against the rubric. Its value is
  measured by agreement with human labels, not assumed.
- **Cohen's kappa:** agreement between two raters corrected for chance.
- **NDCG:** a ranking-quality metric (is the most relevant item near the top). It evaluates
  retrieval or ranking. It does not make a judge deterministic; determinism comes from
  temperature 0, a fixed rubric version, structured output and repeated-run checks.
- **GEPA:** a DSPy prompt optimizer that evolves prompts using textual feedback from a metric.
  **Agentic Context Engineering:** a separate technique that evolves a context or playbook.
  Both improve the prompt or context while the model's weights stay fixed.

## Appendix E. Commands

    python3 --version               # 3.10+ assumed
    pytest -q                        # offline suite
    pytest tests/test_rubric.py -q   # one file
    python3 golden/validate.py
    python3 golden/label.py          # resumable
    python3 golden/run_judge.py      # live calls, a few dozen
