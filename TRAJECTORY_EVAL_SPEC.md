# Trajectory eval spec — task `ag_001` (currency conversion)

Design only. No code yet. Chain being defined:

    trajectory -> expected checkpoints -> allowed tools -> forbidden tools -> evaluator -> regression threshold

## 0. Why this task, and what the harness does today

Source: `test_cases/tool_calling.yaml` case `tc_003` ("How many euros is 100 US dollars?", tools `convert_currency` and `search_web`).
It is the smallest case with a real trajectory (pick tool -> read result -> answer), one clearly wrong tool, and a checkable ground truth.

What the code does now (read from `evals/agent_runner.py` and `evals/scorer.py`):

- `_run_tool_call` makes ONE provider call and scores the tool calls it returns. Tools are never executed, no result goes back to the model, no second step exists. So there is no trajectory yet, only a first decision.
- `_run_multi_turn` replays assistant TEXT only; tool calls are recorded on the TurnResult but never put back into history.
- `Scorer.tool_call_match` only looks at calls whose name equals `expected_function`. Extra calls are ignored.
- Arg matching is bidirectional substring (`expected in actual or actual in expected`).

Two consequences, both reproduced by running the scorer (not inferred):

1. tc_003 PASSES (score 1.0) when the model calls `convert_currency` AND `search_web`. The wrong tool is invisible to it.
2. `amount=1` matches expected `100` ("1" is a substring of "100"), score 1.0.

Neither is a bug for a single-step selection check. Both are the reason a trajectory evaluator needs its own checks: "forbidden" has to be a first-class rule, and numbers have to compare as numbers.

## 1. Trajectory

An ordered list of steps; a step is one model call plus the tool results returned to it.

    Trajectory:
      steps: [Step]
      terminal_reason: final_answer | max_steps | error
    Step:
      index, assistant_text
      tool_calls:   [{id, name, arguments}]      # already normalized by providers/base.py
      tool_results: [{id, name, content, error}] # produced by the mock registry
      prompt_tokens, completion_tokens, latency_ms

Loop: send messages + tools -> if the reply has tool_calls, execute each against the mock registry, append the assistant message and one `tool` message per call, repeat -> stop on a reply with no tool_calls, or at `max_steps`.
Expected happy path: 2 steps (call `convert_currency`, then final answer).

Mock registry (deterministic, in the YAML): `convert_currency` returns `{amount: 81.37, rate: 0.8137, from: USD, to: EUR}`.
The rate is deliberately NOT one a model would recall, so an answer built from memory is detectable.
`search_web` returns an empty result list; calling it is the failure being tested.

## 2. Expected checkpoints (milestones, not an exact script)

| id | check | gate |
|---|---|---|
| selects_dedicated_tool | at least one `convert_currency` call | hard |
| args_correct | amount == 100 (numeric), from_currency == USD, to_currency == EUR (case-insensitive exact; ISO codes) | hard |
| tool_before_answer | first step has a tool call; no final text before a tool result exists | hard |
| answer_matches_tool_result | final text contains 81.37 (tolerance 0.01) | hard |
| terminates_in_budget | terminal_reason == final_answer, steps <= 3 | soft |
| no_unsupported_numbers | no number in the final text other than 100, 81.37, 0.8137 (catches invented dates/rates) | soft |

## 3. Allowed tools

Policy over the OFFERED set, not a different set:

- `convert_currency`: min 1, max 2 calls (one retry tolerated).
- Hard cap `max_steps: 4`.

## 4. Forbidden tools

- `search_web` (offered as a distractor; a dedicated deterministic tool exists).
- Any tool name not in the offered set (hallucinated tool).
- Any call whose arguments miss a schema-required key.
- Violations are a hard fail regardless of the final answer. That is the difference from the current scorer.

## 5. Evaluator (cheapest layer first)

1. Deterministic trajectory checks: pure functions over a Trajectory (allowed/forbidden, arg validation, ordering, budget, termination). Unit-testable with scripted fake providers, no API calls, same style as `tests/test_agent_runner.py`.
2. Outcome check: numeric extraction from the final text vs the mock result.
3. LLM judge only for what code can't decide: "does the answer convey the tool result without unsupported claims (invented source, date, caveats)?" Rubric 0-2. REPORT-ONLY until it agrees with ~20 hand-labeled trajectories (Cohen's kappa; start by requiring >= 0.6, adjust once you see the data).

Output: a result exposing `passed`, `score`, `scorer_result.details = {checkpoints, violations, steps}` so reporters and `DriftDetector` work unchanged.
`passed = no hard violation AND all hard checkpoints hit`; `score` = fraction of all checkpoints hit.

## 6. Regression threshold

Sampling is on (provider default temperature 0.7), so one run says little. Run each case N times (`evals/variance.py` already does this for single-step cases).

- Hard invariants (forbidden/unknown tool, schema violation, wrong args): zero tolerance across the N runs. If the BASELINE has a violation, fix the prompt or tool descriptions first; don't bake it in.
- Task success (hard checkpoints + outcome), N=10, clean 10/10 baseline: warn at 8 or 7 of 10, fail at <= 6 of 10.
  Why 6: one-sided Fisher exact vs 10/10 gives p = 0.24 (8/10), 0.105 (7/10), 0.043 (6/10), 0.016 (5/10). At N=10 only big drops are statistically visible; this is a tripwire, not a significance test.
  To detect 100% -> 80% reliably you need N=30 (p = 0.012). So: N=10 on PRs, N=30 when refreshing the baseline.
- Soft metrics (warn only): median steps <= 3, mean tokens within +25% of baseline, judge mean >= baseline - 0.25.
- Reuse of existing drift code: feed `pass_rate` in as the case `score` and set case `passed` = hard-clean AND success above the fail line. A drop from 1.0 to 0.8 then shows as `score_drop` under the existing 0.1 threshold, and a flip to fail as `regression`.
- Thresholds live in the case YAML (`gate: hard|soft`), as data, like the existing per-scorer `threshold`.

All numbers above are starting values to be re-derived from the first real baseline runs.

## 7. Proposed case YAML

    - id: ag_001
      type: trajectory
      description: Currency conversion - dedicated tool, answer faithful to tool result
      tags: [trajectory, tool-selection, faithfulness]
      prompt: "How many euros is 100 US dollars?"
      runs: 10
      max_steps: 4
      tools: [convert_currency, search_web]        # same definitions as tc_003
      mock_tools:
        convert_currency: {returns: {amount: 81.37, rate: 0.8137, from: USD, to: EUR}}
        search_web:       {returns: {results: []}}
      trajectory:
        allowed_tools:   {convert_currency: {min: 1, max: 2}}
        forbidden_tools: [search_web]
        forbid_unknown_tools: true
        checkpoints:
          - {id: selects_dedicated_tool, gate: hard, tool: convert_currency}
          - {id: args_correct, gate: hard, tool: convert_currency,
             args: {amount: {eq: 100}, from_currency: {eq_ci: USD}, to_currency: {eq_ci: EUR}}}
          - {id: tool_before_answer, gate: hard, order: [convert_currency, final_answer]}
          - {id: answer_matches_tool_result, gate: hard, final_contains_number: {value: 81.37, tol: 0.01}}
          - {id: terminates_in_budget, gate: soft, steps_lte: 3}
          - {id: no_unsupported_numbers, gate: soft, allowed_numbers: [100, 81.37, 0.8137]}
      evaluator:
        deterministic: [allowed, forbidden, checkpoints]
        judge: {rubric: faithfulness_0_2, gate: report_only}
      regression:
        hard_invariants: zero_tolerance
        success: {warn_at_or_below: 8, fail_at_or_below: 6, of: 10}
        soft: {median_steps_max: 3, tokens_pct: 25}

## 8. Build order (each a 30-45 min session ending in a commit)

1. Trajectory types + loop runner + mock registry, OpenAI provider only, tested with a scripted fake provider. (Anthropic needs tool_use/tool_result block translation; later.)
2. Deterministic evaluator as pure functions. First test: the tc_003 "both tools called" case must FAIL here.
3. `ag_001` YAML + N-run aggregation + reporter/drift hookup; live baseline run; commit the baseline.
4. Judge rubric + ~20 hand-labeled trajectories + kappa; then a perturbation variant (first `convert_currency` call returns an error -> expect one retry or an honest "couldn't convert").
