# llm-eval-harness

A production-quality LLM evaluation framework built to validate language model behavior — covering response accuracy, instruction following, variance under repeated runs, and safety baselines.

Built by [Pramathesh Malshe](https://linkedin.com/in/pramathesh), Principal SDET with 12+ years of experience in cloud quality engineering at Palo Alto Networks and Workday. This project applies enterprise-grade SDET practices to the problem of testing non-deterministic AI systems.

---

## Why This Exists

Testing LLMs is fundamentally different from testing deterministic software:

- **Non-determinism** — the same prompt produces different outputs across runs
- **No binary pass/fail** — responses exist on a quality spectrum
- **Prompt sensitivity** — small wording changes can drastically shift behavior
- **Safety requirements** — models must refuse harmful requests consistently

This harness treats those challenges as first-class engineering problems, not afterthoughts.

---

## Project Phases

| Phase | Focus | Status |
|-------|-------|--------|
| **1 — Core Eval Framework** | Runner, scorers, variance analysis, YAML test cases, HTML reports | ✅ Done |
| **2 — Adversarial & Safety Testing** | Prompt injection, hallucination detection, policy compliance | ✅ Done |
| **3 — Multi-Agent Workflow Testing** | Tool call validation, multi-turn conversation testing, AgentRunner | ✅ Done |
| **4 — Observability & CI/CD Pipeline** | GitHub Actions, score drift detection, Streamlit dashboard, Slack alerts | ✅ Done |

---

## Phase 1: Core Eval Framework

### Architecture

```
llm-eval-harness/
├── providers/          # LLM abstraction layer (OpenAI, Anthropic, extensible)
│   ├── base.py         # Abstract interface — add any provider by subclassing
│   ├── openai_provider.py
│   └── anthropic_provider.py
├── evals/
│   ├── runner.py       # Loads YAML suites, drives providers, collects results
│   ├── scorer.py       # Composable scoring strategies (exact, keyword, regex, length, safety)
│   └── variance.py     # Runs prompt N times, measures pass_rate / pass@k / latency percentiles
├── test_cases/         # YAML test suites — human-readable, version-controlled
│   ├── basic_qa.yaml
│   └── instruction_following.yaml
├── reporters/
│   ├── html_reporter.py   # Visual report
│   └── json_reporter.py   # CI-friendly machine output
├── tests/              # Offline unit + integration tests (no API calls)
│   ├── test_scorer.py
│   └── test_runner.py
└── run_evals.py        # CLI entry point
```

### Key Design Decisions

**Scorer composition** — multiple scoring strategies combine via `aggregate()`. A test case can require keywords AND check length AND verify no forbidden content, with configurable pass thresholds per scorer.

**Variance analysis** — every test case runs N times (configurable). Reports `pass_rate`, `pass@k`, unique response ratio, and latency p50/p95. A model that passes 3/5 runs is `FLAKY`; 5/5 is `STABLE`.

**Provider abstraction** — `BaseProvider` defines a single `complete()` interface. Switching from OpenAI to Anthropic is one flag: `--provider anthropic`. Adding a new provider (Gemini, Mistral) requires implementing one method.

**YAML-driven test cases** — test cases live in version-controlled YAML, not code. Non-engineers can add cases. Scorers are declarative.

---

## Setup

```bash
git clone https://github.com/pramathesh/llm-eval-harness
cd llm-eval-harness
pip install -r requirements.txt

cp .env.example .env
# Edit .env — add your OPENAI_API_KEY or ANTHROPIC_API_KEY
```

---

## Running Evals

```bash
# Run all suites against OpenAI (default)
python run_evals.py

# Run against Anthropic Claude
python run_evals.py --provider anthropic

# Run specific suite only
python run_evals.py --suite test_cases/basic_qa.yaml

# Run Phase 2 adversarial suites
python run_evals.py --adversarial

# Run Phase 3 agent suites (tool calling + multi-turn)
python run_evals.py --agent

# Run ALL suites — Phase 1 + Phase 2 + Phase 3
python run_evals.py --all

# CI mode — exit 1 if pass rate below 80%
python run_evals.py --min-pass-rate 0.8

# JSON output only (for pipeline ingestion)
python run_evals.py --format json
```

### Sample Output

```
🧪 LLM Eval Harness — Phase 1
Provider: openai

Running suite: test_cases/basic_qa.yaml
  ✅ [qa_001] Capital city factual recall
  ✅ [qa_002] Simple arithmetic
  ✅ [qa_003] Response length guardrail — concise answer
  ✅ [qa_004] Instruction following — list format
  ✅ [qa_005] No hallucination on verifiable fact
  ✅ [qa_006] Response must not contain apology for benign question

============================================================
Provider : openai
Total    : 11
Passed   : 10
Failed   : 1
Pass Rate: 91%
============================================================

ID       Description                                Result  Score  Variance          Latency
qa_001   Capital city factual recall                PASS    1.00   STABLE (100%)     342ms
qa_002   Simple arithmetic                          PASS    1.00   STABLE (100%)     289ms
...

📄 HTML report: eval_report.html
📊 JSON report: eval_results.json
```

---

## Running Unit Tests (No API Key Needed)

```bash
# Run all offline tests
pytest tests/ -v

# Run scorer tests only
pytest tests/test_scorer.py -v

# Run with HTML report
pytest tests/ --html=pytest_report.html
```

---

## Writing Test Cases

```yaml
cases:
  - id: my_001
    description: Verify model knows Python basics
    tags: [smoke, factual]
    prompt: "What does the 'yield' keyword do in Python?"
    variance_runs: 3          # run 3 times to check consistency
    scorers:
      - type: contains_keywords
        keywords: [generator, iterator, lazy]
        require_all: false
        threshold: 0.5        # at least 1 of 3 keywords
      - type: length_check
        min_words: 10
        max_words: 200
      - type: no_forbidden_content
        forbidden: ["I don't know", "I'm not sure"]
```

### Available Scorers

| Scorer | Required fields | Description |
|--------|----------------|-------------|
| `exact_match` | `expected` | Response must exactly equal expected |
| `contains_keywords` | `keywords` | Response must contain specified terms |
| `regex_match` | `pattern` | Response must match regex |
| `length_check` | `min_words`, `max_words` | Word count must be in range |
| `no_forbidden_content` | `forbidden` | Response must not contain these strings |
| `tool_call_match` | `expected_function` | Model must call the named function with correct args |
| `json_schema_match` | `schema` | Response must be valid JSON matching the schema |
| `no_tool_call` | *(none)* | Model must NOT call any tool |
| `llm_judge` | `judge_prompt`, `pass_if` | Second LLM evaluates response by meaning |

---

## Phase 3: Multi-Agent Workflow Testing

Phase 3 adds an `AgentRunner` that handles two new case types: `tool_call` and `multi_turn`.
Run them with `python run_evals.py --agent`.

### Tool Call Test Cases

Validate that the model calls the right function with the correct arguments.

```yaml
cases:
  - id: tc_001
    type: tool_call
    description: Weather query triggers get_weather with correct location
    tags: [tool-call, smoke]
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
          location: San Francisco     # substring match, case-insensitive
      # To assert NO tool is called:
      # - type: no_tool_call
```

The `tool_call_match` scorer:
- Scores **0.0** if the expected function was never called
- Scores **1.0** if the function was called and all checked args match
- Scores **0.5–1.0** (partial credit) if the function was called but some args differ
- Set `require_all_args: true` to require all expected args to match for a pass

### Multi-Turn Test Cases

Drive a full conversation and score each assistant reply against its own scorers.
The final score is the `min` across all scored turns — every turn must pass.

```yaml
cases:
  - id: mt_001
    type: multi_turn
    description: Model retains user name across turns
    system_prompt: "You are a helpful assistant."
    turns:
      - user: "Hi, my name is Alice."
        scorers: []               # nothing to score here
      - user: "What is my name?"
        scorers:
          - type: contains_keywords
            keywords: [Alice]
```

### Architecture

```
evals/
├── runner.py       # Single-turn: EvalRunner (unchanged from Phase 1/2)
└── agent_runner.py # Multi-turn + tool call: AgentRunner, AgentEvalResult, TurnResult
```

`AgentEvalResult` duck-types `EvalResult`'s key interface (`.passed`, `.prompt`, `.response`,
`.latency_ms`, `.scorer_result`, `.tags`) so all existing reporters and CLI display work
transparently without modification.

---

## Adding a New Provider

```python
# providers/gemini_provider.py
from .base import BaseProvider, CompletionRequest, CompletionResponse

class GeminiProvider(BaseProvider):
    @property
    def name(self): return "gemini"

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        # call Gemini API here
        ...
```

Then register it in `providers/__init__.py`.

---

## Roadmap

**Phase 2 — Adversarial & Safety Testing**
- Prompt injection detection
- Hallucination scoring against ground truth
- Policy/safety category classification
- Jailbreak resistance measurement

**Phase 3 — Multi-Agent Workflow Testing** ✅ Done
- Tool call validation (function calling correctness via `tool_call_match` scorer)
- Multi-turn conversation testing (context retention, instruction persistence, persona consistency)
- `AgentRunner` with message history management
- `json_schema_match` scorer for structured output validation
- Both OpenAI and Anthropic providers support function/tool definitions

**Phase 4 — Observability & CI Pipeline** ✅ Done
- GitHub Actions: two-job workflow (`offline-tests` always, `eval-ci` on main/when key available)
- Score drift detection: `detect_drift.py` compares baseline vs current, classifies regressions/improvements/score drops
- Streamlit dashboard: `streamlit run dashboard.py` — summary metrics, tag breakdown, score/latency charts, drift comparison
- Slack alerts: `detect_drift.py --slack-webhook URL` posts regression summary to Slack
- `--save-baseline` / `--check-drift` flags on `run_evals.py` for inline workflow integration

---

## Phase 4: Observability & CI/CD Pipeline

### Score Drift Detection

Compare any two eval runs to surface regressions before they reach production.

```bash
# Save current results as baseline
python3 run_evals.py --save-baseline

# On the next run, check for drift automatically
python3 run_evals.py --check-drift

# Or run standalone drift check
python3 detect_drift.py \
  --baseline eval_baseline.json \
  --current eval_results.json \
  --threshold 0.1 \
  --fail-on-regression

# Post Slack alert on regression
python3 detect_drift.py --slack-webhook https://hooks.slack.com/services/...
```

Change types detected per case:

| Type | Meaning |
|------|---------|
| `regression` | PASS → FAIL (most severe, fails CI) |
| `improvement` | FAIL → PASS (positive signal) |
| `score_drop` | Still passing but score fell > threshold |
| `score_gain` | Still passing but score rose > threshold |
| `stable` | No meaningful change |
| `new` / `removed` | Case added or removed since baseline |

### Streamlit Dashboard

```bash
# Install dashboard dependencies
pip install streamlit pandas

# Launch
streamlit run dashboard.py

# Point at a specific report
streamlit run dashboard.py -- --report path/to/eval_results.json
```

Shows: summary metrics, pass rate by tag, score distribution, per-case latency,
score drift comparison (if baseline exists), and a filterable results table.

### GitHub Actions CI

The workflow at `.github/workflows/eval_ci.yml` runs two jobs on every push/PR:

1. **`offline-tests`** — runs `pytest tests/` with no API key; always executes
2. **`eval-ci`** — runs a fast eval subset (`basic_qa` + `instruction_following`) against the live API, checks drift against the stored baseline artifact, and saves the new run as the updated baseline

**Setup:**
1. Add `OPENAI_API_KEY` to your GitHub repository secrets
2. Set the `RUN_LIVE_EVALS` repository variable to `true` to enable job 2
3. HTML and JSON reports are uploaded as artifacts on every run

The baseline artifact persists between runs via `actions/upload-artifact`. Drift
detection on CI catches regressions introduced by model version changes or
prompt modifications before they merge.

---

## Background

This project grew out of 12+ years of cloud quality engineering — building test frameworks for identity sync pipelines at Palo Alto Networks, distributed data platforms at Workday, and BI systems at Birst. The same principles apply: define clear pass criteria, measure variance, make failures actionable, integrate with CI/CD. The difference is that LLMs require probabilistic thinking about correctness rather than binary assertions.

---

*Targeting roles in AI quality engineering at Anthropic, OpenAI, Meta, and Google.*
