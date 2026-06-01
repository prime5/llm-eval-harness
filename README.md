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
| 2 — Adversarial & Safety Testing | Prompt injection, hallucination detection, policy compliance | 🔜 |
| 3 — Multi-Agent Workflow Testing | Tool use validation, chain-of-thought consistency, pass@k | 🔜 |
| 4 — Observability & CI/CD Pipeline | GitHub Actions, score drift detection, metrics dashboard | 🔜 |

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

**Phase 3 — Multi-Agent Workflow Testing**
- Tool call validation (function calling correctness)
- Chain-of-thought consistency scoring
- pass@k metric implementation
- Non-deterministic workflow testing patterns

**Phase 4 — Observability & CI Pipeline**
- GitHub Actions workflow for eval runs on PR
- Score drift detection between model versions
- Metrics dashboard (Streamlit)
- Slack alerts on regression

---

## Background

This project grew out of 12+ years of cloud quality engineering — building test frameworks for identity sync pipelines at Palo Alto Networks, distributed data platforms at Workday, and BI systems at Birst. The same principles apply: define clear pass criteria, measure variance, make failures actionable, integrate with CI/CD. The difference is that LLMs require probabilistic thinking about correctness rather than binary assertions.

---

*Targeting roles in AI quality engineering at Anthropic, OpenAI, Meta, and Google.*
