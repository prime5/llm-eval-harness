"""
Phase 5 — RAG Evaluation Harness
Session 3: Eval suite — retrieval precision, faithfulness, context sufficiency,
           and adversarial retrieval

Usage:
    python3 phase5/eval_suite.py
    python3 phase5/eval_suite.py --output phase5/data/rag_eval_results.json
    python3 phase5/eval_suite.py --verbose

Exit codes:
    0 — all evals passed
    1 — one or more evals failed (for CI gate in Session 4)
"""

import argparse
import json
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from phase5.rag_chain import RAGChain, RAGResult

load_dotenv()

# ── Thresholds ────────────────────────────────────────────────────────────────
PRECISION_PASS_RATE   = 1.0   # all precision tests must pass
FAITHFULNESS_MIN      = 0.8   # LLM judge score per answer
SUFFICIENCY_PASS_RATE = 1.0   # all sufficiency tests must pass
ADVERSARIAL_PASS_RATE = 0.5   # adversarial is harder — flag below 50%


# ── Test case definitions ─────────────────────────────────────────────────────

PRECISION_TESTS = [
    # (query, expected_top1_source)
    ("What makes a startup idea worth pursuing?",         "startup_ideas"),
    ("How does writing help clarify your thoughts?",      "writing_and_thinking"),
    ("Does hard work compound over time?",                "how_to_work_hard"),
    ("What role does curiosity play in doing great work?","great_work"),
    ("Why do the best startup ideas seem like bad ideas?","startup_ideas"),
    ("What happens when you rewrite something?",          "writing_and_thinking"),
]

SUFFICIENCY_TESTS = [
    # (query, expect_answer: True = should answer, False = should say INSUFFICIENT CONTEXT)
    ("What is the relationship between interest and effort in great work?", True),
    ("How does writing reveal gaps in your understanding?",                 True),
    ("What programming language does Paul Graham recommend for startups?",  False),
    ("What is Paul Graham's net worth?",                                    False),
    ("How many employees did Y Combinator have in 2023?",                  False),
]

ADVERSARIAL_TESTS = [
    # (misleading_query, correct_source, description)
    (
        "How does compounding work in financial investing?",
        "how_to_work_hard",
        "Uses 'compounding' from how_to_work_hard but sounds like a finance question",
    ),
    (
        "What makes a great piece of creative writing stand out from others?",
        "writing_and_thinking",
        "Asks about 'creative writing' — retriever should still hit writing_and_thinking not great_work",
    ),
    (
        "How do you identify problems that other people have missed?",
        "startup_ideas",
        "Framed as general problem-finding, but answer lives in startup_ideas",
    ),
]


# ── Result data classes ───────────────────────────────────────────────────────

@dataclass
class PrecisionResult:
    query:            str
    expected_source:  str
    actual_top1:      str
    distance:         float
    passed:           bool


@dataclass
class FaithfulnessResult:
    query:               str
    answer:              str
    score:               float          # 0.0–1.0
    unsupported_claims:  list[str]
    passed:              bool


@dataclass
class SufficiencyResult:
    query:            str
    expected_answer:  bool              # True = should give answer
    got_answer:       bool              # True = chain gave an answer
    passed:           bool
    chain_response:   str


@dataclass
class AdversarialResult:
    query:           str
    description:     str
    correct_source:  str
    actual_top1:     str
    distance:        float
    passed:          bool


@dataclass
class EvalReport:
    run_at:              str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    precision_results:   list[PrecisionResult]   = field(default_factory=list)
    faithfulness_results:list[FaithfulnessResult]= field(default_factory=list)
    sufficiency_results: list[SufficiencyResult] = field(default_factory=list)
    adversarial_results: list[AdversarialResult] = field(default_factory=list)
    passed:              bool = True

    # Summary metrics (filled in by run())
    precision_pass_rate:    float = 0.0
    faithfulness_avg_score: float = 0.0
    sufficiency_pass_rate:  float = 0.0
    adversarial_pass_rate:  float = 0.0


# ── Faithfulness judge ────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are a strict factual auditor. Your job is to identify claims in an AI-generated
answer that are NOT supported by the provided context.

Return a JSON object with two fields:
  "unsupported_claims": list of strings (claims not in context; empty list if fully grounded)
  "score": float 0.0–1.0 (1.0 = fully grounded, 0.0 = entirely hallucinated)

Be specific. Quote the exact unsupported phrase. Do not penalize reasonable inference
from stated facts — only flag claims that contradict or go beyond the context."""

JUDGE_USER = """\
Context:
{context}

Answer to audit:
{answer}

Return only the JSON object."""


def judge_faithfulness(client: OpenAI, result: RAGResult) -> FaithfulnessResult:
    """Use GPT as a judge to score answer faithfulness against retrieved context."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(
                context=result.context_text(),
                answer=result.answer,
            )},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = json.loads(response.choices[0].message.content)
    score = float(raw.get("score", 0.0))
    unsupported = raw.get("unsupported_claims", [])
    return FaithfulnessResult(
        query=result.query,
        answer=result.answer,
        score=score,
        unsupported_claims=unsupported,
        passed=score >= FAITHFULNESS_MIN,
    )


# ── Eval runners ──────────────────────────────────────────────────────────────

def run_precision(chain: RAGChain, verbose: bool) -> list[PrecisionResult]:
    print("\n── Eval 1: Retrieval Precision ───────────────────────────────────────")
    results = []
    for query, expected in PRECISION_TESTS:
        r = chain.run(query)
        top1   = r.chunks[0].source
        dist   = r.chunks[0].distance
        passed = top1 == expected
        results.append(PrecisionResult(query, expected, top1, dist, passed))
        status = "✓" if passed else "✗"
        print(f"  {status} [{dist:.3f}] {query[:55]:<55} → {top1} (expected: {expected})")
    return results


def run_faithfulness(chain: RAGChain, client: OpenAI, verbose: bool) -> list[FaithfulnessResult]:
    print("\n── Eval 2: Answer Faithfulness (LLM-as-judge) ───────────────────────")
    # Reuse precision queries — same pipeline, now judge the answers
    queries = [q for q, _ in PRECISION_TESTS[:4]]
    results = []
    for query in queries:
        r      = chain.run(query)
        result = judge_faithfulness(client, r)
        status = "✓" if result.passed else "✗"
        print(f"  {status} score={result.score:.2f}  {query[:55]}")
        if verbose and result.unsupported_claims:
            for claim in result.unsupported_claims:
                print(f"       ⚠ unsupported: {claim}")
        results.append(result)
    return results


def run_sufficiency(chain: RAGChain, verbose: bool) -> list[SufficiencyResult]:
    print("\n── Eval 3: Context Sufficiency ───────────────────────────────────────")
    results = []
    for query, expect_answer in SUFFICIENCY_TESTS:
        r          = chain.run(query)
        got_answer = "INSUFFICIENT CONTEXT" not in r.answer
        passed     = got_answer == expect_answer
        status     = "✓" if passed else "✗"
        label      = "should answer" if expect_answer else "should refuse"
        actual     = "answered" if got_answer else "refused"
        print(f"  {status} [{label} → {actual}] {query[:50]}")
        if verbose:
            print(f"       response: {r.answer[:100]}")
        results.append(SufficiencyResult(query, expect_answer, got_answer, passed, r.answer))
    return results


def run_adversarial(chain: RAGChain, verbose: bool) -> list[AdversarialResult]:
    print("\n── Eval 4: Adversarial Retrieval ─────────────────────────────────────")
    results = []
    for query, correct_source, description in ADVERSARIAL_TESTS:
        r      = chain.run(query)
        top1   = r.chunks[0].source
        dist   = r.chunks[0].distance
        passed = top1 == correct_source
        status = "✓" if passed else "✗"
        print(f"  {status} [{dist:.3f}] {query[:55]}")
        print(f"       {description}")
        print(f"       got: {top1}  expected: {correct_source}")
        results.append(AdversarialResult(query, description, correct_source, top1, dist, passed))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run(verbose: bool = False) -> EvalReport:
    chain  = RAGChain()
    client = OpenAI()

    report = EvalReport()
    report.precision_results    = run_precision(chain, verbose)
    report.faithfulness_results = run_faithfulness(chain, client, verbose)
    report.sufficiency_results  = run_sufficiency(chain, verbose)
    report.adversarial_results  = run_adversarial(chain, verbose)

    # Compute summary metrics
    def rate(lst, key="passed"):
        return sum(1 for x in lst if getattr(x, key)) / len(lst) if lst else 0.0

    report.precision_pass_rate    = rate(report.precision_results)
    report.faithfulness_avg_score = (
        sum(r.score for r in report.faithfulness_results) / len(report.faithfulness_results)
        if report.faithfulness_results else 0.0
    )
    report.sufficiency_pass_rate  = rate(report.sufficiency_results)
    report.adversarial_pass_rate  = rate(report.adversarial_results)

    # Overall pass/fail
    report.passed = (
        report.precision_pass_rate    >= PRECISION_PASS_RATE
        and report.faithfulness_avg_score >= FAITHFULNESS_MIN
        and report.sufficiency_pass_rate  >= SUFFICIENCY_PASS_RATE
        and report.adversarial_pass_rate  >= ADVERSARIAL_PASS_RATE
    )
    return report


def print_summary(report: EvalReport) -> None:
    status = "✓ PASSED" if report.passed else "✗ FAILED"
    print(f"\n{'═' * 60}")
    print(f"  RAG Eval Summary — {status}")
    print(f"{'═' * 60}")
    print(f"  Retrieval Precision    {report.precision_pass_rate * 100:.0f}%"
          f"  (threshold: {PRECISION_PASS_RATE * 100:.0f}%)")
    print(f"  Answer Faithfulness    {report.faithfulness_avg_score:.2f}"
          f"  (threshold: {FAITHFULNESS_MIN:.2f})")
    print(f"  Context Sufficiency    {report.sufficiency_pass_rate * 100:.0f}%"
          f"  (threshold: {SUFFICIENCY_PASS_RATE * 100:.0f}%)")
    print(f"  Adversarial Retrieval  {report.adversarial_pass_rate * 100:.0f}%"
          f"  (threshold: {ADVERSARIAL_PASS_RATE * 100:.0f}%)")
    print(f"{'═' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 5 RAG eval suite")
    parser.add_argument("--output",  default="phase5/data/rag_eval_results.json",
                        help="Path to write JSON results")
    parser.add_argument("--verbose", action="store_true",
                        help="Print answer text and unsupported claims")
    args = parser.parse_args()

    report = run(verbose=args.verbose)
    print_summary(report)

    # Write JSON results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"Results written to {out_path}")

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
