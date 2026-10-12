"""LLM-as-a-judge against a versioned rubric.

The judge scores a multi-turn conversation on every rubric dimension,
returning integer scores with reasons as strict JSON. A reply that is not
parseable or fails validation is retried once; a second failure is recorded
as an error — never silently scored.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from providers.base import CompletionRequest

from .rubric import Rubric


@dataclass
class JudgeResult:
    scores: dict | None   # dimension id -> int score
    reasons: dict | None  # dimension id -> reason string
    rubric_version: str
    rubric_sha: str
    model: str
    raw: str              # the judge's raw reply (last attempt)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class RubricJudge:
    """Scores conversations with an LLM against a loaded rubric."""

    def __init__(self, provider, rubric: Rubric, model: str,
                 temperature: float = 0.0, max_tokens: int = 800):
        self.provider = provider
        self.rubric = rubric
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # -- public API -----------------------------------------------------

    def judge(self, conversation) -> JudgeResult:
        """Score a conversation; conversation is a dict with a 'turns' list
        (a bare list of turns is also accepted)."""
        prompt = self._build_prompt(conversation)
        reply = self._call(prompt)
        scores, reasons, err = self._parse_and_validate(reply)
        if err is not None:
            repair = (
                prompt
                + f"\n\nYour previous reply was invalid ({err}). "
                + "Reply with ONLY the JSON object, no other text."
            )
            reply = self._call(repair)
            scores, reasons, err = self._parse_and_validate(reply)
        if err is not None:
            return JudgeResult(
                scores=None,
                reasons=None,
                rubric_version=self.rubric.version,
                rubric_sha=self.rubric.sha256,
                model=self.model,
                raw=reply,
                error="unparseable",
            )
        return JudgeResult(
            scores=scores,
            reasons=reasons,
            rubric_version=self.rubric.version,
            rubric_sha=self.rubric.sha256,
            model=self.model,
            raw=reply,
            error=None,
        )

    # -- prompt ----------------------------------------------------------

    def _build_prompt(self, conversation) -> str:
        if isinstance(conversation, dict):
            turns = conversation.get("turns", [])
            system = conversation.get("system_prompt")
        else:
            turns = conversation
            system = None

        lines = [
            "You are an impartial judge scoring an AI assistant's conversation",
            "against the rubric below. Judge only what is in the conversation.",
            "",
            f"## Rubric (version {self.rubric.version})",
        ]
        for dim_id, anchors in self.rubric.dimensions.items():
            lines.append(f"### {dim_id}")
            for score in sorted(anchors):
                lines.append(f"- Score {score}: {anchors[score]}")
        lines += [
            "",
            "## Rules",
            "- If unsure between two scores, give the lower one.",
            "- Each reason must cite turn numbers.",
            "- Judge only what is in the conversation.",
            "",
        ]
        if system:
            lines += ["## System prompt", system, ""]
        lines.append("## Conversation")
        for i, turn in enumerate(turns, start=1):
            role = turn.get("role", "?")
            content = turn.get("content", "")
            lines.append(f"Turn {i} ({role}): {content}")
        lines += [
            "",
            "## Output",
            "Write your reasons BEFORE your scores. "
            "Reply with exactly one JSON object and no other text:",
            json.dumps({
                "reasons": {d: f"<cite turn numbers for {d}>" for d in self.rubric.dimensions},
                "scores": {d: 0 for d in self.rubric.dimensions},
            }, indent=2),
        ]
        return "\n".join(lines)

    # -- provider call ----------------------------------------------------

    def _call(self, prompt: str) -> str:
        resp = self.provider.complete(
            CompletionRequest(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )
        return resp.text

    # -- parsing and validation --------------------------------------------

    def _parse_and_validate(self, reply: str):
        """Return (scores, reasons, None) on success, (None, None, err) on failure."""
        obj = _extract_json(reply)
        if obj is None:
            return None, None, "no JSON object found"
        return self._validate(obj)

    def _validate(self, obj):
        if not isinstance(obj, dict):
            return None, None, "top level is not an object"
        reasons = obj.get("reasons")
        scores = obj.get("scores")
        if not isinstance(reasons, dict) or not isinstance(scores, dict):
            return None, None, "missing 'reasons' or 'scores' object"

        dim_ids = self.rubric.dimension_ids
        valid_scores = set(self.rubric.scale)

        for key in list(reasons) + list(scores):
            if key not in dim_ids:
                return None, None, f"unknown dimension {key!r}"

        out_scores, out_reasons = {}, {}
        for dim_id in dim_ids:
            if dim_id not in reasons:
                return None, None, f"missing reason for dimension {dim_id!r}"
            if dim_id not in scores:
                return None, None, f"missing score for dimension {dim_id!r}"
            reason = reasons[dim_id]
            if not isinstance(reason, str) or not reason.strip():
                return None, None, f"empty reason for dimension {dim_id!r}"
            score = scores[dim_id]
            if isinstance(score, bool) or not isinstance(score, int):
                return None, None, (
                    f"score for dimension {dim_id!r} is not an integer: {score!r}"
                )
            if score not in valid_scores:
                return None, None, (
                    f"score {score} for dimension {dim_id!r} "
                    f"is outside the rubric scale {sorted(valid_scores)}"
                )
            out_reasons[dim_id] = reason.strip()
            out_scores[dim_id] = score
        return out_scores, out_reasons, None


def _extract_json(text: str):
    """Return the first JSON object in text, tolerating code fences.

    Returns the parsed object, or None if there is none / it does not parse.
    """
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip(), flags=re.M)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    while start != -1:
        end = _match_brace(cleaned, start)
        if end is not None:
            try:
                obj = json.loads(cleaned[start:end])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        start = cleaned.find("{", start + 1)
    return None


def _match_brace(text: str, start: int):
    """Return the index just past the closing brace matching text[start],
    respecting string literals, or None if unbalanced."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
    return None
