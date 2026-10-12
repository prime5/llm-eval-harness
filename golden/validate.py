"""Validate golden/conversations_v1.jsonl.

Checks: schema, 3+ turns per conversation, unique ids, the 4 repeats,
the fixed 16/8 dev/test split, the per-dimension score-coverage rule, and
that no label fields leaked into the conversation file.

Usage: python3 golden/validate.py [path]
Exit 0 when valid, 1 with error messages otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REQUIRED_KEYS = {"id", "split", "tags", "system_prompt", "turns",
                 "origin", "construction_note"}
LABEL_KEYS = {"scores", "reasons", "labels", "label"}  # must never appear here
DIMS = ("correctness", "instruction_adherence", "context_consistency")
INTENDED_RE = re.compile(
    r"intended:\s*correctness\s+([012]),\s*"
    r"instruction_adherence\s+([012]),\s*"
    r"context_consistency\s+([012])"
)


def validate(path) -> list[str]:
    errors: list[str] = []
    path = Path(path)
    lines = path.read_text().splitlines()
    rows = []
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append(f"line {n}: invalid JSON: {exc}")

    ids = [r.get("id") for r in rows if isinstance(r, dict)]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        errors.append(f"duplicate ids: {dupees}")

    by_id = {}
    for n, r in enumerate(rows, start=1):
        if not isinstance(r, dict):
            errors.append(f"line {n}: not an object")
            continue
        rid = r.get("id", f"line {n}")
        by_id[rid] = r
        missing = REQUIRED_KEYS - set(r)
        if missing:
            errors.append(f"{rid}: missing keys {sorted(missing)}")
        leaked = LABEL_KEYS & set(r)
        if leaked:
            errors.append(f"{rid}: label fields leaked into conversation file: {sorted(leaked)}")
        if r.get("split") not in ("dev", "test"):
            errors.append(f"{rid}: bad split {r.get('split')!r}")
        tags = r.get("tags")
        if not isinstance(tags, list) or not tags or not all(isinstance(t, str) for t in tags):
            errors.append(f"{rid}: tags must be a non-empty list of strings")
        turns = r.get("turns")
        if not isinstance(turns, list) or len(turns) < 3:
            errors.append(f"{rid}: need 3+ turns, got {len(turns) if isinstance(turns, list) else turns!r}")
        else:
            for i, t in enumerate(turns):
                if not isinstance(t, dict) or t.get("role") not in ("user", "assistant") \
                        or not isinstance(t.get("content"), str):
                    errors.append(f"{rid} turn {i+1}: need {{role, content}} with role user|assistant")

    # repeats: ids ending in 'r' mirror a base item
    uniques = [r for r in rows if isinstance(r, dict) and not str(r.get("id", "")).endswith("r")]
    repeats = [r for r in rows if isinstance(r, dict) and str(r.get("id", "")).endswith("r")]
    if len(repeats) != 4:
        errors.append(f"expected 4 repeats, found {len(repeats)}")
    for r in repeats:
        base_id = r["id"][:-1]
        base = by_id.get(base_id)
        if base is None:
            errors.append(f"{r['id']}: base item {base_id} not found")
            continue
        if r["turns"] != base["turns"] or r["system_prompt"] != base["system_prompt"]:
            errors.append(f"{r['id']}: content differs from base {base_id}")
        if r["split"] != base["split"]:
            errors.append(f"{r['id']}: split {r['split']} != base split {base['split']}")

    # fixed split sizes over the 24 unique items
    dev = sum(1 for r in uniques if r.get("split") == "dev")
    test = sum(1 for r in uniques if r.get("split") == "test")
    if (dev, test) != (16, 8):
        errors.append(f"unique split is {dev} dev / {test} test, expected 16 / 8")

    # coverage: every dimension needs every score 0/1/2 at least 4 times
    counts = {d: {0: 0, 1: 0, 2: 0} for d in DIMS}
    for r in uniques:
        m = INTENDED_RE.search(r.get("construction_note", ""))
        if not m:
            errors.append(f"{r.get('id')}: construction_note lacks parseable intended scores")
            continue
        for dim, score in zip(DIMS, m.groups()):
            counts[dim][int(score)] += 1
    for dim in DIMS:
        for score in (0, 1, 2):
            if counts[dim][score] < 4:
                errors.append(
                    f"coverage: {dim} score {score} intended only "
                    f"{counts[dim][score]}x (need >= 4)"
                )
    return errors, counts


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent / "conversations_v1.jsonl"
    )
    errors, counts = validate(path)
    print(f"{path}:")
    for dim in DIMS:
        c = counts[dim]
        print(f"  {dim}: 0x{c[0]} 1x{c[1]} 2x{c[2]}")
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("OK: schema, splits, repeats, coverage all valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
