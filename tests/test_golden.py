"""Tests for the golden conversation set (P3). All offline."""
import json
from pathlib import Path

from golden.validate import validate

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "conversations_v1.jsonl"


def _rows():
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def test_golden_file_valid():
    errors, _ = validate(GOLDEN)
    assert errors == []


def test_counts():
    rows = _rows()
    assert len(rows) == 28
    uniques = [r for r in rows if not r["id"].endswith("r")]
    repeats = [r for r in rows if r["id"].endswith("r")]
    assert len(uniques) == 24
    assert len(repeats) == 4


def test_split_sizes():
    rows = [r for r in _rows() if not r["id"].endswith("r")]
    dev = sum(1 for r in rows if r["split"] == "dev")
    test = sum(1 for r in rows if r["split"] == "test")
    assert (dev, test) == (16, 8)


def test_no_label_fields():
    for r in _rows():
        leaked = {"scores", "reasons", "labels", "label"} & set(r)
        assert not leaked, f"{r['id']}: {leaked}"


def test_every_conversation_is_multiturn():
    for r in _rows():
        assert len(r["turns"]) >= 3, r["id"]


def test_repeats_mirror_base():
    rows = {r["id"]: r for r in _rows()}
    for rid, r in rows.items():
        if rid.endswith("r"):
            base = rows[rid[:-1]]
            assert r["turns"] == base["turns"]
            assert r["split"] == base["split"]


def test_validator_catches_bad_file(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({
        "id": "x1", "split": "dev", "tags": ["good"],
        "system_prompt": "s",
        "turns": [{"role": "user", "content": "hi"}],
        "origin": "t",
        "construction_note": "intended: correctness 2, instruction_adherence 2, context_consistency 2",
    }) + "\n")
    errors, _ = validate(bad)
    assert any("3+ turns" in e for e in errors)
