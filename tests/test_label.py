"""Tests for the P4 labeling tool (golden/label.py). All offline."""
import json
from pathlib import Path

import pytest

from golden.label import (
    append_label,
    build_order,
    collect_item,
    get_order,
    is_repeat,
    already_labeled,
    parse_score,
    render_conversation,
    render_anchors,
    run,
)

REPO = Path(__file__).resolve().parent.parent
RUBRIC_PATH = REPO / "rubrics" / "conversation_v1.yaml"
GOLDEN_PATH = REPO / "golden" / "conversations_v1.jsonl"


def _items():
    return [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line.strip()]


def test_repeat_detection():
    assert is_repeat("g003r")
    assert not is_repeat("g003")


def test_order_repeats_in_second_half():
    order = build_order(_items(), seed=7)
    assert len(order) == 28
    repeats = [i for i in order if is_repeat(i)]
    base = [i for i in order if not is_repeat(i)]
    assert len(repeats) == 4
    assert len(base) == 24
    assert order == base + repeats  # repeats strictly after all base items


def test_order_deterministic_and_covers_all():
    items = _items()
    assert build_order(items, seed=7) == build_order(items, seed=7)
    assert set(build_order(items, seed=7)) == {it["id"] for it in items}


def test_get_order_stores_and_reuses(tmp_path):
    items = _items()
    path = tmp_path / "labeling_order.json"
    first = get_order(path, items, seed=7)
    manifest = json.loads(path.read_text())
    assert manifest["seed"] == 7
    assert manifest["order"] == first
    assert get_order(path, items, seed=99) == first  # stored seed wins, no reshuffle


def test_render_conversation_hides_metadata():
    item = _items()[0]
    text = render_conversation(item, 1, 28)
    assert item["id"] not in text
    assert item["construction_note"] not in text
    assert item["origin"] not in text
    # no metadata field labels leak into the display either
    for field in ("tags", "split", "construction_note", "origin", "system_prompt"):
        assert f'"{field}"' not in text
    assert "[system]" in text
    assert "turn 1" in text


def test_parse_score():
    assert parse_score("2", [0, 1, 2]) == 2
    assert parse_score(" 1 ", [0, 1, 2]) == 1
    assert parse_score("3", [0, 1, 2]) is None
    assert parse_score("no", [0, 1, 2]) is None
    assert parse_score("", [0, 1, 2]) is None


class _FakeInput:
    """Feed canned answers; 'anchors' injection tests the escape hatch."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def _rubric():
    from evals.rubric import load_rubric

    return load_rubric(RUBRIC_PATH)


def test_collect_item_reason_then_score(tmp_path):
    item = _items()[0]
    rubric = _rubric()
    fake = _FakeInput(["r1", "2", "r2", "1", "r3", "0"])
    scores, reasons = collect_item(item, rubric, fake, 1, 28)
    assert scores == {"correctness": 2, "instruction_adherence": 1, "context_consistency": 0}
    assert reasons == {"correctness": "r1", "instruction_adherence": "r2", "context_consistency": "r3"}
    # reason was asked before its score for every dimension (even index = reason, odd = score)
    assert len(fake.prompts) == 6
    assert all("reason" in p for p in fake.prompts[::2])
    assert all("score" in p for p in fake.prompts[1::2])
    assert all("reason" not in p.split("score")[0] for p in fake.prompts[1::2])


def test_collect_item_rejects_bad_score_and_empty_reason():
    item = _items()[0]
    rubric = _rubric()
    fake = _FakeInput(["", "r1", "5", "bogus", "2", "r2", "1", "r3", "0"])
    scores, reasons = collect_item(item, rubric, fake, 1, 28)
    assert scores["correctness"] == 2 and reasons["correctness"] == "r1"


def test_collect_item_anchors_escape_hatch():
    item = _items()[0]
    rubric = _rubric()
    fake = _FakeInput(["anchors", "r1", "2", "r2", "1", "r3", "0"])
    scores, _ = collect_item(item, rubric, fake, 1, 28)
    assert scores["correctness"] == 2
    assert render_anchors(rubric).count("correctness") >= 1


def test_collect_item_scores_validated_by_rubric():
    """The rubric's own validate_scores is the final guard on collected scores."""
    from evals.rubric import RubricError

    item = _items()[0]
    rubric = _rubric()
    fake = _FakeInput(["r1", "2", "r2", "1", "r3", "0"])
    scores, _ = collect_item(item, rubric, fake, 1, 28)
    assert rubric.validate_scores(scores) == scores
    with pytest.raises(RubricError):
        rubric.validate_scores({"correctness": 2})  # missing dimensions rejected


def test_append_label_schema(tmp_path):
    path = tmp_path / "labels.jsonl"
    append_label(path, "g003r", {"correctness": 2}, {"correctness": "r"})
    row = json.loads(path.read_text())
    assert row["id"] == "g003r"
    assert row["round"] == 1
    assert row["labeler"] == "pramathesh"
    assert row["labeled_at"].endswith("Z")
    assert set(row) == {"id", "round", "labeler", "labeled_at", "scores", "reasons"}


def _drive(labels_path, order_path, answers):
    return run(
        rubric_path=RUBRIC_PATH,
        golden_path=GOLDEN_PATH,
        labels_path=labels_path,
        order_path=order_path,
        input_fn=_FakeInput(answers),
    )


def test_run_labels_one_then_quits(tmp_path):
    labels = tmp_path / "labels.jsonl"
    order_p = tmp_path / "labeling_order.json"
    # two answers (reason+score for dim 1) then 'quit'
    assert _drive(labels, order_p, ["r1", "2", "quit"]) == 0
    assert not labels.exists()  # item not finished, nothing appended
    assert already_labeled(labels) == set()


def test_run_resumes_where_it_stopped(tmp_path):
    labels = tmp_path / "labels.jsonl"
    order_p = tmp_path / "labeling_order.json"
    answers = ["r", "2"] * 3  # one full item
    assert _drive(labels, order_p, answers) == 0
    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    assert len(rows) == 1
    first_id = rows[0]["id"]
    # second run skips the labeled item
    assert _drive(labels, order_p, answers) == 0
    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    assert len(rows) == 2
    assert {r["id"] for r in rows} == {first_id, rows[1]["id"]}
    assert first_id != rows[1]["id"]


def test_run_all_items_no_progress_file_leftover(tmp_path):
    labels = tmp_path / "labels.jsonl"
    order_p = tmp_path / "labeling_order.json"
    answers = ["reason cites turn 1", "2"] * 3 * 28
    assert _drive(labels, order_p, answers) == 0
    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    assert len(rows) == 28
    assert len({r["id"] for r in rows}) == 28
