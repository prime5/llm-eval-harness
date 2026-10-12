"""
P1: rubric loader. Offline.

pytest tests/test_rubric.py
"""
import pathlib
import textwrap

import pytest

from evals.rubric import Rubric, RubricError, load_rubric

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHIPPED = ROOT / "rubrics" / "conversation_v1.yaml"

VALID = """\
version: test_v1
scale: [0, 1, 2]
dimensions:
  correctness:
    0: wrong
    1: partly right
    2: right
  tone:
    0: rude
    1: curt
    2: polite
rules:
  - cite turns
"""


def write(tmp_path, text, name="r.yaml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return p


def test_shipped_rubric_loads_with_three_dimensions():
    r = load_rubric(SHIPPED)
    assert r.version == "conversation_v1"
    assert r.dimension_ids == ["correctness", "instruction_adherence", "context_consistency"]
    for anchors in r.dimensions.values():
        assert sorted(anchors) == [0, 1, 2] and all(anchors.values())
    assert len(r.sha256) == 64


def test_valid_rubric_fields(tmp_path):
    r = load_rubric(write(tmp_path, VALID))
    assert isinstance(r, Rubric)
    assert r.dimensions["tone"][2] == "polite"
    assert r.rules == ("cite turns",)
    assert r.scale == (0, 1, 2)


@pytest.mark.parametrize("bad_anchor_line", [
    "    1: partly right\n",   # drop the middle anchor
    "    0: wrong\n",          # drop the lowest
    "    2: right\n",          # drop the highest
])
def test_rejects_missing_anchor(tmp_path, bad_anchor_line):
    text = VALID.replace(bad_anchor_line, "", 1)
    with pytest.raises(RubricError, match="schema violation"):
        load_rubric(write(tmp_path, text))


def test_rejects_empty_anchor_text(tmp_path):
    with pytest.raises(RubricError, match="schema violation"):
        load_rubric(write(tmp_path, VALID.replace("1: partly right", "1: '   '")))


def test_rejects_anchor_score_outside_scale(tmp_path):
    text = VALID.replace("    2: right\n", "    2: right\n    3: superb\n", 1)
    with pytest.raises(RubricError, match="schema violation"):
        load_rubric(write(tmp_path, text))


def test_rejects_scale_other_than_0_1_2(tmp_path):
    with pytest.raises(RubricError, match="schema violation"):
        load_rubric(write(tmp_path, VALID.replace("[0, 1, 2]", "[0, 1, 2, 3]")))


def test_rejects_duplicate_dimension_ids(tmp_path):
    # the duplicate sits inside `dimensions`, before `rules`
    text = VALID.replace("rules:", "  correctness:\n    0: a\n    1: b\n    2: c\nrules:")
    with pytest.raises(RubricError, match="duplicate key 'correctness'"):
        load_rubric(write(tmp_path, text))


def test_rejects_duplicate_anchor_key(tmp_path):
    text = VALID.replace("    1: partly right\n", "    1: partly right\n    1: also one\n", 1)
    with pytest.raises(RubricError, match="duplicate key 1"):
        load_rubric(write(tmp_path, text))


def test_rejects_unknown_top_level_key_and_bad_dimension_id(tmp_path):
    with pytest.raises(RubricError, match="schema violation"):
        load_rubric(write(tmp_path, VALID + "extra: 1\n"))
    with pytest.raises(RubricError, match="schema violation"):
        load_rubric(write(tmp_path, VALID.replace("tone:", "Tone Of Voice:")))


def test_rejects_missing_file_and_bad_yaml_and_non_mapping(tmp_path):
    with pytest.raises(RubricError, match="cannot read"):
        load_rubric(tmp_path / "nope.yaml")
    with pytest.raises(RubricError, match="invalid YAML"):
        load_rubric(write(tmp_path, "a: [unclosed\n"))
    with pytest.raises(RubricError, match="top level"):
        load_rubric(write(tmp_path, "- just\n- a list\n"))


def test_sha256_is_stable_and_tracks_anchor_changes(tmp_path):
    a = load_rubric(write(tmp_path, VALID, "a.yaml"))
    a_again = load_rubric(tmp_path / "a.yaml")
    b = load_rubric(write(tmp_path, VALID.replace("partly right", "mostly right"), "b.yaml"))
    assert a.sha256 == a_again.sha256
    assert a.sha256 != b.sha256


def test_validate_scores(tmp_path):
    r = load_rubric(write(tmp_path, VALID))
    assert r.validate_scores({"correctness": 2, "tone": 0}) == {"correctness": 2, "tone": 0}
    for bad in [
        {"correctness": 2},                           # missing dimension
        {"correctness": 2, "tone": 1, "extra": 0},    # unknown dimension
        {"correctness": 3, "tone": 1},                # outside scale
        {"correctness": -1, "tone": 1},
        {"correctness": 1.5, "tone": 1},              # not an int
        {"correctness": "2", "tone": 1},
        {"correctness": True, "tone": 1},             # bool is not a score
        [2, 1],
    ]:
        with pytest.raises(RubricError):
            r.validate_scores(bad)


def test_render_lists_every_anchor_and_rule(tmp_path):
    text = load_rubric(write(tmp_path, VALID)).render()
    for fragment in ["correctness:", "  0: wrong", "  2: polite", "Rules:", "- cite turns"]:
        assert fragment in text


def test_sha256_is_of_the_file_bytes(tmp_path):
    import hashlib
    p = write(tmp_path, VALID)
    assert load_rubric(p).sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    # a comment-only edit still changes the hash: the record names exact file text
    q = write(tmp_path, "# note\n" + VALID, "q.yaml")
    assert load_rubric(q).sha256 != load_rubric(p).sha256
