"""Tests for P5 agreement metrics (evals/agreement.py). All offline."""
import pytest

from evals.agreement import (
    bootstrap_ci,
    cohen_kappa,
    confusion_matrix,
    percent_agreement,
    within_one,
)

# Worked example from RUBRIC_JUDGE_PLAN.md Appendix A (verified by hand + sklearn).
A = [0, 1, 2, 2, 1, 0, 2, 1, 1, 2]
B = [0, 1, 2, 1, 1, 0, 2, 2, 1, 2]


def test_percent_agreement():
    assert percent_agreement(A, B) == pytest.approx(0.8)


def test_within_one():
    assert within_one(A, B) == pytest.approx(1.0)
    assert within_one([0, 0], [2, 1]) == pytest.approx(0.5)


def test_confusion_matrix():
    cm = confusion_matrix([0, 1, 1, 2], [0, 1, 2, 2])
    assert cm["labels"] == [0, 1, 2]
    assert cm["matrix"] == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
    with pytest.raises(ValueError):
        confusion_matrix([0, 3], [0, 0])  # score outside labels


def test_cohen_kappa_known_answers():
    assert cohen_kappa(A, B) == pytest.approx(0.6875)
    assert cohen_kappa(A, B, weights="linear") == pytest.approx(0.75)
    assert cohen_kappa(A, B, weights="quadratic") == pytest.approx(0.8214, abs=1e-4)
    with pytest.raises(ValueError):
        cohen_kappa(A, B, weights="cubic")


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
    # constant raters: chance agreement is 1.0, kappa undefined -> None, never 1.0
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) is None
    assert cohen_kappa([2, 2, 2, 2], [2, 2, 2, 2], weights="quadratic") is None


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        percent_agreement([0, 1], [0])
    with pytest.raises(ValueError):
        cohen_kappa([0], [0, 1])
    with pytest.raises(ValueError):
        bootstrap_ci(percent_agreement, [0], [0, 1])


def test_bootstrap_ci_deterministic_and_sane():
    lo, hi = bootstrap_ci(lambda a, b: cohen_kappa(a, b, weights="quadratic"), A, B)
    lo2, hi2 = bootstrap_ci(lambda a, b: cohen_kappa(a, b, weights="quadratic"), A, B)
    assert (lo, hi) == (lo2, hi2)  # fixed seed -> reproducible
    point = cohen_kappa(A, B, weights="quadratic")
    assert lo <= point <= hi
    assert lo < hi


def test_bootstrap_ci_undefined_metric():
    lo, hi = bootstrap_ci(cohen_kappa, [1, 1, 1], [1, 1, 1])
    assert (lo, hi) == (None, None)


def test_cohen_kappa_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics", reason="scikit-learn not installed")
    from sklearn.metrics import cohen_kappa_score

    assert cohen_kappa(A, B) == pytest.approx(cohen_kappa_score(A, B))
    assert cohen_kappa(A, B, weights="linear") == pytest.approx(
        cohen_kappa_score(A, B, weights="linear")
    )
    assert cohen_kappa(A, B, weights="quadratic") == pytest.approx(
        cohen_kappa_score(A, B, weights="quadratic")
    )
