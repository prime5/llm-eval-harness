"""P5: agreement metrics between two raters (judge vs human).

Primary metric is quadratic-weighted Cohen's kappa per dimension, with
percent exact agreement, percent within one point, a confusion matrix and a
bootstrap 95% interval. No third-party statistics imports: the formulas are
implemented from Appendix A of RUBRIC_JUDGE_PLAN.md.

Conventions:
- ``a`` and ``b`` are parallel lists of integer scores (e.g. judge, human).
- Labels are the rubric scale; pass ``labels`` explicitly if a level never
  appears, otherwise the confusion matrix would silently drop it.
- kappa is undefined when chance agreement is 1.0 (e.g. every label
  identical on both sides): ``cohen_kappa`` returns None in that case,
  never 1.0 and never a crash. Check ``is not None`` before reporting.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence


def _check_lengths(a: Sequence[int], b: Sequence[int]) -> None:
    if len(a) != len(b):
        raise ValueError(f"score lists must be the same length, got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("score lists must not be empty")


def percent_agreement(a: Sequence[int], b: Sequence[int]) -> float:
    """Proportion of items where the two raters gave the same score."""
    _check_lengths(a, b)
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def within_one(a: Sequence[int], b: Sequence[int]) -> float:
    """Proportion of items where the two scores differ by at most one point."""
    _check_lengths(a, b)
    return sum(1 for x, y in zip(a, b) if abs(x - y) <= 1) / len(a)


def confusion_matrix(a: Sequence[int], b: Sequence[int], labels: Sequence[int] = (0, 1, 2)) -> dict:
    """Counts of (a_label, b_label) pairs.

    Returns {"labels": [...], "matrix": [[...]]} where matrix[i][j] counts
    items with a == labels[i] and b == labels[j].
    """
    _check_lengths(a, b)
    labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        if x not in index or y not in index:
            raise ValueError(f"score {(x, y)} outside labels {labels}")
        matrix[index[x]][index[y]] += 1
    return {"labels": labels, "matrix": matrix}


def _weights(kind: str | None, k: int) -> list[list[float]]:
    if kind is None:
        return [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    if kind == "linear":
        return [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    if kind == "quadratic":
        return [[(abs(i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)]
    raise ValueError(f"weights must be None, 'linear' or 'quadratic', got {kind!r}")


def cohen_kappa(
    a: Sequence[int], b: Sequence[int], weights: str | None = None, labels: Sequence[int] = (0, 1, 2)
) -> float | None:
    """Weighted Cohen's kappa between two raters.

    kappa = 1 - (sum w_ij * observed_ij) / (sum w_ij * expected_ij).
    ``weights``: None (unweighted: any disagreement counts fully),
    "linear" (|i-j|/(k-1)) or "quadratic" (its square).
    Returns None when kappa is undefined (chance agreement is 1.0).
    """
    _check_lengths(a, b)
    cm = confusion_matrix(a, b, labels=labels)
    labels = cm["labels"]
    k = len(labels)
    n = len(a)
    w = _weights(weights, k)

    observed = [[cell / n for cell in row] for row in cm["matrix"]]
    row_sums = [sum(row) for row in observed]
    col_sums = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    expected = [[row_sums[i] * col_sums[j] for j in range(k)] for i in range(k)]

    num = sum(w[i][j] * observed[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * expected[i][j] for i in range(k) for j in range(k))
    if den == 0:
        return None  # undefined: e.g. every label identical on both sides
    return 1.0 - num / den


def bootstrap_ci(
    metric: Callable[[Sequence[int], Sequence[int]], float | None],
    a: Sequence[int],
    b: Sequence[int],
    n: int = 1000,
    seed: int = 7,
) -> tuple[float | None, float | None]:
    """Bootstrap 95% interval for a metric over paired scores.

    Resamples (a_i, b_i) pairs with replacement ``n`` times with a fixed
    seed. Resamples where the metric is undefined are skipped; if fewer
    than 10 usable resamples remain, returns (None, None).
    """
    _check_lengths(a, b)
    rng = random.Random(seed)
    pairs = list(zip(a, b))
    values = []
    for _ in range(n):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        sa = [x for x, _ in sample]
        sb = [y for _, y in sample]
        value = metric(sa, sb)
        if value is not None and math.isfinite(value):
            values.append(value)
    if len(values) < 10:
        return (None, None)
    values.sort()
    lo = values[int(0.025 * len(values))]
    hi = values[int(0.975 * len(values))]
    return (lo, hi)
