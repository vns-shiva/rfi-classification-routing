"""Rater-agnostic, domain-agnostic inter-annotator agreement statistics.

Implements Krippendorff's alpha from the coincidence-matrix formulation
(Krippendorff, *Content Analysis*, 3rd ed., ch. 12; Krippendorff 2011,
"Computing Krippendorff's Alpha-Reliability"), plus companion metrics
(percent agreement, Cohen's kappa, quadratic-weighted kappa, confusion
matrices, MASI/Jaccard for set-valued fields). No project imports — this
module is testable against hand-derived reference values with no corpus
fixtures, and is reused by annotator_agreement.py to compute per-field
RFILabel agreement.

Input contract for `units` passed to `krippendorff_alpha`: a sequence of
per-unit sequences, one inner value per rater in a fixed rater order, with
`None` for a missing annotation. This is Krippendorff's reliability-data
matrix transposed (units x raters instead of raters x units) — the shape
that makes missing data and 3+ raters free rather than special-cased.

Degenerate cases are never silently coerced to 0.0 or 1.0:
- fewer than 2 pairable (non-missing) values across all units -> alpha is
  undefined ("insufficient_data").
- every pairable value is the same category (no variance at all, so the
  expected-disagreement denominator is zero) -> alpha is undefined
  ("no_variance"), never reported as a fake 1.0 or 0.0.
- alpha is not clamped: perfect systematic disagreement produces a negative
  value, which is correct and must be preserved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AlphaResult:
    alpha: float | None
    metric: str
    n_units: int
    n_units_dropped: int
    n_values: int
    n_raters: int
    observed_disagreement: float
    expected_disagreement: float
    categories: list = field(default_factory=list)
    undefined_reason: str | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_bootstrap_undefined: int = 0


def unit_coincidence(values, k_index: dict) -> np.ndarray:
    """K x K contribution of one unit's non-missing values to the coincidence
    matrix. Returns an all-zero matrix if fewer than 2 non-missing values
    (the unit is unpairable and dropped, per Krippendorff's standard rule).

    Each ordered pair of distinct value-slots (i, j), i != j, within the unit
    contributes 1/(m-1) to cell (category(i), category(j)), where m is the
    number of non-missing values in the unit. Summed over the whole matrix,
    one unit therefore contributes exactly m to the grand total.
    """
    non_missing = [v for v in values if v is not None]
    m = len(non_missing)
    K = len(k_index)
    mat = np.zeros((K, K))
    if m < 2:
        return mat
    idx = [k_index[v] for v in non_missing]
    weight = 1.0 / (m - 1)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            mat[idx[i], idx[j]] += weight
    return mat


def coincidence_matrix(units, categories) -> np.ndarray:
    """Sum of every unit's coincidence contribution over the given category
    order. Does not track dropped units — callers needing those counts
    inspect units directly (see krippendorff_alpha)."""
    k_index = {c: i for i, c in enumerate(categories)}
    K = len(categories)
    total = np.zeros((K, K))
    for values in units:
        total += unit_coincidence(values, k_index)
    return total


def delta_squared_matrix(metric: str, categories, marginals=None, distance=None) -> np.ndarray:
    """K x K squared-difference matrix for the given metric.

    - distance: a callable(a, b) -> float overrides `metric`, used for
      custom domains (MASI on sets, 1-cosine on text embeddings). Used
      as-is (NOT squared) — this matches the Passonneau (2006) / nltk
      convention of plugging a distance directly into Krippendorff's delta
      slot, so a MASI- or cosine-based alpha computed here is the same
      statistic the literature reports and is comparable to it. The
      distance must satisfy delta(c, c) == 0 (enforced below).
    - "nominal": 0 if identical else 1.
    - "ordinal": requires `marginals` (the coincidence matrix's row sums, in
      the same declared category order as `categories`) and MUST be
      recomputed from the marginals of whatever data it is scored against
      (including every bootstrap replicate) — it is not a fixed function of
      category identity the way nominal/interval/ratio are.
    - "interval": (c - k)^2 on the numeric category values.
    - "ratio": ((c - k)/(c + k))^2, 0 where c == k == 0.
    """
    K = len(categories)
    if distance is not None:
        mat = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                mat[i, j] = float(distance(categories[i], categories[j]))
        if not np.allclose(np.diag(mat), 0.0):
            raise ValueError(
                "custom distance must satisfy delta(c, c) == 0 for every category "
                f"(got diagonal {np.diag(mat).tolist()})"
            )
        return mat

    if metric == "nominal":
        mat = np.ones((K, K))
        np.fill_diagonal(mat, 0.0)
        return mat

    if metric == "ordinal":
        if marginals is None:
            raise ValueError("ordinal metric requires marginals")
        marginals = np.asarray(marginals, dtype=float)
        cum = np.concatenate([[0.0], np.cumsum(marginals)])
        mat = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                lo, hi = (i, j) if i <= j else (j, i)
                s = cum[hi + 1] - cum[lo]
                mat[i, j] = (s - (marginals[i] + marginals[j]) / 2.0) ** 2
        return mat

    if metric == "interval":
        vals = np.array(categories, dtype=float)
        return (vals[:, None] - vals[None, :]) ** 2

    if metric == "ratio":
        vals = np.array(categories, dtype=float)
        if np.any(vals < 0):
            raise ValueError(
                f"ratio metric requires non-negative category values, got {categories!r}"
            )
        denom = vals[:, None] + vals[None, :]
        diff = vals[:, None] - vals[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denom == 0, 0.0, diff / np.where(denom == 0, 1.0, denom))
        return ratio ** 2

    raise ValueError(f"unknown metric: {metric!r}")


def _sorted_categories(units):
    values = [v for unit in units for v in unit if v is not None]
    try:
        return sorted(set(values))
    except TypeError:
        # unorderable (e.g. mixed types) -> stable order of first appearance
        seen: list = []
        for v in values:
            if v not in seen:
                seen.append(v)
        return seen


def krippendorff_alpha(
    units,
    *,
    metric: str = "nominal",
    categories=None,
    distance=None,
    delta_matrix: np.ndarray | None = None,
    n_bootstrap: int = 0,
    seed: int = 0,
) -> AlphaResult:
    """Krippendorff's alpha over `units` (see module docstring for the input
    contract). For "ordinal", `categories` MUST be passed explicitly in the
    declared low-to-high order (e.g. vocabulary.URGENCY_TIERS) — it is not
    inferred by sorting. For "interval"/"ratio", categories default to the
    sorted distinct numeric values observed if not given.
    """
    if metric == "ordinal" and categories is None:
        raise ValueError(
            "metric='ordinal' requires categories to be passed explicitly in "
            "declared low-to-high order (e.g. vocabulary.URGENCY_TIERS) — "
            "sorting them alphabetically would silently change the statistic"
        )

    for u in units:
        for v in u:
            if isinstance(v, float) and math.isnan(v):
                raise ValueError(
                    "NaN is not a valid label value (the missing marker is None); "
                    "a NaN would otherwise be silently treated as its own category "
                    "or poison the alpha computation"
                )

    n_raters = max((len(u) for u in units), default=0)
    n_units_total = len(units)

    if categories is None:
        categories = _sorted_categories(units)
    categories = list(categories)
    if len(set(categories)) != len(categories):
        raise ValueError(f"categories must not contain duplicates: {categories!r}")
    K = len(categories)
    k_index = {c: i for i, c in enumerate(categories)}

    per_unit_contrib = []
    n_units_dropped = 0
    n_values = 0
    for u in units:
        non_missing = [v for v in u if v is not None]
        if len(non_missing) < 2:
            n_units_dropped += 1
            per_unit_contrib.append(np.zeros((K, K)))
        else:
            n_values += len(non_missing)
            per_unit_contrib.append(unit_coincidence(u, k_index))

    n_units = n_units_total - n_units_dropped

    def _undefined(reason, D_o=0.0, D_e=0.0):
        return AlphaResult(
            alpha=None, metric=metric, n_units=n_units, n_units_dropped=n_units_dropped,
            n_values=n_values, n_raters=n_raters, observed_disagreement=D_o,
            expected_disagreement=D_e, categories=categories, undefined_reason=reason,
        )

    if K == 0 or n_units == 0 or n_values < 2:
        return _undefined("insufficient_data")

    O = sum(per_unit_contrib)
    n = float(O.sum())
    if n <= 1:
        return _undefined("insufficient_data")
    marginals = O.sum(axis=1)

    if delta_matrix is not None:
        delta_sq = delta_matrix
    else:
        delta_sq = delta_squared_matrix(metric, categories, marginals=marginals, distance=distance)

    sum_o_delta = float(np.sum(O * delta_sq))
    sum_marginal_delta = float(np.sum(np.outer(marginals, marginals) * delta_sq))

    if not (np.isfinite(sum_o_delta) and np.isfinite(sum_marginal_delta)):
        return _undefined("non_finite_delta")

    D_o_report = sum_o_delta / n
    D_e_report = sum_marginal_delta / (n * (n - 1)) if n > 1 else 0.0

    if sum_marginal_delta == 0:
        return _undefined("no_variance", D_o=D_o_report, D_e=0.0)

    alpha = 1.0 - (n - 1) * sum_o_delta / sum_marginal_delta

    if not np.isfinite(alpha):
        return _undefined("non_finite_result", D_o=D_o_report, D_e=D_e_report)

    ci_low = ci_high = None
    n_bootstrap_undefined = 0
    if n_bootstrap and n_bootstrap > 0 and n_units > 0:
        ci_low, ci_high, n_bootstrap_undefined = bootstrap_unit_ci(
            per_unit_contrib, metric, categories, distance=distance,
            delta_matrix=delta_matrix, n_bootstrap=n_bootstrap, seed=seed,
        )

    return AlphaResult(
        alpha=alpha, metric=metric, n_units=n_units, n_units_dropped=n_units_dropped,
        n_values=n_values, n_raters=n_raters, observed_disagreement=D_o_report,
        expected_disagreement=D_e_report, categories=categories, undefined_reason=None,
        ci_low=ci_low, ci_high=ci_high, n_bootstrap_undefined=n_bootstrap_undefined,
    )


def bootstrap_unit_ci(
    per_unit_contrib: list[np.ndarray],
    metric: str,
    categories,
    *,
    distance=None,
    delta_matrix: np.ndarray | None = None,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> tuple[float | None, float | None, int]:
    """Unit-level percentile bootstrap CI for alpha. Resamples unit indices
    with replacement B times; each replicate's coincidence matrix is a
    weighted sum of the precomputed per-unit contributions (vectorized via a
    single multinomial-counts @ stacked-contributions matmul), so B=2000 stays
    fast even though this is a from-scratch, non-vendored implementation.

    `delta_matrix`, when given, overrides both `metric` and `distance` and is
    held fixed across replicates — it must be the same override the point
    estimate used, or the CI would answer a different question than the
    alpha it is meant to bracket. For "ordinal" without an override,
    delta-squared depends on the resampled marginals and is rebuilt every
    replicate — freezing it would silently bias the CI, since the ordinal
    delta matrix is a function of the data, not just category identity.
    """
    n_units = len(per_unit_contrib)
    if n_bootstrap <= 0 or n_units == 0:
        return None, None, 0

    K = per_unit_contrib[0].shape[0]
    stack = np.stack(per_unit_contrib, axis=0).reshape(n_units, K * K)

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=n_bootstrap)
    O_flat = counts @ stack
    Os = O_flat.reshape(n_bootstrap, K, K)

    static_delta = None
    if delta_matrix is not None:
        static_delta = delta_matrix
    elif metric != "ordinal":
        static_delta = delta_squared_matrix(metric, categories, distance=distance)

    alphas = []
    n_undefined = 0
    for b in range(n_bootstrap):
        O = Os[b]
        n = float(O.sum())
        marginals = O.sum(axis=1)
        if delta_matrix is None and metric == "ordinal":
            delta_sq = delta_squared_matrix("ordinal", categories, marginals=marginals)
        else:
            delta_sq = static_delta
        sum_o_delta = float(np.sum(O * delta_sq))
        sum_marginal_delta = float(np.sum(np.outer(marginals, marginals) * delta_sq))
        if sum_marginal_delta == 0 or n <= 1:
            n_undefined += 1
            continue
        alphas.append(1.0 - (n - 1) * sum_o_delta / sum_marginal_delta)

    if not alphas:
        return None, None, n_undefined
    lo, hi = np.percentile(alphas, [2.5, 97.5])
    return float(lo), float(hi), n_undefined


def percent_agreement(a: list, b: list) -> float:
    """Fraction of paired, non-missing positions where a[i] == b[i]."""
    if len(a) != len(b):
        raise ValueError(f"a and b must be the same length, got {len(a)} and {len(b)}")
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def confusion_matrix(a: list, b: list, categories) -> tuple[np.ndarray, list]:
    """K x K matrix counting (a[i], b[i]) pairs over non-missing positions."""
    if len(a) != len(b):
        raise ValueError(f"a and b must be the same length, got {len(a)} and {len(b)}")
    categories = list(categories)
    k_index = {c: i for i, c in enumerate(categories)}
    K = len(categories)
    mat = np.zeros((K, K), dtype=int)
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        mat[k_index[x], k_index[y]] += 1
    return mat, categories


def cohens_kappa(a: list, b: list, categories) -> float | None:
    """Unweighted Cohen's kappa for two raters. None if undefined (no
    pairable data, or zero expected agreement)."""
    mat, _ = confusion_matrix(a, b, categories)
    n = mat.sum()
    if n == 0:
        return None
    po = np.trace(mat) / n
    row_marg = mat.sum(axis=1) / n
    col_marg = mat.sum(axis=0) / n
    pe = float(np.sum(row_marg * col_marg))
    if pe == 1.0:
        return None
    return float((po - pe) / (1 - pe))


def quadratic_weighted_kappa(a: list, b: list, ordered_categories) -> float | None:
    """Cohen's kappa with quadratic weights over an explicitly ordered
    category list (index position is the ordinal value)."""
    ordered_categories = list(ordered_categories)
    mat, _ = confusion_matrix(a, b, ordered_categories)
    n = mat.sum()
    if n == 0:
        return None
    K = len(ordered_categories)
    idx = np.arange(K)
    weights = (idx[:, None] - idx[None, :]) ** 2
    row_marg = mat.sum(axis=1)
    col_marg = mat.sum(axis=0)
    expected = np.outer(row_marg, col_marg) / n
    denom = float(np.sum(weights * expected))
    if denom == 0:
        return None
    observed = float(np.sum(weights * mat))
    return float(1.0 - observed / denom)


def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard distance (1 - |intersection|/|union|). 0 for two empty sets
    (treated as identical, not undefined)."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def masi_distance(a: frozenset, b: frozenset) -> float:
    """MASI distance (Passonneau 2006): 1 - Jaccard * M, where M is 1 for
    identical sets, 2/3 when one is a proper subset of the other, 1/3 when
    they overlap without a subset relation, and 0 when disjoint. 0 for two
    empty sets."""
    if not a and not b:
        return 0.0
    if a == b:
        m = 1.0
    elif a <= b or b <= a:
        m = 2.0 / 3.0
    elif a & b:
        m = 1.0 / 3.0
    else:
        m = 0.0
    j = jaccard(a, b)
    return 1.0 - (1.0 - j) * m


def set_micro_prf(pairs: list[tuple[frozenset, frozenset]]) -> dict:
    """Micro-averaged precision/recall/F1 treating the first element of each
    pair as reference and the second as prediction, pooling counts across all
    pairs before dividing (so units with larger sets aren't down-weighted)."""
    tp = fp = fn = 0
    for ref, pred in pairs:
        tp += len(ref & pred)
        fp += len(pred - ref)
        fn += len(ref - pred)
    precision_defined = (tp + fp) > 0
    recall_defined = (tp + fn) > 0
    precision = tp / (tp + fp) if precision_defined else float("nan")
    recall = tp / (tp + fn) if recall_defined else float("nan")
    if precision_defined and recall_defined:
        # Both are real numbers here (possibly both 0.0); only 0/0 (no true
        # positives and no false positives/negatives predicted) needs the
        # explicit 0.0 fallback instead of a ZeroDivisionError.
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        f1 = float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
