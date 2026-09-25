"""Tests for agreement_metrics.py, written before anything depends on the
numbers it produces (per the module's own docstring caveat: no reference
Krippendorff-alpha package is installed in this environment, so correctness
is established three independent ways rather than by comparison to a
remembered published constant):

(a) an independent, brute-force O(n^2) reference implementation of alpha
    written directly in this file (not importing agreement_metrics'
    coincidence-matrix machinery), cross-checked across ~200 seeded random
    inputs per metric, including ragged units, 3 raters, and missing values;
(b) hand-derived worked examples (small enough to check with pencil and
    paper) plus analytic sanity properties (perfect agreement -> 1.0,
    systematic disagreement -> negative, large-N independent random -> ~0);
(c) a tight cross-check of quadratic_weighted_kappa against
    sklearn.metrics.cohen_kappa_score(weights="quadratic"), which is the same
    statistic under a different name, so it should match almost exactly
    (unlike plain alpha vs. Cohen's kappa, which use different chance models
    and are not asserted equal anywhere in this file);
(d) degenerate-case coverage (no_variance, insufficient_data, and an ordinal
    bootstrap check that a per-replicate delta-squared recompute differs from
    a wrongly-frozen one on a skewed fixture);
(e) MASI/Jaccard identities computed by hand.

Run with: python -m unittest discover -s code/tests -t code
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from agreement_metrics import (
    AlphaResult,
    bootstrap_unit_ci,
    cohens_kappa,
    confusion_matrix,
    delta_squared_matrix,
    jaccard,
    krippendorff_alpha,
    masi_distance,
    percent_agreement,
    quadratic_weighted_kappa,
    set_micro_prf,
    unit_coincidence,
)


# ---------------------------------------------------------------------------
# (a) independent brute-force reference implementation
# ---------------------------------------------------------------------------

def _nominal_delta(a, b):
    return 0.0 if a == b else 1.0


def _interval_delta(a, b):
    return (a - b) ** 2


def _ratio_delta(a, b):
    if a + b == 0:
        return 0.0
    return ((a - b) / (a + b)) ** 2


def _brute_force_alpha(units, delta_fn):
    """Independent implementation: loops over pairs of raw values directly,
    never builds a coincidence matrix or a category index. Do is the mean,
    over pairable units, of the within-unit mean pairwise delta; De is the
    mean pairwise delta over ALL pairable values in the dataset regardless of
    unit (this is what turns per-category marginal counts into a sum in the
    matrix formulation; here it is just a flat double loop).
    """
    n = 0
    do_num = 0.0
    flat = []
    for u in units:
        vals = [v for v in u if v is not None]
        m = len(vals)
        if m < 2:
            continue
        n += m
        flat.extend(vals)
        s = 0.0
        for i in range(m):
            for j in range(m):
                if i != j:
                    s += delta_fn(vals[i], vals[j])
        do_num += s / (m - 1)
    if n <= 1 or not flat:
        return None
    do = do_num / n
    de_num = 0.0
    for i in range(len(flat)):
        for j in range(len(flat)):
            if i != j:
                de_num += delta_fn(flat[i], flat[j])
    de = de_num / (n * (n - 1))
    if de == 0:
        return None
    return 1.0 - do / de


class TestBruteForceCrossCheck(unittest.TestCase):
    def _run_random_trials(self, metric, delta_fn, value_pool, n_trials=200):
        rng = random.Random(20260923 + hash(metric) % 1000)
        checked = 0
        for _ in range(n_trials):
            n_units = rng.randint(3, 15)
            n_raters = rng.randint(2, 4)
            units = []
            for _ in range(n_units):
                unit = []
                for _ in range(n_raters):
                    if rng.random() < 0.15:
                        unit.append(None)
                    else:
                        unit.append(rng.choice(value_pool))
                units.append(unit)

            expected = _brute_force_alpha(units, delta_fn)
            result = krippendorff_alpha(units, metric=metric)

            if expected is None:
                self.assertIsNone(result.alpha, f"trial units={units}")
            else:
                self.assertIsNotNone(result.alpha, f"trial units={units}")
                self.assertAlmostEqual(result.alpha, expected, places=10, msg=f"units={units}")
                checked += 1
        self.assertGreater(checked, n_trials // 4, "too many trials were degenerate to be a meaningful check")

    def test_nominal_matches_brute_force_across_random_inputs(self):
        self._run_random_trials("nominal", _nominal_delta, ["A", "B", "C"])

    def test_interval_matches_brute_force_across_random_inputs(self):
        self._run_random_trials("interval", _interval_delta, [1, 2, 3, 4, 5])

    def test_ratio_matches_brute_force_across_random_inputs(self):
        self._run_random_trials("ratio", _ratio_delta, [1, 2, 3, 4, 5, 6])


# ---------------------------------------------------------------------------
# (b) hand-derived worked examples and analytic properties
# ---------------------------------------------------------------------------

class TestHandDerivedExamples(unittest.TestCase):
    def test_two_rater_nominal_example_matches_hand_calculation(self):
        # Rater1: A A B B A / Rater2: A B B B A -> 4/5 units agree.
        # By hand: O[A,A]=4, O[B,B]=4, O[A,B]=O[B,A]=1, n=10, marginals=[5,5].
        # sum_o_delta=2, sum_marginal_delta=50, D_e=50/90.
        # alpha = 1 - 9*2/50 = 1 - 18/50 = 0.64
        units = [["A", "A"], ["A", "B"], ["B", "B"], ["B", "B"], ["A", "A"]]
        result = krippendorff_alpha(units, metric="nominal", categories=["A", "B"])
        self.assertAlmostEqual(result.alpha, 0.64, places=10)
        self.assertAlmostEqual(result.observed_disagreement, 0.2, places=10)
        self.assertAlmostEqual(result.expected_disagreement, 50 / 90, places=10)

    def test_two_rater_systematic_swap_matches_hand_calculation(self):
        # Rater1: A B A / Rater2: B A B (perfectly anti-correlated, 2 cats).
        # By hand: n=6, O[A,B]=O[B,A]=3, marginals=[3,3], sum_o_delta=6,
        # sum_marginal_delta=18, alpha = 1 - 5*6/18 = 1 - 30/18 = -2/3.
        units = [["A", "B"], ["B", "A"], ["A", "B"]]
        result = krippendorff_alpha(units, metric="nominal", categories=["A", "B"])
        self.assertAlmostEqual(result.alpha, -2.0 / 3.0, places=10)

    def test_perfect_agreement_gives_exactly_one(self):
        rng = random.Random(1)
        units = [[v, v, v] for v in (rng.choice(["A", "B", "C"]) for _ in range(30))]
        result = krippendorff_alpha(units, metric="nominal")
        self.assertAlmostEqual(result.alpha, 1.0, places=10)

    def test_large_n_independent_random_labels_alpha_near_zero(self):
        rng = random.Random(42)
        categories = ["A", "B", "C", "D"]
        units = [[rng.choice(categories), rng.choice(categories)] for _ in range(20000)]
        result = krippendorff_alpha(units, metric="nominal", categories=categories)
        self.assertLess(abs(result.alpha), 0.05)

    def test_interval_metric_treats_off_by_one_as_less_severe_than_nominal(self):
        # Same systematic constant-offset dataset scored two ways: nominal
        # counts "off by one" as maximally wrong, interval counts it as a
        # small squared distance, so interval alpha must be strictly higher.
        rng = random.Random(7)
        base = [rng.randint(1, 5) for _ in range(50)]
        units = [[v, v + 1] for v in base]
        categories = list(range(1, 7))
        nominal_result = krippendorff_alpha(units, metric="nominal", categories=categories)
        interval_result = krippendorff_alpha(units, metric="interval", categories=categories)
        self.assertLess(nominal_result.alpha, interval_result.alpha)


# ---------------------------------------------------------------------------
# (c) tight cross-check of QWK against sklearn's same statistic
# ---------------------------------------------------------------------------

class TestQuadraticWeightedKappaAgainstSklearn(unittest.TestCase):
    def test_matches_sklearn_cohen_kappa_quadratic_weights(self):
        from sklearn.metrics import cohen_kappa_score

        categories = [1, 2, 3, 4, 5]
        rng = random.Random(99)
        for _ in range(25):
            n = rng.randint(10, 60)
            a = [rng.choice(categories) for _ in range(n)]
            b = [rng.choice(categories) for _ in range(n)]
            ours = quadratic_weighted_kappa(a, b, categories)
            theirs = cohen_kappa_score(a, b, labels=categories, weights="quadratic")
            self.assertAlmostEqual(ours, theirs, places=9)

    def test_cohens_kappa_is_a_different_number_from_alpha_same_chance_model_caveat(self):
        # Documented, not asserted equal: alpha and Cohen's kappa use
        # different chance-agreement models (pooled marginals vs. per-rater
        # marginal product), so they are expected to diverge except in
        # symmetric-marginal special cases. This test only pins down that
        # cohens_kappa runs and returns a plausible value, as a loose
        # cross-check that our confusion-matrix plumbing is sane.
        units = [["A", "A"], ["A", "B"], ["B", "B"], ["B", "B"], ["A", "A"]]
        a = [u[0] for u in units]
        b = [u[1] for u in units]
        kappa = cohens_kappa(a, b, ["A", "B"])
        self.assertIsNotNone(kappa)
        self.assertGreater(kappa, 0.0)
        self.assertLessEqual(kappa, 1.0)


# ---------------------------------------------------------------------------
# (d) degenerate cases
# ---------------------------------------------------------------------------

class TestDegenerateCases(unittest.TestCase):
    def test_empty_units_is_insufficient_data(self):
        result = krippendorff_alpha([], metric="nominal")
        self.assertIsNone(result.alpha)
        self.assertEqual(result.undefined_reason, "insufficient_data")

    def test_all_units_unpairable_is_insufficient_data(self):
        result = krippendorff_alpha([["A"], ["B"], [None, None]], metric="nominal")
        self.assertIsNone(result.alpha)
        self.assertEqual(result.undefined_reason, "insufficient_data")

    def test_all_same_category_is_no_variance_not_zero_or_one(self):
        units = [["A", "A"], ["A", "A"], ["A", "A"]]
        result = krippendorff_alpha(units, metric="nominal")
        self.assertIsNone(result.alpha)
        self.assertEqual(result.undefined_reason, "no_variance")

    def test_alpha_is_never_clamped_negative(self):
        units = [["A", "B"], ["B", "A"], ["A", "B"]]
        result = krippendorff_alpha(units, metric="nominal", categories=["A", "B"])
        self.assertLess(result.alpha, 0.0)

    def test_ordinal_bootstrap_recomputes_delta_per_replicate_not_frozen(self):
        # Skewed fixture: category "C" is rare, so resampling shifts the
        # marginals a lot from replicate to replicate. If the ordinal delta
        # matrix were wrongly frozen at the full-sample marginals (a classic
        # implementation bug this module's docstring calls out explicitly),
        # every replicate would reuse the same delta matrix and the resulting
        # CI would be a tight, mechanically-narrow band around the point
        # estimate purely due to loss of that source of variability.
        rng = random.Random(3)
        categories = ["low", "mid", "high"]
        units = []
        for _ in range(60):
            base = rng.choices(categories, weights=[0.6, 0.35, 0.05])[0]
            other = base if rng.random() < 0.7 else rng.choice(categories)
            units.append([base, other])

        result = krippendorff_alpha(
            units, metric="ordinal", categories=categories, n_bootstrap=500, seed=1,
        )
        self.assertIsNotNone(result.alpha)
        self.assertIsNotNone(result.ci_low)
        self.assertIsNotNone(result.ci_high)

        # Build the wrongly-frozen alternative directly: same resampling,
        # same seed, but delta_sq fixed at the full-sample ordinal matrix.
        k_index = {c: i for i, c in enumerate(categories)}
        per_unit = [unit_coincidence(u, k_index) for u in units]
        full_marginals = sum(per_unit).sum(axis=1)
        frozen_delta = delta_squared_matrix("ordinal", categories, marginals=full_marginals)

        n_units = len(per_unit)
        stack = np.stack(per_unit, axis=0).reshape(n_units, len(categories) ** 2)
        rng_np = np.random.default_rng(1)
        counts = rng_np.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=500)
        Os = (counts @ stack).reshape(500, len(categories), len(categories))

        frozen_alphas = []
        for b in range(500):
            O = Os[b]
            n = float(O.sum())
            if n <= 1:
                continue
            sum_o_delta = float(np.sum(O * frozen_delta))
            marg = O.sum(axis=1)
            sum_marginal_delta = float(np.sum(np.outer(marg, marg) * frozen_delta))
            if sum_marginal_delta == 0:
                continue
            frozen_alphas.append(1.0 - (n - 1) * sum_o_delta / sum_marginal_delta)

        frozen_lo, frozen_hi = np.percentile(frozen_alphas, [2.5, 97.5])
        correct_width = result.ci_high - result.ci_low
        frozen_width = frozen_hi - frozen_lo
        # The two CIs must not be identical -- if they were, the production
        # bootstrap would be silently reusing the frozen matrix instead of
        # recomputing it per replicate.
        self.assertNotAlmostEqual(correct_width, frozen_width, places=6)


# ---------------------------------------------------------------------------
# (e) MASI / Jaccard identities, computed by hand
# ---------------------------------------------------------------------------

class TestMasiJaccardIdentities(unittest.TestCase):
    def test_identical_sets_masi_is_zero(self):
        a = frozenset({"A", "B"})
        self.assertEqual(masi_distance(a, a), 0.0)
        self.assertEqual(jaccard(a, a), 0.0)

    def test_disjoint_sets_masi_is_one(self):
        a, b = frozenset({"A"}), frozenset({"B"})
        self.assertEqual(masi_distance(a, b), 1.0)

    def test_proper_subset_matches_hand_derivation(self):
        # a={A,B}, b={A}: |intersection|=1, |union|=2, similarity=0.5, m=2/3.
        # masi = 1 - similarity*m = 1 - 0.5*(2/3) = 2/3.
        a, b = frozenset({"A", "B"}), frozenset({"A"})
        self.assertAlmostEqual(masi_distance(a, b), 2.0 / 3.0, places=10)

    def test_both_empty_is_zero_not_nan(self):
        a = b = frozenset()
        self.assertEqual(masi_distance(a, b), 0.0)
        self.assertFalse(np.isnan(masi_distance(a, b)))

    def test_overlap_without_subset_uses_one_third_monotonicity(self):
        # a={A,B}, b={B,C}: intersection={B} (1), union={A,B,C} (3),
        # similarity=1/3, m=1/3 (overlap, no subset). masi = 1 - (1/3)*(1/3).
        a, b = frozenset({"A", "B"}), frozenset({"B", "C"})
        self.assertAlmostEqual(masi_distance(a, b), 1.0 - (1.0 / 3.0) * (1.0 / 3.0), places=10)


# ---------------------------------------------------------------------------
# Companion metric sanity checks
# ---------------------------------------------------------------------------

class TestCompanionMetrics(unittest.TestCase):
    def test_percent_agreement_ignores_missing_pairs(self):
        a = ["A", "B", None, "A"]
        b = ["A", "A", "B", None]
        self.assertAlmostEqual(percent_agreement(a, b), 0.5, places=10)

    def test_percent_agreement_all_missing_is_nan(self):
        self.assertTrue(np.isnan(percent_agreement([None, None], [None, None])))

    def test_confusion_matrix_counts_pairs_and_skips_missing(self):
        # a[2] is None so that pair is skipped; remaining pairs are
        # (A,A) -> mat[A,A] and (B,A) -> mat[B,A].
        mat, cats = confusion_matrix(["A", "B", None], ["A", "A", "B"], ["A", "B"])
        self.assertEqual(mat.tolist(), [[1, 0], [1, 0]])

    def test_set_micro_prf_pools_counts_across_units(self):
        pairs = [
            (frozenset({"A", "B"}), frozenset({"A"})),
            (frozenset({"C"}), frozenset({"C", "D"})),
        ]
        result = set_micro_prf(pairs)
        # tp = |{A}| + |{C}| = 2; fp = |{}| + |{D}| = 1; fn = |{B}| + |{}| = 1
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertAlmostEqual(result["precision"], 2 / 3, places=10)
        self.assertAlmostEqual(result["recall"], 2 / 3, places=10)


# ---------------------------------------------------------------------------
# (f) regression tests for the reviewer-flagged fixes in this window
# ---------------------------------------------------------------------------

class TestDistanceIsUsedAsIsNotSquared(unittest.TestCase):
    def test_custom_distance_is_not_squared(self):
        # Passonneau (2006)/nltk convention: a distance callable plugs
        # directly into the delta slot. A constant 0.5 off-diagonal distance
        # must appear as 0.5 in the matrix, not 0.25 (its square).
        mat = delta_squared_matrix(
            "nominal", ["A", "B"], distance=lambda a, b: 0.0 if a == b else 0.5,
        )
        self.assertEqual(mat.tolist(), [[0.0, 0.5], [0.5, 0.0]])

    def test_distance_with_nonzero_diagonal_raises(self):
        with self.assertRaises(ValueError):
            delta_squared_matrix("nominal", ["A", "B"], distance=lambda a, b: 1.0)


class TestOrdinalRequiresExplicitCategories(unittest.TestCase):
    def test_ordinal_without_categories_raises(self):
        with self.assertRaises(ValueError):
            krippendorff_alpha([["low", "high"], ["mid", "mid"]], metric="ordinal")


class TestNaNIsRejected(unittest.TestCase):
    def test_nan_value_raises(self):
        with self.assertRaises(ValueError):
            krippendorff_alpha([["A", float("nan")], ["A", "A"]], metric="nominal")


class TestDuplicateCategoriesRejected(unittest.TestCase):
    def test_duplicate_categories_raises(self):
        with self.assertRaises(ValueError):
            krippendorff_alpha([["A", "B"], ["B", "A"]], metric="nominal", categories=["A", "A", "B"])


class TestRatioRejectsNegativeValues(unittest.TestCase):
    def test_negative_category_raises(self):
        with self.assertRaises(ValueError):
            krippendorff_alpha([[-1, 2], [2, -1]], metric="ratio", categories=[-1, 2])


class TestDeltaMatrixThreadedToBootstrap(unittest.TestCase):
    def test_bootstrap_ci_uses_the_same_delta_matrix_as_the_point_estimate(self):
        rng = random.Random(11)
        categories = ["A", "B", "C"]
        units = [[rng.choice(categories), rng.choice(categories)] for _ in range(40)]
        # A deliberately unusual delta (not 0/1 nominal) so a bootstrap that
        # silently fell back to the default nominal delta would diverge from
        # a direct call to bootstrap_unit_ci with this same override.
        delta_matrix = np.array([[0.0, 3.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 0.0]])

        result = krippendorff_alpha(
            units, metric="nominal", categories=categories,
            delta_matrix=delta_matrix, n_bootstrap=200, seed=5,
        )

        k_index = {c: i for i, c in enumerate(categories)}
        per_unit = [unit_coincidence(u, k_index) for u in units]
        expected_lo, expected_hi, _ = bootstrap_unit_ci(
            per_unit, "nominal", categories, delta_matrix=delta_matrix,
            n_bootstrap=200, seed=5,
        )

        self.assertAlmostEqual(result.ci_low, expected_lo, places=12)
        self.assertAlmostEqual(result.ci_high, expected_hi, places=12)

    def test_bootstrap_ci_with_override_differs_from_default_nominal_delta(self):
        # Sanity check that the override above actually changes something --
        # otherwise the previous test could pass vacuously. Note: a delta
        # matrix that is a uniform scalar multiple of the nominal 0/1 matrix
        # would cancel out of alpha's ratio and look identical, so this uses
        # non-uniform off-diagonal weights instead.
        rng = random.Random(11)
        categories = ["A", "B", "C"]
        units = [[rng.choice(categories), rng.choice(categories)] for _ in range(40)]
        delta_matrix = np.array([[0.0, 1.0, 9.0], [1.0, 0.0, 1.0], [9.0, 1.0, 0.0]])

        with_override = krippendorff_alpha(
            units, metric="nominal", categories=categories,
            delta_matrix=delta_matrix, n_bootstrap=200, seed=5,
        )
        default = krippendorff_alpha(
            units, metric="nominal", categories=categories, n_bootstrap=200, seed=5,
        )
        self.assertNotAlmostEqual(with_override.ci_low, default.ci_low, places=6)


class TestSetMicroPrfZeroButDefinedF1(unittest.TestCase):
    def test_zero_precision_and_recall_gives_f1_zero_not_nan(self):
        # ref={A}, pred={B}: tp=0, fp=1, fn=1 -> precision=0.0, recall=0.0,
        # both individually defined (denominators are nonzero), so f1 must
        # be the real number 0.0, not nan.
        pairs = [(frozenset({"A"}), frozenset({"B"}))]
        result = set_micro_prf(pairs)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)
        self.assertFalse(np.isnan(result["f1"]))

    def test_undefined_precision_still_gives_nan_f1(self):
        # No predictions and no references at all -> tp+fp==0, precision
        # undefined -> f1 must stay nan, not silently become 0.0.
        pairs = [(frozenset(), frozenset())]
        result = set_micro_prf(pairs)
        self.assertTrue(np.isnan(result["precision"]))
        self.assertTrue(np.isnan(result["f1"]))


class TestLengthMismatchGuards(unittest.TestCase):
    def test_percent_agreement_raises_on_length_mismatch(self):
        with self.assertRaises(ValueError):
            percent_agreement(["A", "B"], ["A"])

    def test_confusion_matrix_raises_on_length_mismatch(self):
        with self.assertRaises(ValueError):
            confusion_matrix(["A", "B"], ["A"], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
