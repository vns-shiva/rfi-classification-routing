"""Unit tests for the corpus stratification/planning module. Run with:

    python -m unittest discover -s code/tests -t code

Exercises stratification.py against the two hand-authored seed templates
(t01 dev, t09 eval) plus small synthetic templates (built directly via the
Template dataclass, bypassing validate_template, since these tests only need
stratification.py's own logic -- not full JSON-authoring validity).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stratification import (
    AXIS_NAMES,
    AXIS_VOCAB,
    AXIS_WEIGHTS,
    DEFAULT_SEED,
    DEV_POOL,
    DIVERGENCE_QUOTAS,
    EVAL_POOL,
    FEATURE_NAMES,
    GoldRoutingRow,
    QUOTAS,
    REVIEWER_QUOTAS,
    REVIEWER_ROLES,
    SPLIT_SIZES,
    Template,
    Cell,
    CorpusPlan,
    CoverageError,
    assert_coverage,
    build_plan,
    cell_features,
    cell_incoherence_reasons,
    cell_weight,
    cells_by_split,
    cells_for_template,
    coverage_report,
    coverage_shortfalls,
    expressible_pairs,
    format_coverage_markdown,
    is_coherent,
    is_expressible,
    load_templates,
    plan_from_dict,
    plan_to_dict,
    read_plan,
    select_overlap,
    unreachable_quotas,
    validate_quotas,
    weighted_choice,
    write_plan,
)

T01 = "t01_design_clarification_dev"
T09 = "t09_design_clarification_eval"


def _base_cell_kwargs(**overrides) -> dict:
    base = dict(
        split=DEV_POOL,
        rfi_type="design_clarification",
        primary_discipline="Architectural",
        urgency="routine",
        cost_impact=False,
        schedule_impact=False,
        deadline_family="none",
        adversarial="none",
        thread_shape="single_message",
    )
    base.update(overrides)
    return base


def _cell(**overrides) -> Cell:
    return Cell(**_base_cell_kwargs(**overrides))


def _make_template(template_id: str, pool: str, axes: dict, gold_routing_table: tuple) -> Template:
    return Template(
        template_id=template_id,
        pool=pool,
        axes=axes,
        csi_by_discipline={},
        secondary_by_adversarial={},
        slots={},
        urgency_phrasing={},
        cost_phrasing={},
        schedule_phrasing={},
        deadline_phrasing={},
        decoy_phrasing=(),
        boundary_phrasing=(),
        ambiguity_phrasing=(),
        unresolvable_deadline_phrasing=(),
        messages=(),
        gold={},
        gold_routing_table=gold_routing_table,
    )


def _full_coverage_template(template_id: str, pool: str) -> Template:
    axes = {axis: AXIS_VOCAB[axis] for axis in AXIS_NAMES}
    # One row per remaining REVIEWER_ROLES value (index 0 is the catch-all
    # below, index 1 is the divergent row above) so that REVIEWER_QUOTAS
    # reachability checks see every role as expressible on this "full
    # coverage" synthetic template, not just the two roles the original
    # two-row table happened to name. Five of the six rfi_type values are
    # each claimed by one role; the sixth ("reserved_type") is split by
    # urgency tier among three more roles, deliberately leaving one
    # (reserved_type, "critical") combination unclaimed so the catch-all
    # row -- and REVIEWER_ROLES[0], which it assigns -- stays reachable.
    rfi_types = AXIS_VOCAB["rfi_type"]
    claimed_types, reserved_type = rfi_types[:-1], rfi_types[-1]
    role_rows = tuple(
        GoldRoutingRow(
            when={"rfi_type": rfi_type},
            assigned_reviewer=role,
            routing_rationale="reviewer-coverage test row",
            escalation=False,
        )
        for rfi_type, role in zip(claimed_types, REVIEWER_ROLES[2:2 + len(claimed_types)])
    )
    leftover_roles = REVIEWER_ROLES[2 + len(claimed_types):]
    urgency_tiers = [t for t in AXIS_VOCAB["urgency"] if t != "critical"]
    role_rows += tuple(
        GoldRoutingRow(
            when={"rfi_type": reserved_type, "urgency": urgency},
            assigned_reviewer=role,
            routing_rationale="reviewer-coverage test row",
            escalation=(urgency == "urgent"),
        )
        for urgency, role in zip(urgency_tiers, leftover_roles)
    )
    gold_routing_table = (
        GoldRoutingRow(
            when={"cost_impact": True, "schedule_impact": True},
            assigned_reviewer=REVIEWER_ROLES[1],
            routing_rationale="divergent test row",
            escalation=True,
            policy_divergence=True,
            divergence_reason="test",
        ),
        *role_rows,
        GoldRoutingRow(
            when={},
            assigned_reviewer=REVIEWER_ROLES[0],
            routing_rationale="catch-all test row",
            escalation=False,
        ),
    )
    return _make_template(template_id, pool, axes, gold_routing_table)


class TestCellValidation(unittest.TestCase):
    def test_valid_cell_constructs(self):
        cell = _cell()
        self.assertEqual(cell.split, DEV_POOL)

    def test_invalid_split_raises(self):
        with self.assertRaises(ValueError):
            _cell(split="bogus")

    def test_invalid_axis_value_raises(self):
        with self.assertRaises(ValueError):
            _cell(rfi_type="not_a_real_type")

    def test_non_bool_cost_impact_raises(self):
        with self.assertRaises(ValueError):
            _cell(cost_impact=1)

    def test_non_bool_schedule_impact_raises(self):
        with self.assertRaises(ValueError):
            _cell(schedule_impact=0)

    def test_to_dict_from_dict_round_trip(self):
        cell = _cell(urgency="priority", cost_impact=True)
        restored = Cell.from_dict(cell.to_dict())
        self.assertEqual(cell, restored)


class TestCoherence(unittest.TestCase):
    def test_baseline_cell_is_coherent(self):
        cell = _cell()
        self.assertTrue(is_coherent(cell))
        self.assertEqual(cell_incoherence_reasons(cell), [])

    def test_c1_unresolvable_deadline_requires_unresolvable_family(self):
        bad = _cell(adversarial="unresolvable_deadline", deadline_family="weekday")
        self.assertFalse(is_coherent(bad))
        good = _cell(adversarial="unresolvable_deadline", deadline_family="ambiguous")
        self.assertTrue(is_coherent(good))

    def test_c2_distractor_cost_language_requires_cost_impact_false(self):
        bad = _cell(adversarial="distractor_cost_language", cost_impact=True)
        self.assertFalse(is_coherent(bad))
        good = _cell(adversarial="distractor_cost_language", cost_impact=False)
        self.assertTrue(is_coherent(good))

    def test_c3_critical_urgency_requires_cost_or_schedule_impact(self):
        bad = _cell(urgency="critical", cost_impact=False, schedule_impact=False)
        self.assertFalse(is_coherent(bad))
        good = _cell(urgency="critical", cost_impact=True, schedule_impact=False)
        self.assertTrue(is_coherent(good))

    def test_c4_boundary_urgency_requires_priority_or_urgent(self):
        bad = _cell(adversarial="boundary_urgency", urgency="routine")
        self.assertFalse(is_coherent(bad))
        good = _cell(adversarial="boundary_urgency", urgency="priority")
        self.assertTrue(is_coherent(good))


class TestExpressibility(unittest.TestCase):
    def setUp(self):
        self.templates = load_templates()

    def test_cells_for_template_t01_has_32_coherent_cells(self):
        cells = cells_for_template(self.templates[T01])
        self.assertEqual(len(cells), 32)
        self.assertTrue(all(c.split == DEV_POOL for c in cells))

    def test_cells_for_template_t09_has_64_coherent_cells(self):
        cells = cells_for_template(self.templates[T09])
        self.assertEqual(len(cells), 64)
        self.assertTrue(all(c.split == EVAL_POOL for c in cells))

    def test_is_expressible_true_for_t01_baseline_cell(self):
        cell = _cell(deadline_family="none")
        self.assertTrue(is_expressible(self.templates[T01], cell))

    def test_is_expressible_false_for_wrong_pool(self):
        cell = _cell(split=EVAL_POOL)
        self.assertFalse(is_expressible(self.templates[T01], cell))

    def test_is_expressible_false_for_incoherent_cell_even_if_axis_compatible(self):
        synthetic = _full_coverage_template("synthetic_dev", DEV_POOL)
        incoherent = Cell(
            split=DEV_POOL, rfi_type="design_clarification", primary_discipline="Architectural",
            urgency="routine", cost_impact=True, schedule_impact=False,
            deadline_family="none", adversarial="distractor_cost_language", thread_shape="single_message",
        )
        self.assertFalse(is_expressible(synthetic, incoherent))

    def test_expressible_pairs_deduplicated_and_sorted(self):
        pairs = expressible_pairs(self.templates)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(pairs, sorted(pairs, key=lambda p: (p[0], tuple(getattr(p[1], a) for a in AXIS_NAMES))))
        # The dev pool has 9 templates (t01-t08, t10) authored to close every
        # dev quota; the eval pool has 12 templates (t09, t11-t21) authored
        # the same way. These totals move whenever a template is added or
        # its axes change.
        self.assertEqual(len(pairs), 216 + 316)

    def test_cells_by_split_partitions_expressible_pairs(self):
        dev_pairs = cells_by_split(self.templates, DEV_POOL)
        eval_pairs = cells_by_split(self.templates, EVAL_POOL)
        self.assertEqual(len(dev_pairs), 216)
        self.assertEqual(len(eval_pairs), 316)
        self.assertTrue(all(c.split == DEV_POOL for _, c in dev_pairs))
        self.assertTrue(all(c.split == EVAL_POOL for _, c in eval_pairs))

    def test_cell_features_matches_axis_values(self):
        cell = _cell(urgency="priority")
        features = cell_features(cell)
        self.assertIn(("urgency", "priority"), features)
        self.assertEqual(len(features), len(AXIS_NAMES))

    def test_feature_names_is_axis_names(self):
        self.assertEqual(FEATURE_NAMES, AXIS_NAMES)


class TestWeightedChoice(unittest.TestCase):
    def test_empty_values_raises(self):
        with self.assertRaises(ValueError):
            weighted_choice(__import__("random").Random(1), [], {})

    def test_missing_weight_key_raises_keyerror(self):
        with self.assertRaises(KeyError):
            weighted_choice(__import__("random").Random(1), ["a"], {})

    def test_negative_weight_raises(self):
        with self.assertRaises(ValueError):
            weighted_choice(__import__("random").Random(1), ["a"], {"a": -1.0})

    def test_non_positive_total_raises(self):
        with self.assertRaises(ValueError):
            weighted_choice(__import__("random").Random(1), ["a", "b"], {"a": 0.0, "b": 0.0})

    def test_zero_weight_value_never_returned(self):
        import random
        rng = random.Random(7)
        results = {weighted_choice(rng, ["a", "b"], {"a": 0.0, "b": 1.0}) for _ in range(200)}
        self.assertEqual(results, {"b"})

    def test_deterministic_for_given_seed(self):
        import random
        rng1 = random.Random(42)
        draws1 = [weighted_choice(rng1, ["a", "b", "c"], {"a": 0.2, "b": 0.3, "c": 0.5}) for _ in range(30)]
        rng2 = random.Random(42)
        draws2 = [weighted_choice(rng2, ["a", "b", "c"], {"a": 0.2, "b": 0.3, "c": 0.5}) for _ in range(30)]
        self.assertEqual(draws1, draws2)

    def test_respects_relative_weights_over_many_draws(self):
        import random
        rng = random.Random(123)
        counts = {"a": 0, "b": 0}
        for _ in range(2000):
            counts[weighted_choice(rng, ["a", "b"], {"a": 0.9, "b": 0.1})] += 1
        self.assertGreater(counts["a"], counts["b"])


class TestAxisWeights(unittest.TestCase):
    def test_every_axis_present(self):
        self.assertEqual(set(AXIS_WEIGHTS.keys()), set(AXIS_NAMES))

    def test_each_axis_covers_every_vocab_value_and_sums_to_one(self):
        for axis in AXIS_NAMES:
            with self.subTest(axis=axis):
                self.assertEqual(set(AXIS_WEIGHTS[axis].keys()), set(AXIS_VOCAB[axis]))
                self.assertAlmostEqual(sum(AXIS_WEIGHTS[axis].values()), 1.0, places=6)

    def test_cell_weight_is_product_of_axis_weights(self):
        cell = _cell(urgency="priority", cost_impact=True)
        expected = 1.0
        for axis in AXIS_NAMES:
            expected *= AXIS_WEIGHTS[axis][getattr(cell, axis)]
        self.assertAlmostEqual(cell_weight(cell), expected, places=12)


class TestQuotas(unittest.TestCase):
    def test_real_quotas_are_valid(self):
        self.assertEqual(validate_quotas(), [])

    def test_unknown_split_flagged(self):
        # Sanity check validate_quotas actually inspects content, using a
        # deliberately-broken copy rather than mutating the real QUOTAS.
        import stratification as strat
        bad_quotas = {"not_a_split": {"rfi_type": {"design_clarification": 1}}}
        original = strat.QUOTAS
        try:
            strat.QUOTAS = bad_quotas
            errors = validate_quotas()
        finally:
            strat.QUOTAS = original
        self.assertTrue(any("unknown split" in e for e in errors))


class TestUnreachableQuotas(unittest.TestCase):
    def test_real_dev_pool_has_no_unreachable_quotas(self):
        # Both the dev pool's 9 templates (t01-t08, t10) and the eval pool's
        # 12 templates (t09, t11-t21) were authored to close every quota for
        # their split, so the real corpus should have no unreachable quotas
        # at all.
        templates = load_templates()
        problems = unreachable_quotas(templates)
        self.assertEqual(problems, [])

    def test_full_coverage_templates_have_no_unreachable_quotas(self):
        templates = {
            "synthetic_dev": _full_coverage_template("synthetic_dev", DEV_POOL),
            "synthetic_eval": _full_coverage_template("synthetic_eval", EVAL_POOL),
        }
        self.assertEqual(unreachable_quotas(templates), [])


def _two_role_template(template_id: str, pool: str) -> Template:
    """Full axis coverage, but only REVIEWER_ROLES[0]/[1] ever get assigned --
    the shape the original _full_coverage_template had before REVIEWER_QUOTAS
    existed. Used to exercise the reviewer-quota-deficient path deliberately."""
    axes = {axis: AXIS_VOCAB[axis] for axis in AXIS_NAMES}
    gold_routing_table = (
        GoldRoutingRow(
            when={"cost_impact": True, "schedule_impact": True},
            assigned_reviewer=REVIEWER_ROLES[1],
            routing_rationale="divergent test row",
            escalation=True,
            policy_divergence=True,
            divergence_reason="test",
        ),
        GoldRoutingRow(
            when={},
            assigned_reviewer=REVIEWER_ROLES[0],
            routing_rationale="catch-all test row",
            escalation=False,
        ),
    )
    return _make_template(template_id, pool, axes, gold_routing_table)


class TestReviewerQuotas(unittest.TestCase):
    def test_real_reviewer_quotas_are_valid(self):
        self.assertEqual(validate_quotas(), [])

    def test_unknown_split_flagged(self):
        import stratification as strat
        original = strat.REVIEWER_QUOTAS
        try:
            strat.REVIEWER_QUOTAS = {"not_a_split": 1}
            errors = validate_quotas()
        finally:
            strat.REVIEWER_QUOTAS = original
        self.assertTrue(any("REVIEWER_QUOTAS has unknown split" in e for e in errors))

    def test_negative_value_flagged(self):
        import stratification as strat
        original = strat.REVIEWER_QUOTAS
        try:
            strat.REVIEWER_QUOTAS = {DEV_POOL: -1, EVAL_POOL: 5}
            errors = validate_quotas()
        finally:
            strat.REVIEWER_QUOTAS = original
        self.assertTrue(any("is negative" in e for e in errors))

    def test_total_exceeding_split_size_flagged(self):
        import stratification as strat
        original = strat.REVIEWER_QUOTAS
        try:
            strat.REVIEWER_QUOTAS = {DEV_POOL: SPLIT_SIZES[DEV_POOL], EVAL_POOL: 5}
            errors = validate_quotas()
        finally:
            strat.REVIEWER_QUOTAS = original
        self.assertTrue(any("exceeding split size" in e for e in errors))

    def test_unreachable_role_flagged_when_only_two_roles_assigned(self):
        templates = {
            "synthetic_dev": _two_role_template("synthetic_dev", DEV_POOL),
            "synthetic_eval": _two_role_template("synthetic_eval", EVAL_POOL),
        }
        problems = unreachable_quotas(templates)
        for role in REVIEWER_ROLES[2:]:
            with self.subTest(role=role):
                self.assertTrue(any(repr(role) in p for p in problems))
        for role in REVIEWER_ROLES[:2]:
            with self.subTest(role=role):
                self.assertFalse(any(repr(role) in p for p in problems))

    def test_build_plan_meets_reviewer_quota_when_not_contended_by_other_quotas(self):
        # QUOTAS/DIVERGENCE_QUOTAS are each closed by their own greedy Phase-A
        # loop *before* REVIEWER_QUOTAS runs, and every loop shares the same
        # split_size ceiling. With the real QUOTAS values (e.g. eval's urgency
        # minimums alone sum to 100) a single synthetic full-coverage template
        # can have its whole split filled by axis/divergence quotas before the
        # reviewer loop gets a turn -- that's the same honest-shortfall
        # behavior test_lenient_mode_records_real_shortfalls already treats as
        # expected for an incomplete template set, not a bug in the reviewer
        # loop. So isolate the reviewer loop here by neutralizing the other
        # two quotas, to confirm *it* correctly closes every reachable role
        # given room; the contended case is covered separately below.
        import stratification as strat
        original_quotas, original_divergence = strat.QUOTAS, strat.DIVERGENCE_QUOTAS
        try:
            strat.QUOTAS = {DEV_POOL: {}, EVAL_POOL: {}}
            strat.DIVERGENCE_QUOTAS = {DEV_POOL: 0, EVAL_POOL: 0}
            templates = {
                "synthetic_dev": _full_coverage_template("synthetic_dev", DEV_POOL),
                "synthetic_eval": _full_coverage_template("synthetic_eval", EVAL_POOL),
            }
            plan = build_plan(templates, seed=DEFAULT_SEED, strict=False)
        finally:
            strat.QUOTAS, strat.DIVERGENCE_QUOTAS = original_quotas, original_divergence
        report = coverage_report(plan, templates)
        for split in (DEV_POOL, EVAL_POOL):
            reviewer_counts = report["splits"][split]["reviewer_counts"]
            for role in REVIEWER_ROLES:
                with self.subTest(split=split, role=role):
                    self.assertGreaterEqual(reviewer_counts[role], REVIEWER_QUOTAS[split])

    def test_build_plan_records_reviewer_shortfall_when_quota_contended_by_other_quotas(self):
        # Same full-coverage template, but under the *real* QUOTAS/
        # DIVERGENCE_QUOTAS -- confirms that when those earlier Phase-A loops
        # fill the split first, the reviewer loop doesn't fail silently: it
        # records an honest shortfall instead of just leaving counts at 0.
        templates = {
            "synthetic_dev": _full_coverage_template("synthetic_dev", DEV_POOL),
            "synthetic_eval": _full_coverage_template("synthetic_eval", EVAL_POOL),
        }
        plan = build_plan(templates, seed=DEFAULT_SEED, strict=False)
        self.assertTrue(any("reviewer quota" in s and "only reached" in s for s in plan.shortfalls))

    def test_build_plan_records_reviewer_shortfall_when_role_unreachable(self):
        templates = {
            "synthetic_dev": _two_role_template("synthetic_dev", DEV_POOL),
            "synthetic_eval": _two_role_template("synthetic_eval", EVAL_POOL),
        }
        plan = build_plan(templates, seed=DEFAULT_SEED, strict=False)
        self.assertTrue(any("reviewer quota" in s and "unreachable" in s for s in plan.shortfalls))

    def test_coverage_shortfalls_reports_reviewer_gap(self):
        templates = {
            "synthetic_dev": _two_role_template("synthetic_dev", DEV_POOL),
            "synthetic_eval": _two_role_template("synthetic_eval", EVAL_POOL),
        }
        plan = build_plan(templates, seed=DEFAULT_SEED, strict=False)
        report = coverage_report(plan, templates)
        shortfalls = coverage_shortfalls(report)
        self.assertTrue(any("reviewer" in s and repr(REVIEWER_ROLES[2]) in s for s in shortfalls))


class TestBuildPlan(unittest.TestCase):
    def setUp(self):
        self.templates = load_templates()

    def test_produces_exact_split_sizes(self):
        plan = build_plan(self.templates, seed=DEFAULT_SEED)
        self.assertEqual(len(plan.by_split(DEV_POOL)), SPLIT_SIZES[DEV_POOL])
        self.assertEqual(len(plan.by_split(EVAL_POOL)), SPLIT_SIZES[EVAL_POOL])
        self.assertEqual(len(plan.entries), sum(SPLIT_SIZES.values()))

    def test_thread_ids_unique_and_sequential_per_split(self):
        plan = build_plan(self.templates, seed=DEFAULT_SEED)
        for split in (DEV_POOL, EVAL_POOL):
            ids = [e.thread_id for e in plan.by_split(split)]
            self.assertEqual(len(ids), len(set(ids)))
            expected = [f"{split}-{i:04d}" for i in range(1, SPLIT_SIZES[split] + 1)]
            self.assertEqual(sorted(ids), expected)

    def test_deterministic_for_given_seed(self):
        plan1 = build_plan(self.templates, seed=DEFAULT_SEED)
        plan2 = build_plan(self.templates, seed=DEFAULT_SEED)
        self.assertEqual(plan1.to_dict(), plan2.to_dict())

    def test_different_seed_gives_different_plan(self):
        plan1 = build_plan(self.templates, seed=1)
        plan2 = build_plan(self.templates, seed=2)
        self.assertNotEqual(plan1.to_dict(), plan2.to_dict())

    def test_variant_index_increments_correctly_for_repeats(self):
        plan = build_plan(self.templates, seed=DEFAULT_SEED)
        groups: dict[tuple, list[int]] = {}
        for e in plan.entries:
            key = (e.template_id, e.cell)
            groups.setdefault(key, []).append(e.variant_index)
        for key, indices in groups.items():
            with self.subTest(key=key):
                self.assertEqual(sorted(indices), list(range(len(indices))))

    def test_lenient_mode_has_no_shortfalls_now_that_coverage_is_complete(self):
        # With all 21 templates (t01-t21) authored to close every quota, the
        # real corpus has no shortfalls left to record. The shortfall-
        # recording mechanism itself is exercised separately in
        # TestReviewerQuotas using deliberately-incomplete synthetic
        # templates.
        plan = build_plan(self.templates, seed=DEFAULT_SEED, strict=False)
        self.assertEqual(len(plan.shortfalls), 0)

    def test_strict_mode_succeeds_now_that_coverage_is_complete(self):
        plan = build_plan(self.templates, seed=DEFAULT_SEED, strict=True)
        self.assertEqual(len(plan.entries), sum(SPLIT_SIZES.values()))

    def test_zero_templates_for_a_split_raises_unconditionally(self):
        eval_only = {tid: t for tid, t in self.templates.items() if t.pool == EVAL_POOL}
        with self.assertRaises(CoverageError):
            build_plan(eval_only, seed=DEFAULT_SEED, strict=False)
        with self.assertRaises(CoverageError):
            build_plan(eval_only, seed=DEFAULT_SEED, strict=True)

    def test_full_coverage_templates_build_without_raising(self):
        templates = {
            "synthetic_dev": _full_coverage_template("synthetic_dev", DEV_POOL),
            "synthetic_eval": _full_coverage_template("synthetic_eval", EVAL_POOL),
        }
        plan = build_plan(templates, seed=DEFAULT_SEED, strict=False)
        self.assertEqual(len(plan.by_split(DEV_POOL)), SPLIT_SIZES[DEV_POOL])
        self.assertEqual(len(plan.by_split(EVAL_POOL)), SPLIT_SIZES[EVAL_POOL])


class TestPlanIO(unittest.TestCase):
    def test_dict_round_trip(self):
        templates = load_templates()
        plan = build_plan(templates, seed=DEFAULT_SEED)
        restored = plan_from_dict(plan_to_dict(plan))
        self.assertEqual(restored, plan)

    def test_file_round_trip(self):
        templates = load_templates()
        plan = build_plan(templates, seed=DEFAULT_SEED)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "plan.json"
            write_plan(plan, path)
            restored = read_plan(path)
            self.assertEqual(restored, plan)
            # Confirm it's real, human-readable JSON on disk, not a pickle.
            json.loads(path.read_text(encoding="utf-8"))


class TestCoverageReporting(unittest.TestCase):
    def setUp(self):
        self.templates = load_templates()
        self.plan = build_plan(self.templates, seed=DEFAULT_SEED)
        self.report = coverage_report(self.plan, self.templates)

    def test_corpus_size(self):
        self.assertEqual(self.report["corpus_size"], sum(SPLIT_SIZES.values()))

    def test_axis_counts_sum_to_thread_count_per_split(self):
        for split in (DEV_POOL, EVAL_POOL):
            split_report = self.report["splits"][split]
            for axis in AXIS_NAMES:
                total = sum(split_report["axis_counts"][axis].values())
                self.assertEqual(total, split_report["thread_count"])

    def test_report_is_json_serializable(self):
        json.dumps(self.report)

    def test_reviewer_counts_and_divergence_present_when_templates_given(self):
        for split in (DEV_POOL, EVAL_POOL):
            split_report = self.report["splits"][split]
            self.assertIn("reviewer_counts", split_report)
            self.assertIn("policy_divergence_count", split_report)
            self.assertEqual(sum(split_report["reviewer_counts"].values()), split_report["thread_count"])

    def test_coverage_shortfalls_empty_now_that_coverage_is_complete(self):
        # All 21 templates (t01-t21) close every quota for their split, so
        # the real corpus has no shortfalls to flag.
        shortfalls = coverage_shortfalls(self.report)
        self.assertEqual(shortfalls, [])

    def test_assert_coverage_succeeds_now_that_coverage_is_complete(self):
        assert_coverage(self.plan, self.templates)

    def test_format_coverage_markdown_contains_expected_sections(self):
        md = format_coverage_markdown(self.report)
        self.assertIn("# Corpus coverage report", md)
        self.assertIn("## dev", md)
        self.assertIn("## eval", md)
        self.assertIn("### reviewer", md)


class TestSelectOverlap(unittest.TestCase):
    def setUp(self):
        self.templates = load_templates()
        self.plan = build_plan(self.templates, seed=DEFAULT_SEED)

    def test_selects_exactly_n_distinct_eval_thread_ids(self):
        ids = select_overlap(self.plan, seed=DEFAULT_SEED)
        self.assertEqual(len(ids), 60)
        self.assertEqual(len(set(ids)), 60)
        eval_ids = {e.thread_id for e in self.plan.by_split(EVAL_POOL)}
        self.assertTrue(all(i in eval_ids for i in ids))

    def test_deterministic_for_given_seed(self):
        ids1 = select_overlap(self.plan, seed=DEFAULT_SEED)
        ids2 = select_overlap(self.plan, seed=DEFAULT_SEED)
        self.assertEqual(ids1, ids2)

    def test_raises_when_n_exceeds_available(self):
        with self.assertRaises(ValueError):
            select_overlap(self.plan, seed=DEFAULT_SEED, n=1000)


if __name__ == "__main__":
    unittest.main()
