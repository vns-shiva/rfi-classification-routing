"""Unit tests for routing_eval.py. Run with:

    python -m unittest discover -s code/tests -t code

Every test builds its own corpus/predictions tree under a
tempfile.TemporaryDirectory() that is removed automatically on exit (pass or
fail) -- no fixture is left behind, per the project's "delete test data
after every test" rule. Unlike test_policy_audit.py, these fixtures never
need a real Cell or template -- routing_eval.py's load_splits() reads split
membership out of provenance/*.json as a plain dict, so a hand-written
{"split": "dev"|"eval"} is enough; Cell/stratification are outside this
module's import-decoupling contract entirely.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import routing_eval as re
from gating import gate
from routing_policy import route_from_fields
from schema import RFILabel, write_label


def _label(**overrides) -> RFILabel:
    # Defaults are internally consistent with routing_policy.route_rfi():
    # rfi_type="design_clarification" (no special rule fires) + primary_discipline
    # "Structural" -> R4_discipline_default -> "Structural Engineer", and
    # escalation_for(False, False, "priority") is False -- so this base fixture
    # already agrees with the policy; tests override fields to diverge deliberately.
    base = dict(
        thread_id="rfi-001", rfi_type="design_clarification",
        primary_discipline="Structural", secondary_disciplines=[],
        csi_division="05 00 00 (Metals)", urgency="priority",
        question_summary="Confirm beam-to-column connection detail at gridline C4.",
        referenced_documents=["S-401", "S-501"], proposed_solution="—",
        cost_impact=False, schedule_impact=False, answer_in_documents=False,
        deadline_text="within two weeks", assigned_reviewer="Structural Engineer",
        routing_rationale="Structural connection detail; no cost/code trigger.",
        escalation=False, confidence=0.8, status="open", annotator="gold",
        deadline_iso=None, deadline_resolution_status="unresolved",
    )
    base.update(overrides)
    return RFILabel(**base)


def _write_gold(corpus_dir: Path, label: RFILabel, split: str, template_id: str = "tmpl-1") -> None:
    labels_dir = corpus_dir / "labels"
    prov_dir = corpus_dir / "provenance"
    labels_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)
    write_label(label, labels_dir / f"{label.thread_id}.json")
    prov = {"thread_id": label.thread_id, "template_id": template_id, "cell": {"split": split}}
    (prov_dir / f"{label.thread_id}.json").write_text(json.dumps(prov), encoding="utf-8")


def _write_predicted(predictions_dir: Path, label: RFILabel, annotator_id: str = "pipeline") -> None:
    labels_dir = predictions_dir / f"labels_{annotator_id}"
    labels_dir.mkdir(parents=True, exist_ok=True)
    write_label(label, labels_dir / f"{label.thread_id}.json")


class TestLoaders(unittest.TestCase):
    def test_load_gold_labels_keys_by_thread_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_gold(corpus_dir, _label(thread_id="rfi-a"), split="eval")
            _write_gold(corpus_dir, _label(thread_id="rfi-b"), split="dev")
            gold = re.load_gold_labels(corpus_dir)
        self.assertEqual({"rfi-a", "rfi-b"}, set(gold))
        self.assertEqual("rfi-a", gold["rfi-a"].thread_id)

    def test_load_gold_labels_raises_on_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                re.load_gold_labels(corpus_dir)

    def test_load_gold_labels_raises_on_invalid_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_gold(corpus_dir, _label(thread_id="rfi-bad", rfi_type="not_a_real_rfi_type"), split="eval")
            with self.assertRaisesRegex(ValueError, "invalid gold label"):
                re.load_gold_labels(corpus_dir)

    def test_load_gold_labels_raises_on_duplicate_thread_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            labels_dir = corpus_dir / "labels"
            labels_dir.mkdir(parents=True)
            write_label(_label(thread_id="rfi-dup"), labels_dir / "a_first.json")
            write_label(_label(thread_id="rfi-dup"), labels_dir / "b_second.json")
            with self.assertRaisesRegex(ValueError, "duplicate gold thread_id"):
                re.load_gold_labels(corpus_dir)

    def test_load_predicted_labels_reads_annotator_specific_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictions_dir = Path(tmp)
            _write_predicted(predictions_dir, _label(thread_id="rfi-a", annotator="pipeline"), annotator_id="pipeline")
            _write_predicted(predictions_dir, _label(thread_id="rfi-c", annotator="annotator_b"), annotator_id="annotator_b")
            pipeline, pipeline_invalid = re.load_predicted_labels(predictions_dir, "pipeline")
            annotator_b, annotator_b_invalid = re.load_predicted_labels(predictions_dir, "annotator_b")
        self.assertEqual({"rfi-a"}, set(pipeline))
        self.assertEqual({"rfi-c"}, set(annotator_b))
        self.assertEqual([], pipeline_invalid)
        self.assertEqual([], annotator_b_invalid)

    def test_load_predicted_labels_raises_on_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictions_dir = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                re.load_predicted_labels(predictions_dir, "pipeline")

    def test_load_predicted_labels_collects_invalid_prediction_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictions_dir = Path(tmp)
            bad = _label(thread_id="rfi-bad", rfi_type="not_a_real_rfi_type")
            _write_predicted(predictions_dir, bad, annotator_id="pipeline")
            predicted, invalid = re.load_predicted_labels(predictions_dir, "pipeline")
        self.assertEqual({}, predicted)
        self.assertEqual(1, len(invalid))
        self.assertEqual("rfi-bad", invalid[0].thread_id)
        self.assertTrue(any("rfi_type" in e for e in invalid[0].errors))

    def test_load_predicted_labels_collects_duplicate_thread_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictions_dir = Path(tmp)
            labels_dir = predictions_dir / "labels_pipeline"
            labels_dir.mkdir(parents=True)
            write_label(_label(thread_id="rfi-dup"), labels_dir / "a_first.json")
            write_label(_label(thread_id="rfi-dup"), labels_dir / "b_second.json")
            predicted, invalid = re.load_predicted_labels(predictions_dir, "pipeline")
        self.assertEqual({"rfi-dup"}, set(predicted))
        self.assertEqual(1, len(invalid))
        self.assertEqual("rfi-dup", invalid[0].thread_id)
        self.assertEqual(["duplicate thread_id"], invalid[0].errors)

    def test_load_splits_reads_split_from_provenance_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_gold(corpus_dir, _label(thread_id="rfi-a"), split="eval")
            _write_gold(corpus_dir, _label(thread_id="rfi-b"), split="dev")
            splits = re.load_splits(corpus_dir)
        self.assertEqual({"rfi-a": "eval", "rfi-b": "dev"}, splits)

    def test_load_splits_skips_provenance_missing_split_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            prov_dir = corpus_dir / "provenance"
            prov_dir.mkdir(parents=True)
            (prov_dir / "rfi-x.json").write_text(json.dumps({"thread_id": "rfi-x", "cell": {}}), encoding="utf-8")
            splits = re.load_splits(corpus_dir)
        self.assertEqual({}, splits)


class TestMatchThreads(unittest.TestCase):
    def test_matches_by_thread_id_and_reports_missing(self):
        gold = {"a": _label(thread_id="a"), "b": _label(thread_id="b")}
        predicted = {"a": _label(thread_id="a", annotator="pipeline")}
        matched, missing, extra = re.match_threads(gold, predicted, {"a": "eval", "b": "eval"})
        self.assertEqual(1, len(matched))
        self.assertEqual("a", matched[0].thread_id)
        self.assertEqual("eval", matched[0].split)
        self.assertEqual(["b"], missing)
        self.assertEqual([], extra)

    def test_unresolvable_split_is_none_not_a_crash(self):
        gold = {"a": _label(thread_id="a")}
        predicted = {"a": _label(thread_id="a")}
        matched, _, _ = re.match_threads(gold, predicted, {})
        self.assertIsNone(matched[0].split)

    def test_reports_extra_predicted_threads_with_no_gold_label(self):
        gold = {"a": _label(thread_id="a")}
        predicted = {"a": _label(thread_id="a"), "z": _label(thread_id="z")}
        matched, missing, extra = re.match_threads(gold, predicted, {"a": "eval"})
        self.assertEqual(1, len(matched))
        self.assertEqual([], missing)
        self.assertEqual(["z"], extra)


class TestScoreOracle(unittest.TestCase):
    def test_agrees_when_gold_matches_policy(self):
        thread = re.MatchedThread(thread_id="t1", split="eval", gold=_label(), predicted=_label())
        result = re.score_oracle([thread])
        self.assertEqual(1, result.n)
        self.assertEqual(1.0, result.reviewer_accuracy)
        self.assertEqual(1.0, result.escalation_accuracy)

    def test_diverges_when_gold_assigned_reviewer_overridden(self):
        # design_clarification/Structural policy-routes to "Structural Engineer";
        # authoring the gold row with a different reviewer simulates a
        # policy_divergence=True template row (per policy_audit.py's divergence_census).
        gold = _label(assigned_reviewer="Cost/Change-Order Manager")
        thread = re.MatchedThread(thread_id="t1", split="eval", gold=gold, predicted=gold)
        result = re.score_oracle([thread])
        self.assertEqual(0, result.n_reviewer_correct)
        self.assertEqual(0.0, result.reviewer_accuracy)


class TestScoreComposed(unittest.TestCase):
    def test_routes_predicted_fields_through_policy_not_raw_reviewer_text(self):
        gold = _label(thread_id="t1", primary_discipline="Structural", assigned_reviewer="Structural Engineer")
        # LLM misclassified the discipline as Mechanical but happened to copy the
        # gold reviewer text verbatim -- a raw-field comparison would call this
        # "correct"; score_composed must route predicted fields through the policy
        # (-> "Mechanical Engineer") and score it as a miss instead.
        predicted = _label(thread_id="t1", primary_discipline="Mechanical", assigned_reviewer="Structural Engineer")
        thread = re.MatchedThread(thread_id="t1", split="eval", gold=gold, predicted=predicted)
        result = re.score_composed([thread], gate_threshold=0.6)
        self.assertEqual(0, result.n_reviewer_correct)
        gb = result.gate_breakdown
        # confidence=0.8 (the _label default) clears gate_threshold=0.6 and
        # escalation=False, so this thread is ungated, not gated.
        self.assertEqual(0, gb.n_gated)
        self.assertEqual(0, gb.n_gated_reviewer_correct)
        self.assertEqual(1, gb.n_ungated)
        self.assertEqual(0, gb.n_ungated_reviewer_correct)

    def test_composed_compares_against_golds_raw_reviewer_text_not_policy_on_gold(self):
        # gold is policy-divergent: its stored assigned_reviewer disagrees with
        # what route_from_fields(gold) would produce from its own fields
        # (Structural -> "Structural Engineer"), simulating a template row
        # with policy_divergence=True. predicted's fields also route to
        # "Structural Engineer". If score_composed compared against
        # route_from_fields(gold) instead of gold's raw assigned_reviewer
        # text, this would incorrectly score as a match.
        gold = _label(thread_id="t1", assigned_reviewer="Cost/Change-Order Manager")
        predicted = _label(thread_id="t1", assigned_reviewer="Cost/Change-Order Manager")
        thread = re.MatchedThread(thread_id="t1", split="eval", gold=gold, predicted=predicted)
        result = re.score_composed([thread], gate_threshold=0.6)
        self.assertEqual(0, result.n_reviewer_correct)

    def test_composed_derives_escalation_via_policy_not_predicted_raw_field(self):
        # predicted.escalation=True is the LLM's raw guess, but composed must
        # route predicted's cost/schedule/urgency fields through the policy
        # (escalation_for(False, False, "priority") is False per the _label
        # docstring) rather than trusting that raw field -- direct, which
        # does trust the raw field, must disagree with composed here.
        gold = _label(thread_id="t1", escalation=False)
        predicted = _label(thread_id="t1", escalation=True, urgency="priority", confidence=0.9)
        thread = re.MatchedThread(thread_id="t1", split="eval", gold=gold, predicted=predicted)
        composed = re.score_composed([thread], gate_threshold=0.6)
        direct = re.score_direct([thread], gate_threshold=0.6)
        self.assertEqual(1, composed.n_escalation_correct)
        self.assertEqual(0, direct.n_escalation_correct)

    def test_gate_threshold_is_a_real_parameter_not_hardcoded(self):
        gold = _label(thread_id="t1")
        predicted = _label(thread_id="t1", confidence=0.7)
        thread = re.MatchedThread(thread_id="t1", split="eval", gold=gold, predicted=predicted)
        lenient = re.score_composed([thread], gate_threshold=0.6)
        strict = re.score_composed([thread], gate_threshold=0.95)
        self.assertEqual(0, lenient.gate_breakdown.n_gated)
        self.assertEqual(1, strict.gate_breakdown.n_gated)

    def test_gate_breakdown_splits_by_predicted_confidence(self):
        gold = _label(thread_id="t1", assigned_reviewer="Structural Engineer")
        confident = _label(thread_id="t1", confidence=0.9, assigned_reviewer="Structural Engineer")
        gold2 = _label(thread_id="t2", assigned_reviewer="Structural Engineer")
        unconfident = _label(thread_id="t2", confidence=0.1, assigned_reviewer="Structural Engineer")
        threads = [
            re.MatchedThread(thread_id="t1", split="eval", gold=gold, predicted=confident),
            re.MatchedThread(thread_id="t2", split="eval", gold=gold2, predicted=unconfident),
        ]
        result = re.score_composed(threads, gate_threshold=0.6)
        gb = result.gate_breakdown
        self.assertEqual(1, gb.n_gated)
        self.assertEqual(1, gb.n_ungated)
        self.assertEqual(
            result.n_reviewer_correct,
            gb.n_gated_reviewer_correct + gb.n_ungated_reviewer_correct,
        )

    def test_gate_breakdown_distinguishes_escalation_from_confidence_gating(self):
        # t1: confidence clears the threshold but escalation=True must still gate it,
        # under the "escalation" category, not lumped in with low-confidence gating.
        gold1 = _label(thread_id="t1", assigned_reviewer="Structural Engineer")
        escalated = _label(thread_id="t1", confidence=0.9, assigned_reviewer="Structural Engineer", escalation=True)
        # t2: confidence alone is below threshold, no escalation -- must land in
        # the "confidence" category.
        gold2 = _label(thread_id="t2", assigned_reviewer="Structural Engineer")
        unconfident = _label(thread_id="t2", confidence=0.1, assigned_reviewer="Structural Engineer", escalation=False)
        threads = [
            re.MatchedThread(thread_id="t1", split="eval", gold=gold1, predicted=escalated),
            re.MatchedThread(thread_id="t2", split="eval", gold=gold2, predicted=unconfident),
        ]
        result = re.score_composed(threads, gate_threshold=0.6)
        by_category = result.gate_breakdown.n_gated_by_category
        self.assertEqual(1, by_category.get("escalation", 0))
        self.assertEqual(1, by_category.get("confidence", 0))

    def test_gate_breakdown_categorizes_missing_required_field(self):
        gold = _label(thread_id="t1")
        # confidence and escalation both clear the gate; only the placeholder
        # assigned_reviewer should land this in the "missing_field" category.
        predicted = _label(thread_id="t1", confidence=0.9, escalation=False, assigned_reviewer="tbd")
        thread = re.MatchedThread(thread_id="t1", split="eval", gold=gold, predicted=predicted)
        result = re.score_composed([thread], gate_threshold=0.6)
        by_category = result.gate_breakdown.n_gated_by_category
        self.assertEqual(1, by_category.get("missing_field", 0))
        self.assertEqual(0, by_category.get("confidence", 0))
        self.assertEqual(0, by_category.get("escalation", 0))


class TestScoreDirect(unittest.TestCase):
    def test_compares_predicted_reviewer_directly_no_policy(self):
        gold = _label(thread_id="t1", assigned_reviewer="Structural Engineer")
        # predicted fields would route through policy to "Structural Engineer" too,
        # but the LLM's own raw assigned_reviewer guess is different -- direct arm
        # must use that raw value, not re-derive it via route_from_fields().
        predicted = _label(thread_id="t1", primary_discipline="Structural", assigned_reviewer="Mechanical Engineer")
        thread = re.MatchedThread(thread_id="t1", split="eval", gold=gold, predicted=predicted)
        result = re.score_direct([thread], gate_threshold=0.6)
        self.assertEqual(0, result.n_reviewer_correct)


class TestScorePolicyFidelity(unittest.TestCase):
    def test_compares_policy_on_predicted_vs_policy_on_gold(self):
        # gold fields route to "Structural Engineer"; predicted fields (different
        # discipline) route to "Mechanical Engineer" -- policy_fidelity compares
        # those two policy outputs directly, ignoring both labels' own
        # assigned_reviewer text entirely.
        gold = _label(thread_id="t1", primary_discipline="Structural", assigned_reviewer="anything")
        predicted = _label(thread_id="t1", primary_discipline="Mechanical", assigned_reviewer="anything else")
        thread = re.MatchedThread(thread_id="t1", split="eval", gold=gold, predicted=predicted)
        result = re.score_policy_fidelity([thread])
        self.assertEqual(0, result.n_reviewer_correct)

    def test_agrees_when_predicted_and_gold_fields_route_identically(self):
        gold = _label(thread_id="t1", primary_discipline="Structural")
        predicted = _label(thread_id="t1", primary_discipline="Structural")
        thread = re.MatchedThread(thread_id="t1", split="eval", gold=gold, predicted=predicted)
        result = re.score_policy_fidelity([thread])
        self.assertEqual(1, result.n_reviewer_correct)


class TestMajorityBaseline(unittest.TestCase):
    def test_fit_returns_none_on_empty_dev_pool(self):
        reviewer, escalation = re.fit_majority_baseline([])
        self.assertIsNone(reviewer)
        self.assertIsNone(escalation)

    def test_fit_picks_most_common_reviewer_and_escalation(self):
        labels = [
            _label(thread_id="d1", assigned_reviewer="Structural Engineer", escalation=False),
            _label(thread_id="d2", assigned_reviewer="Structural Engineer", escalation=False),
            _label(thread_id="d3", assigned_reviewer="Structural Engineer", escalation=True),
            _label(thread_id="d4", assigned_reviewer="Mechanical Engineer", escalation=False),
        ]
        reviewer, escalation = re.fit_majority_baseline(labels)
        self.assertEqual("Structural Engineer", reviewer)
        self.assertEqual(False, escalation)

    def test_score_majority_applies_fixed_baseline_uniformly(self):
        threads = [
            re.MatchedThread(thread_id="e1", split="eval", gold=_label(thread_id="e1", assigned_reviewer="Structural Engineer"), predicted=_label(thread_id="e1")),
            re.MatchedThread(thread_id="e2", split="eval", gold=_label(thread_id="e2", assigned_reviewer="Mechanical Engineer"), predicted=_label(thread_id="e2")),
        ]
        result = re.score_majority(threads, "Structural Engineer", False)
        self.assertEqual(2, result.n)
        self.assertEqual(1, result.n_reviewer_correct)

    def test_score_majority_is_undefined_with_no_dev_data(self):
        threads = [re.MatchedThread(thread_id="e1", split="eval", gold=_label(), predicted=_label())]
        result = re.score_majority(threads, None, None)
        self.assertEqual(0, result.n)
        self.assertTrue(result.notes)


class TestSafeKappa(unittest.TestCase):
    def test_insufficient_data_below_two_units(self):
        kappa, reason = re._safe_kappa(["a"], ["a"], ["a"])
        self.assertIsNone(kappa)
        self.assertEqual("insufficient_data", reason)

    def test_no_variance_reported_when_single_category(self):
        kappa, reason = re._safe_kappa(["a", "a", "a"], ["a", "a", "a"], ["a"])
        self.assertIsNone(kappa)
        self.assertEqual("no_variance_in_categories", reason)

    def test_finite_kappa_on_a_real_two_category_disagreement(self):
        # Confusion matrix [[2,1],[1,2]] over n=6: po=4/6, pe=0.5 -> kappa=(4/6-0.5)/0.5=1/3.
        a = ["x", "x", "y", "y", "x", "y"]
        b = ["x", "y", "y", "x", "x", "y"]
        kappa, reason = re._safe_kappa(a, b, ["x", "y"])
        self.assertIsNone(reason)
        self.assertAlmostEqual(1 / 3, kappa)


class TestRunEval(unittest.TestCase):
    def test_end_to_end_scores_only_eval_split_and_fits_majority_from_dev(self):
        with tempfile.TemporaryDirectory() as corpus_tmp, tempfile.TemporaryDirectory() as pred_tmp:
            corpus_dir = Path(corpus_tmp)
            predictions_dir = Path(pred_tmp)

            # DEV split: 3 gold-only threads (no predictions), fits the majority baseline.
            _write_gold(corpus_dir, _label(thread_id="dev-1", assigned_reviewer="Structural Engineer"), split="dev")
            _write_gold(corpus_dir, _label(thread_id="dev-2", assigned_reviewer="Structural Engineer"), split="dev")
            _write_gold(corpus_dir, _label(thread_id="dev-3", assigned_reviewer="Mechanical Engineer"), split="dev")

            # EVAL split: 2 threads with matched predictions, 1 gold thread with no
            # prediction at all (exercises missing_thread_ids / strict_failures).
            _write_gold(corpus_dir, _label(thread_id="eval-1", assigned_reviewer="Structural Engineer"), split="eval")
            _write_predicted(predictions_dir, _label(thread_id="eval-1", primary_discipline="Structural", assigned_reviewer="Structural Engineer", confidence=0.9))
            _write_gold(corpus_dir, _label(thread_id="eval-2", assigned_reviewer="Mechanical Engineer", primary_discipline="Mechanical", csi_division="23 00 00 (HVAC)"), split="eval")
            _write_predicted(predictions_dir, _label(thread_id="eval-2", primary_discipline="Mechanical", csi_division="23 00 00 (HVAC)", assigned_reviewer="Structural Engineer", confidence=0.9))
            _write_gold(corpus_dir, _label(thread_id="eval-3", assigned_reviewer="Structural Engineer"), split="eval")

            report = re.run_eval(corpus_dir, predictions_dir, "pipeline")

        self.assertEqual(6, report.n_gold)
        self.assertEqual(2, report.n_predicted)
        self.assertEqual(2, report.n_matched)
        self.assertEqual(2, report.n_eval_split_matched)
        self.assertEqual(3, report.n_dev_split_gold)
        # dev-1..3 never got predictions either (this fixture only predicts on
        # eval threads, as a real experiment would) -- match_threads() doesn't
        # discriminate by split when reporting missing predictions.
        self.assertEqual(["dev-1", "dev-2", "dev-3", "eval-3"], report.missing_thread_ids)
        self.assertEqual("Structural Engineer", report.majority_reviewer)
        self.assertEqual(1, report.arms["majority"].n_reviewer_correct)  # majority guesses "Structural Engineer" for both eval threads, matches eval-1 only
        self.assertEqual(2, report.arms["oracle"].n)
        self.assertIn("1 EVAL-split gold threads have no matched prediction", report.strict_failures)

    def test_strict_failure_when_no_eval_matches_exist(self):
        with tempfile.TemporaryDirectory() as corpus_tmp, tempfile.TemporaryDirectory() as pred_tmp:
            corpus_dir = Path(corpus_tmp)
            predictions_dir = Path(pred_tmp)
            _write_gold(corpus_dir, _label(thread_id="dev-1"), split="dev")
            _write_predicted(predictions_dir, _label(thread_id="dev-1"))
            report = re.run_eval(corpus_dir, predictions_dir, "pipeline")
        self.assertEqual(0, report.n_eval_split_matched)
        self.assertTrue(any("no EVAL-split" in f for f in report.strict_failures))
        # run_eval must score every arm from eval_matched, not the raw matched
        # list -- dev-1 is matched but belongs to the DEV split, so every arm
        # (including majority, fit on this same DEV data) must report n=0.
        self.assertEqual(0, report.arms["oracle"].n)
        self.assertEqual(0, report.arms["composed"].n)
        self.assertEqual(0, report.arms["direct"].n)
        self.assertEqual(0, report.arms["policy_fidelity"].n)
        self.assertEqual(0, report.arms["majority"].n)

    def test_strict_failure_when_no_dev_split_gold_labels_exist(self):
        with tempfile.TemporaryDirectory() as corpus_tmp, tempfile.TemporaryDirectory() as pred_tmp:
            corpus_dir = Path(corpus_tmp)
            predictions_dir = Path(pred_tmp)
            _write_gold(corpus_dir, _label(thread_id="eval-1"), split="eval")
            _write_predicted(predictions_dir, _label(thread_id="eval-1"))
            report = re.run_eval(corpus_dir, predictions_dir, "pipeline")
        self.assertIn(
            "no DEV-split gold labels found; majority baseline is undefined",
            report.strict_failures,
        )
        self.assertEqual(0, report.arms["majority"].n)

    def test_invalid_and_extra_predictions_are_excluded_and_reported(self):
        with tempfile.TemporaryDirectory() as corpus_tmp, tempfile.TemporaryDirectory() as pred_tmp:
            corpus_dir = Path(corpus_tmp)
            predictions_dir = Path(pred_tmp)
            _write_gold(corpus_dir, _label(thread_id="eval-1", assigned_reviewer="Structural Engineer"), split="eval")
            _write_predicted(predictions_dir, _label(thread_id="eval-1", primary_discipline="Structural", assigned_reviewer="Structural Engineer", confidence=0.9))
            # Predicted thread with no matching gold label at all.
            _write_predicted(predictions_dir, _label(thread_id="eval-extra", confidence=0.9))
            # Predicted label that fails schema validation -- must be excluded from
            # scoring entirely, not silently dropped or crash the run.
            _write_predicted(predictions_dir, _label(thread_id="eval-bad", rfi_type="not_a_real_rfi_type"))

            report = re.run_eval(corpus_dir, predictions_dir, "pipeline")

        self.assertEqual(1, report.n_invalid_predictions)
        self.assertEqual(["eval-bad"], report.invalid_prediction_thread_ids)
        self.assertEqual(["eval-extra"], report.extra_thread_ids)
        self.assertIn(
            "1 predicted labels failed schema.validate_label() and were excluded from scoring",
            report.strict_failures,
        )
        self.assertIn(
            "1 predicted thread_ids have no matching gold label",
            report.strict_failures,
        )


class TestReportWriters(unittest.TestCase):
    def test_json_and_markdown_reports_are_written_and_parseable(self):
        with tempfile.TemporaryDirectory() as corpus_tmp, tempfile.TemporaryDirectory() as pred_tmp, tempfile.TemporaryDirectory() as out_tmp:
            corpus_dir = Path(corpus_tmp)
            predictions_dir = Path(pred_tmp)
            out_dir = Path(out_tmp)
            _write_gold(corpus_dir, _label(thread_id="dev-1"), split="dev")
            _write_gold(corpus_dir, _label(thread_id="eval-1"), split="eval")
            _write_predicted(predictions_dir, _label(thread_id="eval-1"))

            report = re.run_eval(corpus_dir, predictions_dir, "pipeline")
            re.write_json_report(report, out_dir / "report.json")
            re.write_markdown_report(report, out_dir / "report.md")

            parsed = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report.n_gold, parsed["n_gold"])
            markdown = (out_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("# Routing Evaluation Report", markdown)
        self.assertIn("## oracle", markdown)
        self.assertIn("## majority", markdown)

    def test_nan_accuracy_serializes_as_null_not_nan_literal(self):
        # An arm with n=0 (e.g. majority with no dev data) has nan accuracy;
        # _to_jsonable must turn that into JSON null, not the invalid "NaN" token.
        with tempfile.TemporaryDirectory() as out_tmp:
            out_dir = Path(out_tmp)
            result = re.score_majority([], None, None)
            report = re.EvalReport(
                corpus_dir="c", predictions_dir="p", annotator_id="pipeline",
                n_gold=0, n_predicted=0, n_matched=0, n_eval_split_matched=0, n_dev_split_gold=0,
                missing_thread_ids=[], gate_threshold=0.6, majority_reviewer=None, majority_escalation=None,
                arms={"majority": result}, strict_failures=[],
            )
            re.write_json_report(report, out_dir / "report.json")
            parsed = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        self.assertIsNone(parsed["arms"]["majority"]["reviewer_accuracy"])


if __name__ == "__main__":
    unittest.main()
