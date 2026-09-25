"""Unit tests for annotator_agreement.py. Run with:

    python -m unittest discover -s code/tests -t code

Every test builds its own corpus under a tempfile.TemporaryDirectory() and
that directory is removed automatically on exit (pass or fail) — no fixture
is left behind, per the project's "delete test data after every test" rule.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import annotator_agreement as aa
from schema import RFILabel, write_label


def _label(**overrides) -> RFILabel:
    base = dict(
        thread_id="rfi-001",
        rfi_type="design_clarification",
        primary_discipline="Structural",
        secondary_disciplines=[],
        csi_division="05 00 00 (Metals)",
        urgency="priority",
        question_summary="Confirm beam-to-column connection detail at gridline C4.",
        referenced_documents=["S-401", "S-501"],
        proposed_solution="—",
        cost_impact=False,
        schedule_impact=True,
        answer_in_documents=False,
        deadline_text="within two weeks",
        assigned_reviewer="Structural Engineer",
        routing_rationale="Structural connection detail; no cost/code trigger.",
        escalation=False,
        confidence=0.8,
        status="open",
        annotator="gold",
        deadline_iso=None,
        deadline_resolution_status="unresolved",
    )
    base.update(overrides)
    return RFILabel(**base)


def _annotator_dir(corpus_dir: Path, annotator_id: str) -> Path:
    return corpus_dir / ("labels" if annotator_id == "gold" else f"labels_{annotator_id}")


def _write_corpus(corpus_dir: Path, labels_by_annotator: dict[str, list[RFILabel]]) -> None:
    for annotator_id, labels in labels_by_annotator.items():
        label_dir = _annotator_dir(corpus_dir, annotator_id)
        label_dir.mkdir(parents=True, exist_ok=True)
        for label in labels:
            write_label(label, label_dir / f"{label.thread_id}.json")


def _write_provenance(corpus_dir: Path, thread_id: str, *, rfi_type: str, adversarial: str) -> None:
    prov_dir = corpus_dir / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    data = {"cell": {"rfi_type": rfi_type, "adversarial": adversarial}}
    (prov_dir / f"{thread_id}.json").write_text(json.dumps(data), encoding="utf-8")


class TestFieldSpecsExhaustiveness(unittest.TestCase):
    def test_covers_every_rfilabel_field_exactly_once(self):
        rfilabel_fields = {f.name for f in dataclasses.fields(RFILabel)}
        spec_fields = {s.name for s in aa.FIELD_SPECS}
        self.assertEqual(rfilabel_fields, spec_fields)
        self.assertEqual(len(rfilabel_fields), 21)
        self.assertEqual(len(aa.FIELD_SPECS), len(spec_fields), "FIELD_SPECS has a duplicate field name")


class TestAlignment(unittest.TestCase):
    def test_annotator_directory_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            label_dir = corpus_dir / "labels_annotator_b"
            label_dir.mkdir(parents=True)
            bad_label = _label(annotator="gold")  # written under labels_annotator_b/, but annotator says "gold"
            write_label(bad_label, label_dir / "rfi-001.json")
            with self.assertRaises(ValueError):
                aa.load_annotator_labels(label_dir, "annotator_b")

    def test_invalid_label_surfaces_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            label_dir = corpus_dir / "labels"
            label_dir.mkdir(parents=True)
            bad_label = _label(rfi_type="not_a_real_type")
            write_label(bad_label, label_dir / "rfi-001.json")
            with self.assertRaises(ValueError):
                aa.load_annotator_labels(label_dir, "gold")

    def test_partial_coverage_intersects_thread_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(thread_id="rfi-001", annotator="gold"), _label(thread_id="rfi-002", annotator="gold")],
                "annotator_b": [_label(thread_id="rfi-001", annotator="annotator_b")],
            })
            aligned = aa.align_annotators(corpus_dir)
            self.assertEqual(aligned.thread_ids, ["rfi-001"])
            self.assertEqual(sorted(aligned.annotator_ids), ["annotator_b", "gold"])

    def test_framing_is_inter_annotator_for_two_human_annotators(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(thread_id="rfi-001", annotator="gold")],
                "annotator_b": [_label(thread_id="rfi-001", annotator="annotator_b")],
            })
            aligned = aa.align_annotators(corpus_dir)
            self.assertEqual(aligned.framing, "inter_annotator")

    def test_framing_is_recoverability_when_pipeline_is_an_annotator(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(thread_id="rfi-001", annotator="gold")],
                "pipeline": [_label(thread_id="rfi-001", annotator="pipeline")],
            })
            aligned = aa.align_annotators(corpus_dir)
            self.assertEqual(aligned.framing, "recoverability")


class TestFieldAgreementIdenticalLabels(unittest.TestCase):
    def _aligned_identical(self, n=3):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            gold_labels = []
            b_labels = []
            for i in range(n):
                tid = f"rfi-{i:03d}"
                gold_labels.append(_label(thread_id=tid, annotator="gold"))
                b_labels.append(_label(thread_id=tid, annotator="annotator_b"))
            _write_corpus(corpus_dir, {"gold": gold_labels, "annotator_b": b_labels})
            return aa.align_annotators(corpus_dir)

    def test_nominal_field_is_perfect_or_no_variance(self):
        aligned = self._aligned_identical()
        res = aa.field_agreement(aligned, "rfi_type")
        self.assertIn(res.summary["undefined_reason"], (None, "no_variance"))
        if res.summary["alpha"] is not None:
            self.assertEqual(res.summary["alpha"], 1.0)
        self.assertEqual(res.summary["percent_agreement"], 1.0)

    def test_ordinal_field_percent_agreement_is_perfect(self):
        aligned = self._aligned_identical()
        res = aa.field_agreement(aligned, "urgency")
        self.assertEqual(res.summary["percent_agreement"], 1.0)

    def test_set_field_exact_match_rate_is_perfect(self):
        aligned = self._aligned_identical()
        res = aa.field_agreement(aligned, "referenced_documents")
        self.assertEqual(res.summary["exact_match_rate"], 1.0)
        self.assertEqual(res.summary["micro_f1"], 1.0)

    def test_text_field_similarity_is_near_one_for_identical_text(self):
        aligned = self._aligned_identical()
        res = aa.field_agreement(aligned, "question_summary")
        self.assertGreater(res.summary["mean_cosine_similarity"], 0.99)

    def test_sentinel_text_field_agreement_rate_is_perfect(self):
        aligned = self._aligned_identical()
        res = aa.field_agreement(aligned, "proposed_solution")
        self.assertEqual(res.summary["sentinel_agreement_rate"], 1.0)
        self.assertEqual(res.summary["n_both_sentinel"], 3)

    def test_no_disagreements_when_identical(self):
        aligned = self._aligned_identical()
        diffs = aa.disagreements(aligned)
        self.assertEqual(diffs, [])


class TestFieldAgreementDisagreements(unittest.TestCase):
    def test_nominal_disagreement_drops_percent_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(thread_id="rfi-001", annotator="gold", rfi_type="design_clarification")],
                "annotator_b": [_label(thread_id="rfi-001", annotator="annotator_b", rfi_type="document_discrepancy")],
            })
            aligned = aa.align_annotators(corpus_dir)
            res = aa.field_agreement(aligned, "rfi_type")
            self.assertEqual(res.summary["percent_agreement"], 0.0)

    def test_boolean_disagreement_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(thread_id="rfi-001", annotator="gold", cost_impact=False)],
                "annotator_b": [_label(thread_id="rfi-001", annotator="annotator_b", cost_impact=True)],
            })
            aligned = aa.align_annotators(corpus_dir)
            res = aa.field_agreement(aligned, "cost_impact")
            self.assertEqual(res.summary["percent_agreement"], 0.0)

    def test_set_field_partial_overlap_is_between_zero_and_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(thread_id="rfi-001", annotator="gold", referenced_documents=["S-401", "S-501"])],
                "annotator_b": [_label(thread_id="rfi-001", annotator="annotator_b", referenced_documents=["S-401"])],
            })
            aligned = aa.align_annotators(corpus_dir)
            res = aa.field_agreement(aligned, "referenced_documents")
            self.assertLess(res.summary["exact_match_rate"], 1.0)
            self.assertGreater(res.summary["micro_f1"], 0.0)

    def test_sentinel_mismatch_is_recorded_separately_from_similarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(thread_id="rfi-001", annotator="gold", proposed_solution="—")],
                "annotator_b": [_label(
                    thread_id="rfi-001", annotator="annotator_b",
                    proposed_solution="Add a stiffener plate at the connection.",
                )],
            })
            aligned = aa.align_annotators(corpus_dir)
            res = aa.field_agreement(aligned, "proposed_solution")
            self.assertEqual(res.summary["n_sentinel_mismatch"], 1)
            self.assertEqual(res.summary["n_both_real"], 0)

    def test_date_field_offset_computed_only_when_both_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(
                    thread_id="rfi-001", annotator="gold",
                    deadline_resolution_status="resolved_to_date", deadline_iso="2025-03-14",
                )],
                "annotator_b": [_label(
                    thread_id="rfi-001", annotator="annotator_b",
                    deadline_resolution_status="resolved_to_date", deadline_iso="2025-03-17",
                )],
            })
            aligned = aa.align_annotators(corpus_dir)
            res = aa.field_agreement(aligned, "deadline_iso")
            self.assertEqual(res.summary["mean_abs_day_offset"], 3.0)
            self.assertEqual(res.n_applicable, 1)

    def test_csi_division_excludes_both_general_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(
                    thread_id="rfi-001", annotator="gold",
                    primary_discipline="General", csi_division="—",
                )],
                "annotator_b": [_label(
                    thread_id="rfi-001", annotator="annotator_b",
                    primary_discipline="General", csi_division="—",
                )],
            })
            aligned = aa.align_annotators(corpus_dir)
            res = aa.field_agreement(aligned, "csi_division")
            self.assertEqual(res.summary["n_excluded_general_general"], 1)
            self.assertEqual(res.n_applicable, 0)

    def test_constant_field_mismatch_is_flagged_but_not_reportable(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(thread_id="rfi-001", annotator="gold", status="open")],
                "annotator_b": [_label(thread_id="rfi-001", annotator="annotator_b", status="closed")],
            })
            aligned = aa.align_annotators(corpus_dir)
            res = aa.field_agreement(aligned, "status")
            self.assertFalse(res.reportable)
            self.assertEqual(res.summary["n_mismatched"], 1)
            self.assertNotEqual(res.note, "")


class TestDisagreements(unittest.TestCase):
    def test_disagreement_list_reports_thread_and_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(thread_id="rfi-001", annotator="gold", urgency="priority")],
                "annotator_b": [_label(thread_id="rfi-001", annotator="annotator_b", urgency="urgent")],
            })
            aligned = aa.align_annotators(corpus_dir)
            diffs = aa.disagreements(aligned, field_names=("urgency",))
            self.assertEqual(len(diffs), 1)
            self.assertEqual(diffs[0].thread_id, "rfi-001")
            self.assertEqual(diffs[0].field, "urgency")

    def test_set_field_order_only_difference_is_not_a_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(thread_id="rfi-001", annotator="gold", referenced_documents=["S-401", "S-501"])],
                "annotator_b": [_label(thread_id="rfi-001", annotator="annotator_b", referenced_documents=["S-501", "S-401"])],
            })
            aligned = aa.align_annotators(corpus_dir)
            diffs = aa.disagreements(aligned, field_names=("referenced_documents",))
            self.assertEqual(diffs, [])


class TestPolicyAgreement(unittest.TestCase):
    def test_matching_inputs_and_reviewer_gives_full_validity(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(thread_id="rfi-001", annotator="gold")],
                "annotator_b": [_label(thread_id="rfi-001", annotator="annotator_b")],
            })
            aligned = aa.align_annotators(corpus_dir)
            result = aa.policy_agreement(aligned)
            self.assertEqual(result.input_agreement_rate, 1.0)
            self.assertEqual(result.policy_validity_rate["gold"], 1.0)
            self.assertEqual(result.policy_validity_rate["annotator_b"], 1.0)
            self.assertEqual(result.n_input_agreement_units, 1)

    def test_mismatched_inputs_excluded_from_conditional_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [_label(thread_id="rfi-001", annotator="gold", urgency="priority")],
                "annotator_b": [_label(thread_id="rfi-001", annotator="annotator_b", urgency="urgent")],
            })
            aligned = aa.align_annotators(corpus_dir)
            result = aa.policy_agreement(aligned)
            self.assertEqual(result.input_agreement_rate, 0.0)
            self.assertEqual(result.n_input_agreement_units, 0)


class TestStratifiedAgreement(unittest.TestCase):
    def test_groups_by_provenance_axis_not_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus(corpus_dir, {
                "gold": [
                    _label(thread_id="rfi-001", annotator="gold"),
                    _label(thread_id="rfi-002", annotator="gold"),
                ],
                "annotator_b": [
                    _label(thread_id="rfi-001", annotator="annotator_b"),
                    _label(thread_id="rfi-002", annotator="annotator_b"),
                ],
            })
            _write_provenance(corpus_dir, "rfi-001", rfi_type="design_clarification", adversarial="none")
            _write_provenance(corpus_dir, "rfi-002", rfi_type="design_clarification", adversarial="boundary_urgency")
            aligned = aa.align_annotators(corpus_dir)
            strata = aa.stratified_agreement(aligned, corpus_dir, "adversarial", fields_to_score=("rfi_type",))
            values = {s.value: s.n_units for s in strata}
            self.assertEqual(values, {"none": 1, "boundary_urgency": 1})


class TestReportWriters(unittest.TestCase):
    def test_json_and_markdown_writers_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp) / "corpus"
            out_dir = Path(tmp) / "out"
            _write_corpus(corpus_dir, {
                "gold": [_label(thread_id="rfi-001", annotator="gold")],
                "annotator_b": [_label(thread_id="rfi-001", annotator="annotator_b", urgency="urgent")],
            })
            _write_provenance(corpus_dir, "rfi-001", rfi_type="design_clarification", adversarial="none")

            report = aa.build_report(corpus_dir)
            self.assertEqual(report.framing, "inter_annotator")

            out_dir.mkdir()
            json_path = out_dir / "report.json"
            md_path = out_dir / "report.md"
            worklist_path = out_dir / "worklist.md"
            aa.write_json_report(report, json_path)
            aa.write_markdown_report(report, md_path)
            aa.write_adjudication_worklist(report.disagreements, worklist_path)

            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["framing"], "inter_annotator")
            self.assertIn("rfi_type", loaded["field_results"])

            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("Annotator Agreement Report", md_text)

            worklist_text = worklist_path.read_text(encoding="utf-8")
            self.assertIn("Adjudication Worklist", worklist_text)


if __name__ == "__main__":
    unittest.main()
