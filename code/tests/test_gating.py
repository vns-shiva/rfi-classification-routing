"""Unit tests for gating.py's confidence gate. Run with:

    python -m unittest discover -s code/tests -t code
"""
from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gating import REQUIRED_NONEMPTY_FIELDS, gate
from schema import RFILabel


def _valid_label(**overrides) -> RFILabel:
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
    )
    base.update(overrides)
    return RFILabel(**base)


class TestGate(unittest.TestCase):
    def test_clean_high_confidence_label_is_not_flagged(self):
        labels = [_valid_label()]
        _, flagged = gate(labels)
        self.assertEqual(flagged, [])

    def test_low_confidence_label_is_flagged_with_reason(self):
        labels = [_valid_label(confidence=0.3)]
        _, flagged = gate(labels)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0].thread_id, "rfi-001")
        self.assertTrue(any("confidence" in r for r in flagged[0].reasons))

    def test_confidence_exactly_at_threshold_is_not_flagged(self):
        labels = [_valid_label(confidence=0.6)]
        _, flagged = gate(labels, confidence_threshold=0.6)
        self.assertEqual(flagged, [])

    def test_confidence_just_below_threshold_is_flagged(self):
        labels = [_valid_label(confidence=0.5999)]
        _, flagged = gate(labels, confidence_threshold=0.6)
        self.assertEqual(len(flagged), 1)

    def test_confidence_above_one_is_flagged_not_treated_as_maximally_confident(self):
        # A model answering "confidence: 85" meaning 85% must not be read as
        # more trustworthy than a well-calibrated 0.85 would be.
        labels = [_valid_label(confidence=85.0)]
        _, flagged = gate(labels)
        self.assertEqual(len(flagged), 1)
        self.assertTrue(any("confidence" in r and "out of" in r for r in flagged[0].reasons))

    def test_negative_confidence_is_flagged(self):
        labels = [_valid_label(confidence=-1.0)]
        _, flagged = gate(labels)
        self.assertEqual(len(flagged), 1)
        self.assertTrue(any("out of" in r for r in flagged[0].reasons))

    def test_nan_confidence_is_flagged(self):
        # NaN compares False against every threshold, so a naive `< threshold`
        # check can never catch it; must be rejected explicitly.
        labels = [_valid_label(confidence=float("nan"))]
        _, flagged = gate(labels)
        self.assertEqual(len(flagged), 1)
        self.assertTrue(any("out of" in r for r in flagged[0].reasons))

    def test_missing_question_summary_is_flagged(self):
        labels = [_valid_label(question_summary="")]
        _, flagged = gate(labels)
        self.assertEqual(len(flagged), 1)
        self.assertTrue(any("question_summary" in r for r in flagged[0].reasons))

    def test_whitespace_only_question_summary_counts_as_missing(self):
        # question_summary is the one REQUIRED_NONEMPTY_FIELDS entry
        # RFILabel.__post_init__ does not strip, so this must be normalized
        # inside gate() itself, not relied on from schema.py.
        labels = [_valid_label(question_summary="   \n")]
        _, flagged = gate(labels)
        self.assertEqual(len(flagged), 1)
        self.assertTrue(any("question_summary" in r for r in flagged[0].reasons))

    def test_padded_dash_question_summary_counts_as_missing(self):
        labels = [_valid_label(question_summary=" — ")]
        _, flagged = gate(labels)
        self.assertEqual(len(flagged), 1)
        self.assertTrue(any("question_summary" in r for r in flagged[0].reasons))

    def test_ascii_hyphen_assigned_reviewer_counts_as_missing(self):
        labels = [_valid_label(assigned_reviewer="-")]
        _, flagged = gate(labels)
        self.assertEqual(len(flagged), 1)
        self.assertTrue(any("assigned_reviewer" in r for r in flagged[0].reasons))

    def test_placeholder_dash_assigned_reviewer_counts_as_missing(self):
        labels = [_valid_label(assigned_reviewer="—")]
        _, flagged = gate(labels)
        self.assertEqual(len(flagged), 1)
        self.assertTrue(any("assigned_reviewer" in r for r in flagged[0].reasons))

    def test_na_variant_rfi_type_counts_as_missing(self):
        for placeholder in ("N/A", "None", "TBD", "null", "--"):
            with self.subTest(placeholder=placeholder):
                labels = [_valid_label(rfi_type=placeholder)]
                _, flagged = gate(labels)
                self.assertEqual(len(flagged), 1)
                self.assertTrue(any("rfi_type" in r for r in flagged[0].reasons))

    def test_en_dash_question_summary_counts_as_missing(self):
        labels = [_valid_label(question_summary="–")]
        _, flagged = gate(labels)
        self.assertEqual(len(flagged), 1)
        self.assertTrue(any("question_summary" in r for r in flagged[0].reasons))

    def test_escalated_label_is_flagged_even_at_full_confidence(self):
        labels = [_valid_label(escalation=True, confidence=1.0)]
        _, flagged = gate(labels)
        self.assertEqual(len(flagged), 1)
        self.assertTrue(any("escalation" in r for r in flagged[0].reasons))

    def test_multiple_reasons_all_recorded_for_one_label(self):
        labels = [_valid_label(confidence=0.1, rfi_type="", escalation=True)]
        _, flagged = gate(labels)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(len(flagged[0].reasons), 3)

    def test_gate_never_mutates_or_drops_labels(self):
        # gate()'s contract (its own docstring) is "never deletes an item".
        # Comparing `returned == labels` is not sufficient: gate() happens to
        # return the same list object it was given, so that assertion is
        # true even if the implementation deleted or overwrote entries in
        # place. Snapshot deep copies up front and check length + every
        # field survives untouched.
        labels = [_valid_label(confidence=0.1), _valid_label(thread_id="rfi-002")]
        snapshot = copy.deepcopy(labels)
        returned, flagged = gate(labels)
        self.assertEqual(len(returned), len(snapshot))
        for original, after in zip(snapshot, returned):
            self.assertEqual(original, after)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0].thread_id, "rfi-001")

    def test_custom_confidence_threshold_is_respected(self):
        labels = [_valid_label(confidence=0.7)]
        _, flagged = gate(labels, confidence_threshold=0.9)
        self.assertEqual(len(flagged), 1)

    def test_empty_label_list_returns_no_flags(self):
        returned, flagged = gate([])
        self.assertEqual(returned, [])
        self.assertEqual(flagged, [])

    def test_flagged_has_one_entry_per_flagged_label_in_input_order(self):
        labels = [
            _valid_label(thread_id="rfi-001", confidence=0.1),
            _valid_label(thread_id="rfi-002"),
            _valid_label(thread_id="rfi-003", escalation=True),
        ]
        _, flagged = gate(labels)
        self.assertEqual([f.thread_id for f in flagged], ["rfi-001", "rfi-003"])

    def test_flagged_carries_annotator_so_same_thread_id_labels_are_distinguishable(self):
        # Two labels for the same thread_id (e.g. gold vs. a second
        # annotator) must not collide if a consumer keys flagged items by
        # thread_id.
        labels = [
            _valid_label(thread_id="rfi-001", annotator="gold", confidence=0.1),
            _valid_label(thread_id="rfi-001", annotator="annotator_b", confidence=0.1),
        ]
        _, flagged = gate(labels)
        self.assertEqual(len(flagged), 2)
        self.assertEqual({f.annotator for f in flagged}, {"gold", "annotator_b"})

    def test_required_nonempty_fields_are_real_rfilabel_fields(self):
        # A schema.py rename of any of these should fail this fast, readable
        # assertion instead of an AttributeError mid-pipeline.
        for field_name in REQUIRED_NONEMPTY_FIELDS:
            self.assertTrue(hasattr(_valid_label(), field_name))


if __name__ == "__main__":
    unittest.main()
