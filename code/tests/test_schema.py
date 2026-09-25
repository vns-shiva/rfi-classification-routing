"""Unit tests for schema.py: RFIThread/RFILabel JSON round-trip and
validate_label()'s error surface. Run with:

    python -m unittest discover -s code/tests -t code
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schema import (
    RFILabel,
    RFIMessage,
    RFIThread,
    read_label,
    read_thread,
    validate_label,
    write_label,
    write_thread,
)


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


class TestRFIThreadRoundTrip(unittest.TestCase):
    def test_json_round_trip(self):
        thread = RFIThread(
            thread_id="rfi-001",
            project="Riverside Medical Office Building",
            rfi_number="RFI-014",
            date_submitted="2025-03-14",
            submitted_by_role="GC Superintendent",
            subject="Beam-to-column connection at gridline C4",
            messages=[
                RFIMessage(1, "GC Superintendent", "Marcus Webb", "2025-03-14", "initial_question",
                           "Structural drawing S-401 shows a moment connection at C4 that conflicts with S-501's shear tab detail. Please clarify."),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rfi-001.json"
            write_thread(thread, path)
            loaded = read_thread(path)
        self.assertEqual(loaded.thread_id, thread.thread_id)
        self.assertEqual(len(loaded.messages), 1)
        self.assertEqual(loaded.messages[0].sender_name, "Marcus Webb")


class TestRFILabelRoundTrip(unittest.TestCase):
    def test_json_round_trip(self):
        label = _valid_label()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rfi-001-label.json"
            write_label(label, path)
            loaded = read_label(path)
        self.assertEqual(loaded, label)


class TestValidateLabel(unittest.TestCase):
    def test_valid_label_has_no_errors(self):
        self.assertEqual(validate_label(_valid_label()), [])

    def test_bad_rfi_type(self):
        errors = validate_label(_valid_label(rfi_type="not_a_real_type"))
        self.assertTrue(any("rfi_type" in e for e in errors))

    def test_secondary_discipline_must_not_repeat_primary(self):
        errors = validate_label(_valid_label(secondary_disciplines=["Structural"]))
        self.assertTrue(any("secondary_disciplines" in e for e in errors))

    def test_csi_division_must_match_discipline(self):
        errors = validate_label(_valid_label(csi_division="26 00 00 (Electrical)"))
        self.assertTrue(any("csi_division" in e for e in errors))

    def test_general_discipline_requires_dash_csi(self):
        errors = validate_label(_valid_label(primary_discipline="General", csi_division="05 00 00 (Metals)"))
        self.assertTrue(any("csi_division" in e for e in errors))

    def test_general_discipline_with_dash_csi_is_valid(self):
        errors = validate_label(_valid_label(
            primary_discipline="General", csi_division="—",
            assigned_reviewer="GC Superintendent",
        ))
        self.assertEqual(errors, [])

    def test_bad_confidence_range(self):
        errors = validate_label(_valid_label(confidence=1.5))
        self.assertTrue(any("confidence" in e for e in errors))

    def test_empty_question_summary(self):
        errors = validate_label(_valid_label(question_summary="  "))
        self.assertTrue(any("question_summary" in e for e in errors))

    def test_bad_reviewer_role(self):
        errors = validate_label(_valid_label(assigned_reviewer="Random Guy"))
        self.assertTrue(any("assigned_reviewer" in e for e in errors))

    def test_resolved_to_date_requires_a_deadline_iso(self):
        errors = validate_label(_valid_label(
            deadline_resolution_status="resolved_to_date", deadline_iso=None,
        ))
        self.assertTrue(any("deadline_resolution_status" in e for e in errors))

    def test_resolved_to_date_with_a_deadline_iso_is_valid(self):
        errors = validate_label(_valid_label(
            deadline_resolution_status="resolved_to_date", deadline_iso="2025-03-28",
        ))
        self.assertEqual(errors, [])

    def test_unresolved_must_not_carry_a_deadline_iso(self):
        errors = validate_label(_valid_label(
            deadline_resolution_status="unresolved", deadline_iso="2025-03-28",
        ))
        self.assertTrue(any("deadline_resolution_status" in e for e in errors))

    def test_unresolved_with_no_deadline_iso_is_valid(self):
        errors = validate_label(_valid_label(
            deadline_resolution_status="unresolved", deadline_iso=None,
        ))
        self.assertEqual(errors, [])

    def test_unrecognized_deadline_resolution_status(self):
        errors = validate_label(_valid_label(deadline_resolution_status="pending_review"))
        self.assertTrue(any("deadline_resolution_status" in e for e in errors))

    def test_incidental_whitespace_on_a_vocabulary_field_is_tolerated(self):
        # A trailing newline/space is a formatting artifact some LLM output
        # carries, not an out-of-vocabulary value.
        errors = validate_label(_valid_label(rfi_type="design_clarification\n"))
        self.assertEqual(errors, [])

    def test_wrong_case_on_a_vocabulary_field_is_still_rejected(self):
        # Whitespace is tolerated, but case is not folded — these
        # vocabularies use case meaningfully, so a case mismatch is a real
        # out-of-vocabulary value, not incidental formatting.
        errors = validate_label(_valid_label(primary_discipline="structural"))
        self.assertTrue(any("primary_discipline" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
