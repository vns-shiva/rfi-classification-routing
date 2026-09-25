"""Unit tests for prompts.py: condition-gated system-prompt assembly. Run
with:

    python -m unittest discover -s code/tests -t code
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts import (
    build_classification_prompt,
    build_classification_system_prompt,
    build_policy_block,
    build_repair_instruction,
    build_vocabulary_block,
)
from routing_policy import route_rfi
from schema import RFIMessage, RFIThread
from vocabulary import DISCIPLINES, REVIEWER_ROLES, RFI_TYPES

_MARKER_VOCAB = "closed vocabularies"
_MARKER_POLICY = "routing policy"

_RULE_RE = re.compile(r"^(\d+)\.\s(.+?)(?=^\d+\.\s|^Set escalation|\Z)", re.M | re.S)


def _rule_lines(block: str) -> dict[int, str]:
    """Splits build_policy_block()'s numbered decision-table rules into
    {rule_number: rule_text}, so a test can assert against one specific
    rule's actual wording rather than the whole block — a whole-block
    substring check would pass even if the wording under a given number
    drifted onto a different rule's condition."""
    return {int(num): text.strip() for num, text in _RULE_RE.findall(block)}


class TestSystemPromptConditions(unittest.TestCase):
    def test_bare_excludes_vocab_and_policy(self):
        prompt = build_classification_system_prompt("bare")
        self.assertNotIn(_MARKER_VOCAB, prompt)
        self.assertNotIn(_MARKER_POLICY, prompt)
        # the label-fields block itself is always present
        self.assertIn('"rfi_type"', prompt)

    def test_bare_does_not_leak_a_closed_vocabulary_value(self):
        # The csi_division field note used to name the literal discipline
        # value "General" unconditionally, leaking one DISCIPLINES member
        # into the supposedly taxonomy-free "bare" condition.
        prompt = build_classification_system_prompt("bare")
        self.assertNotIn("General", prompt)

    def test_vocab_includes_vocab_but_not_policy(self):
        prompt = build_classification_system_prompt("vocab")
        self.assertIn(_MARKER_VOCAB, prompt)
        self.assertNotIn(_MARKER_POLICY, prompt)

    def test_policy_includes_both(self):
        prompt = build_classification_system_prompt("policy")
        self.assertIn(_MARKER_VOCAB, prompt)
        self.assertIn(_MARKER_POLICY, prompt)

    def test_unknown_condition_raises(self):
        # An unrecognized condition must fail loudly rather than silently
        # degrade to the bare prompt (matching llm_client.py's clients,
        # which already validate condition in their own __init__).
        with self.assertRaises(ValueError):
            build_classification_system_prompt("not_a_condition")


class TestVocabularyBlock(unittest.TestCase):
    def test_contains_every_closed_vocabulary_value(self):
        block = build_vocabulary_block()
        for rfi_type in RFI_TYPES:
            self.assertIn(rfi_type, block)
        for discipline in DISCIPLINES:
            self.assertIn(discipline, block)
        for role in REVIEWER_ROLES:
            self.assertIn(role, block)


class TestPolicyBlock(unittest.TestCase):
    def test_cites_the_source_standards(self):
        block = build_policy_block()
        self.assertIn("AIA A201", block)
        self.assertIn("ConsensusDocs 200", block)

    def test_names_every_special_case_reviewer(self):
        block = build_policy_block()
        self.assertIn("Code Official/AHJ Liaison", block)
        self.assertIn("Cost/Change-Order Manager", block)
        self.assertIn("GC Superintendent", block)


class TestPolicyBlockMatchesRoutingPolicy(unittest.TestCase):
    """build_policy_block() is a hand-written plain-language rendering of
    routing_policy.route_rfi()'s decision table, kept deliberately separate
    from the code so the model never sees routing_policy.py itself. That
    separation means the two can drift silently if route_rfi() ever changes
    without a matching prompt edit — each test below asserts against that
    specific numbered rule's *actual text* (via _rule_lines(), not just a
    whole-block substring check) as well as route_rfi()'s behavior, so a
    wording change under the wrong number, or a code change the text no
    longer describes, both fail here."""

    def setUp(self):
        self.rules = _rule_lines(build_policy_block())

    def test_rule_1_code_compliance_overrides_discipline(self):
        text = self.rules[1]
        self.assertIn("code_compliance_question", text)
        self.assertIn("Code Official/AHJ Liaison", text)
        decision = route_rfi(
            rfi_type="code_compliance_question", primary_discipline="Structural",
            cost_impact=False, schedule_impact=False, urgency="routine",
        )
        self.assertEqual(decision.assigned_reviewer, "Code Official/AHJ Liaison")

    def test_rule_2_substitution_with_cost_impact(self):
        text = self.rules[2]
        self.assertIn("substitution_request", text)
        self.assertIn("cost_impact", text)
        self.assertIn("Cost/Change-Order Manager", text)
        decision = route_rfi(
            rfi_type="substitution_request", primary_discipline="Mechanical",
            cost_impact=True, schedule_impact=False, urgency="routine",
        )
        self.assertEqual(decision.assigned_reviewer, "Cost/Change-Order Manager")

    def test_rule_3_urgent_schedule_impacting_field_condition(self):
        text = self.rules[3]
        self.assertIn("field_condition_conflict", text)
        self.assertIn("schedule_impact", text)
        self.assertIn("urgent", text)
        self.assertIn("critical", text)
        for urgency in ("urgent", "critical"):
            decision = route_rfi(
                rfi_type="field_condition_conflict", primary_discipline="Structural",
                cost_impact=False, schedule_impact=True, urgency=urgency,
            )
            self.assertEqual(decision.assigned_reviewer, "GC Superintendent", urgency)
        # The text's stated threshold is "urgent" or "critical" specifically
        # — "priority" (one tier below) must NOT trigger this rule, or the
        # text and the code have drifted apart on where the line is.
        decision_priority = route_rfi(
            rfi_type="field_condition_conflict", primary_discipline="Structural",
            cost_impact=False, schedule_impact=True, urgency="priority",
        )
        self.assertNotEqual(decision_priority.assigned_reviewer, "GC Superintendent")

    def test_rule_4_default_routes_plumbing_and_fire_protection_to_the_same_role(self):
        # build_policy_block's rule 4 note calls out that Plumbing and
        # Fire_Protection both land on "Plumbing/Fire Protection Engineer".
        text = self.rules[4]
        self.assertIn("Architect of Record", text)
        self.assertIn("Plumbing", text)
        self.assertIn("Fire_Protection", text)
        self.assertIn("Plumbing/Fire Protection Engineer", text)
        for discipline in ("Plumbing", "Fire_Protection"):
            decision = route_rfi(
                rfi_type="design_clarification", primary_discipline=discipline,
                cost_impact=False, schedule_impact=False, urgency="routine",
            )
            self.assertEqual(decision.assigned_reviewer, "Plumbing/Fire Protection Engineer", discipline)

    def test_escalation_rule_matches_block_text(self):
        # "urgency is urgent or critical, or both cost_impact and
        # schedule_impact are true."
        block = build_policy_block()
        self.assertIn("Set escalation to true if urgency is", block)
        self.assertTrue(route_rfi("design_clarification", "Electrical", False, False, "urgent").escalation)
        self.assertTrue(route_rfi("design_clarification", "Electrical", False, False, "critical").escalation)
        self.assertTrue(route_rfi("design_clarification", "Electrical", True, True, "routine").escalation)
        self.assertFalse(route_rfi("design_clarification", "Electrical", True, False, "routine").escalation)


class TestClassificationPrompt(unittest.TestCase):
    def test_includes_project_header_and_rendered_thread(self):
        thread = RFIThread(
            thread_id="rfi-001",
            project="Riverside Medical Office Building",
            rfi_number="RFI-014",
            date_submitted="2025-03-14",
            submitted_by_role="GC Superintendent",
            subject="Test subject",
            messages=[
                RFIMessage(1, "GC Superintendent", "Marcus Webb", "2025-03-14", "initial_question", "Body text."),
            ],
        )
        prompt = build_classification_prompt(thread)
        self.assertIn("Riverside Medical Office Building", prompt)
        self.assertIn("RFI-014", prompt)
        self.assertIn("Body text.", prompt)
        self.assertIn("JSON object only", prompt)


class TestBuildRepairInstruction(unittest.TestCase):
    def test_includes_previous_payload_and_error_detail(self):
        instruction = build_repair_instruction("{bad json", "Expecting value: line 1")
        self.assertIn("{bad json", instruction)
        self.assertIn("Expecting value: line 1", instruction)
        self.assertIn("JSON object", instruction)

    def test_truncates_a_payload_longer_than_max_payload_chars(self):
        payload = "x" * 2000
        instruction = build_repair_instruction(payload, "some error", max_payload_chars=100)
        self.assertIn("x" * 100, instruction)
        self.assertNotIn("x" * 101, instruction)
        self.assertIn("[truncated]", instruction)

    def test_short_payload_is_not_marked_truncated(self):
        instruction = build_repair_instruction("short", "some error", max_payload_chars=100)
        self.assertNotIn("[truncated]", instruction)

    def test_output_is_additive_relative_to_the_base_classification_prompt(self):
        # build_repair_instruction() is meant to be appended to the user
        # prompt so the cache key (system, user) changes on a repair retry —
        # its text must not collide with build_classification_prompt()'s own
        # closing instruction, or a retry could hash to the same cache key
        # as the original request and replay the same cached failure.
        thread = RFIThread(
            thread_id="rfi-001",
            project="Riverside Medical Office Building",
            rfi_number="RFI-014",
            date_submitted="2025-03-14",
            submitted_by_role="GC Superintendent",
            subject="Test subject",
            messages=[
                RFIMessage(1, "GC Superintendent", "Marcus Webb", "2025-03-14", "initial_question", "Body text."),
            ],
        )
        base_prompt = build_classification_prompt(thread)
        repaired_prompt = base_prompt + build_repair_instruction("{bad", "parse error")
        self.assertNotEqual(base_prompt, repaired_prompt)
        self.assertIn("Body text.", repaired_prompt)
        self.assertIn("previous response could not be parsed", repaired_prompt)


if __name__ == "__main__":
    unittest.main()
