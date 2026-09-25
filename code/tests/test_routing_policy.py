"""Unit tests for routing_policy.py's decision table. Run with:

    python -m unittest discover -s code/tests -t code
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass

from routing_policy import RULES, escalation_for, reachable_reviewers, route_from_fields, route_rfi
from schema import RFILabel
from vocabulary import DISCIPLINES, REVIEWER_ROLES, RFI_TYPES, URGENCY_TIERS, urgency_rank


@dataclass(frozen=True)
class _DecisionV0:
    assigned_reviewer: str
    routing_rationale: str
    escalation: bool


def _route_rfi_v0(rfi_type, primary_discipline, cost_impact, schedule_impact, urgency):
    """Frozen verbatim copy of route_rfi()'s pre-refactor body (the decision
    table before RULES/rule_id were extracted). Used only to prove the
    refactor is behavior-preserving across the full input grid."""
    rank = urgency_rank(urgency)
    escalation = rank >= urgency_rank("urgent") or (cost_impact and schedule_impact)

    if rfi_type == "code_compliance_question":
        return _DecisionV0(
            assigned_reviewer="Code Official/AHJ Liaison",
            routing_rationale=(
                "Code/compliance interpretation sits outside the design team's contractual "
                "authority; routed to the AHJ liaison regardless of discipline."
            ),
            escalation=escalation,
        )

    if rfi_type == "substitution_request" and cost_impact:
        return _DecisionV0(
            assigned_reviewer="Cost/Change-Order Manager",
            routing_rationale=(
                "Substitution request with a stated cost impact is a change-order-track item "
                "(ConsensusDocs 200 §12.2); the cost representative leads, with the discipline "
                "engineer/architect as technical reviewer of record."
            ),
            escalation=escalation,
        )

    if rfi_type == "field_condition_conflict" and schedule_impact and rank >= urgency_rank("urgent"):
        return _DecisionV0(
            assigned_reviewer="GC Superintendent",
            routing_rationale=(
                "Urgent, schedule-impacting field condition: ball-in-court shifts to the "
                "superintendent to direct interim means/methods pending the design team's answer "
                "(AIA A201 §3.2.4 does not suspend the Contractor's duty to proceed safely)."
            ),
            escalation=escalation,
        )

    from routing_policy import DISCIPLINE_REVIEWER

    reviewer = DISCIPLINE_REVIEWER.get(primary_discipline, "Architect of Record")
    return _DecisionV0(
        assigned_reviewer=reviewer,
        routing_rationale=(
            f"Default discipline routing under AIA A201 §3.2.4: the design professional of "
            f"record for {primary_discipline} interprets and clarifies the Contract Documents."
        ),
        escalation=escalation,
    )


class TestRouteRFI(unittest.TestCase):
    def test_code_compliance_always_routes_to_ahj_regardless_of_discipline(self):
        decision = route_rfi(
            rfi_type="code_compliance_question", primary_discipline="Structural",
            cost_impact=False, schedule_impact=False, urgency="routine",
        )
        self.assertEqual(decision.assigned_reviewer, "Code Official/AHJ Liaison")

    def test_substitution_with_cost_impact_routes_to_cost_manager(self):
        decision = route_rfi(
            rfi_type="substitution_request", primary_discipline="Mechanical",
            cost_impact=True, schedule_impact=False, urgency="routine",
        )
        self.assertEqual(decision.assigned_reviewer, "Cost/Change-Order Manager")

    def test_substitution_without_cost_impact_routes_to_discipline_engineer(self):
        decision = route_rfi(
            rfi_type="substitution_request", primary_discipline="Mechanical",
            cost_impact=False, schedule_impact=False, urgency="routine",
        )
        self.assertEqual(decision.assigned_reviewer, "Mechanical Engineer")

    def test_urgent_schedule_impacting_field_condition_routes_to_superintendent(self):
        decision = route_rfi(
            rfi_type="field_condition_conflict", primary_discipline="Structural",
            cost_impact=False, schedule_impact=True, urgency="urgent",
        )
        self.assertEqual(decision.assigned_reviewer, "GC Superintendent")

    def test_critical_schedule_impacting_field_condition_routes_to_superintendent(self):
        # rule 3's guard is `rank >= urgency_rank("urgent")`, not an exact
        # match on "urgent" — "critical" (one rank higher) must also qualify.
        decision = route_rfi(
            rfi_type="field_condition_conflict", primary_discipline="Structural",
            cost_impact=False, schedule_impact=True, urgency="critical",
        )
        self.assertEqual(decision.assigned_reviewer, "GC Superintendent")

    def test_routine_field_condition_routes_to_discipline_engineer(self):
        decision = route_rfi(
            rfi_type="field_condition_conflict", primary_discipline="Structural",
            cost_impact=False, schedule_impact=True, urgency="routine",
        )
        self.assertEqual(decision.assigned_reviewer, "Structural Engineer")

    def test_priority_schedule_impacting_field_condition_does_not_escalate_to_superintendent(self):
        # Collapse case: schedule_impact alone is not sufficient for rule 3 —
        # "priority" sits one rank below "urgent" and must fall through to
        # the default discipline-engineer routing, not the superintendent.
        decision = route_rfi(
            rfi_type="field_condition_conflict", primary_discipline="Structural",
            cost_impact=False, schedule_impact=True, urgency="priority",
        )
        self.assertEqual(decision.assigned_reviewer, "Structural Engineer")

    def test_urgent_field_condition_without_schedule_impact_does_not_escalate_to_superintendent(self):
        # Collapse case: urgency alone is not sufficient for rule 3 either —
        # schedule_impact must also be true.
        decision = route_rfi(
            rfi_type="field_condition_conflict", primary_discipline="Structural",
            cost_impact=False, schedule_impact=False, urgency="urgent",
        )
        self.assertEqual(decision.assigned_reviewer, "Structural Engineer")

    def test_default_routes_by_discipline(self):
        decision = route_rfi(
            rfi_type="design_clarification", primary_discipline="Electrical",
            cost_impact=False, schedule_impact=False, urgency="routine",
        )
        self.assertEqual(decision.assigned_reviewer, "Electrical Engineer")

    def test_general_discipline_defaults_to_architect(self):
        decision = route_rfi(
            rfi_type="design_clarification", primary_discipline="General",
            cost_impact=False, schedule_impact=False, urgency="routine",
        )
        self.assertEqual(decision.assigned_reviewer, "Architect of Record")

    def test_escalation_true_for_urgent_or_critical(self):
        for urgency in ("urgent", "critical"):
            decision = route_rfi(
                rfi_type="design_clarification", primary_discipline="Electrical",
                cost_impact=False, schedule_impact=False, urgency=urgency,
            )
            self.assertTrue(decision.escalation, urgency)

    def test_escalation_true_for_combined_cost_and_schedule_impact(self):
        decision = route_rfi(
            rfi_type="design_clarification", primary_discipline="Electrical",
            cost_impact=True, schedule_impact=True, urgency="routine",
        )
        self.assertTrue(decision.escalation)

    def test_escalation_false_for_routine_single_impact(self):
        decision = route_rfi(
            rfi_type="design_clarification", primary_discipline="Electrical",
            cost_impact=True, schedule_impact=False, urgency="routine",
        )
        self.assertFalse(decision.escalation)

    def test_unknown_urgency_raises(self):
        with self.assertRaises(ValueError):
            route_rfi(
                rfi_type="design_clarification", primary_discipline="Electrical",
                cost_impact=False, schedule_impact=False, urgency="whenever",
            )


class TestRouteRFIRefactorEquivalence(unittest.TestCase):
    def test_route_rfi_matches_frozen_v0_on_full_768_cell_grid(self):
        for rfi_type in sorted(RFI_TYPES):
            for discipline in sorted(DISCIPLINES):
                for cost_impact in (False, True):
                    for schedule_impact in (False, True):
                        for urgency in URGENCY_TIERS:
                            expected = _route_rfi_v0(rfi_type, discipline, cost_impact, schedule_impact, urgency)
                            actual = route_rfi(rfi_type, discipline, cost_impact, schedule_impact, urgency)
                            self.assertEqual(actual.assigned_reviewer, expected.assigned_reviewer)
                            self.assertEqual(actual.routing_rationale, expected.routing_rationale)
                            self.assertEqual(actual.escalation, expected.escalation)


class TestRuleAttribution(unittest.TestCase):
    def test_every_decision_carries_a_known_rule_id(self):
        known = {rule.rule_id for rule in RULES}
        for rfi_type in sorted(RFI_TYPES):
            decision = route_rfi(rfi_type, "Structural", False, False, "routine")
            self.assertIn(decision.rule_id, known)

    def test_code_compliance_fires_rule_1(self):
        decision = route_rfi("code_compliance_question", "Structural", False, False, "routine")
        self.assertEqual(decision.rule_id, "R1_code_compliance")

    def test_substitution_with_cost_fires_rule_2(self):
        decision = route_rfi("substitution_request", "Mechanical", True, False, "routine")
        self.assertEqual(decision.rule_id, "R2_substitution_cost")

    def test_urgent_field_condition_fires_rule_3(self):
        decision = route_rfi("field_condition_conflict", "Structural", False, True, "urgent")
        self.assertEqual(decision.rule_id, "R3_field_condition_schedule_urgent")

    def test_default_fires_rule_4(self):
        decision = route_rfi("design_clarification", "Electrical", False, False, "routine")
        self.assertEqual(decision.rule_id, "R4_discipline_default")


class TestEscalationFor(unittest.TestCase):
    def test_matches_route_rfi_escalation_on_full_grid(self):
        for rfi_type in sorted(RFI_TYPES):
            for discipline in sorted(DISCIPLINES):
                for cost_impact in (False, True):
                    for schedule_impact in (False, True):
                        for urgency in URGENCY_TIERS:
                            decision = route_rfi(rfi_type, discipline, cost_impact, schedule_impact, urgency)
                            self.assertEqual(
                                escalation_for(cost_impact, schedule_impact, urgency), decision.escalation
                            )


class TestReachableReviewers(unittest.TestCase):
    def test_owners_representative_is_unreachable(self):
        reachable = reachable_reviewers()
        self.assertNotIn("Owner's Representative", reachable)

    def test_nine_of_ten_roles_are_reachable(self):
        reachable = reachable_reviewers()
        self.assertEqual(len(reachable), 9)
        self.assertTrue(reachable.issubset(set(REVIEWER_ROLES)))


class TestRouteFromFields(unittest.TestCase):
    def test_accepts_an_rfilabel(self):
        label = RFILabel(
            thread_id="rfi-001", rfi_type="code_compliance_question",
            primary_discipline="Fire_Protection", secondary_disciplines=[],
            csi_division="21 00 00 (Fire Suppression)", urgency="priority",
            question_summary="Is the standpipe spacing compliant?",
            referenced_documents=[], proposed_solution="—",
            cost_impact=False, schedule_impact=False, answer_in_documents=False,
            deadline_text="within one week", assigned_reviewer="Fire Marshal",
            routing_rationale="placeholder", escalation=False,
        )
        decision = route_from_fields(label)
        self.assertEqual(decision.assigned_reviewer, "Code Official/AHJ Liaison")


if __name__ == "__main__":
    unittest.main()
