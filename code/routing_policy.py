"""Deterministic RFI routing policy: classification fields -> (reviewer,
rationale, escalation).

This is an operationalization, not a legal citation, of the "ball-in-court"
practice standard construction contracts assign for RFI response
responsibility — AIA A201-2017 §3.2.4 (the Architect, or the engineer of
record for that discipline, interprets and clarifies the Contract
Documents) and ConsensusDocs 200 §12.2 (similar allocation, with the
Owner's cost representative pulled in once a request carries a change-order
implication). Real contract administration also weighs project-specific
delegation, the RFI log's prior history, and a human reviewer's judgment
that this table cannot encode — it exists to give the paper (a) a
non-circular way to construct Composed/Oracle routing-evaluation arms from
classification fields instead of an LLM's own routing guess, and (b) a
policy-validity-agreement check during annotation (do two annotators agree
on what the policy *would* route to, independent of what they each think
the "true" answer is).

This module must never be used as the sole source of gold routing labels
(that would make routing evaluation circular — scoring a policy against
itself). Gold `assigned_reviewer` is set by human annotators; this module
only supplies the Composed and Oracle evaluation arms in routing_eval.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from vocabulary import DISCIPLINES, RFI_TYPES, URGENCY_TIERS, urgency_rank

DISCIPLINE_REVIEWER: dict[str, str] = {
    "Structural": "Structural Engineer",
    "Mechanical": "Mechanical Engineer",
    "Electrical": "Electrical Engineer",
    "Plumbing": "Plumbing/Fire Protection Engineer",
    "Fire_Protection": "Plumbing/Fire Protection Engineer",
    "Civil": "Civil Engineer",
    "Architectural": "Architect of Record",
    "General": "Architect of Record",
}

RULE_IDS = (
    "R1_code_compliance",
    "R2_substitution_cost",
    "R3_field_condition_schedule_urgent",
    "R4_discipline_default",
)


@dataclass(frozen=True)
class RoutingDecision:
    assigned_reviewer: str
    routing_rationale: str
    escalation: bool
    rule_id: str = ""


def escalation_for(cost_impact: bool, schedule_impact: bool, urgency: str) -> bool:
    """The one escalation formula, shared by route_rfi() and policy_audit.py
    so the two can never silently diverge. `urgency` must be a value from
    vocabulary.URGENCY_TIERS; an out-of-vocabulary tier fails loudly via
    urgency_rank() rather than silently defaulting to "not escalated"."""
    rank = urgency_rank(urgency)
    return rank >= urgency_rank("urgent") or (cost_impact and schedule_impact)


def _fires_code_compliance(rfi_type, primary_discipline, cost_impact, schedule_impact, rank) -> bool:
    return rfi_type == "code_compliance_question"


def _fires_substitution_cost(rfi_type, primary_discipline, cost_impact, schedule_impact, rank) -> bool:
    return rfi_type == "substitution_request" and cost_impact


def _fires_field_condition_urgent(rfi_type, primary_discipline, cost_impact, schedule_impact, rank) -> bool:
    return rfi_type == "field_condition_conflict" and schedule_impact and rank >= urgency_rank("urgent")


def _fires_discipline_default(rfi_type, primary_discipline, cost_impact, schedule_impact, rank) -> bool:
    return True


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    description: str
    predicate: Callable[[str, str, bool, bool, int], bool]


def _decide_code_compliance(rfi_type, primary_discipline, cost_impact, schedule_impact, rank) -> tuple[str, str]:
    return (
        "Code Official/AHJ Liaison",
        "Code/compliance interpretation sits outside the design team's contractual "
        "authority; routed to the AHJ liaison regardless of discipline.",
    )


def _decide_substitution_cost(rfi_type, primary_discipline, cost_impact, schedule_impact, rank) -> tuple[str, str]:
    return (
        "Cost/Change-Order Manager",
        "Substitution request with a stated cost impact is a change-order-track item "
        "(ConsensusDocs 200 §12.2); the cost representative leads, with the discipline "
        "engineer/architect as technical reviewer of record.",
    )


def _decide_field_condition_urgent(rfi_type, primary_discipline, cost_impact, schedule_impact, rank) -> tuple[str, str]:
    return (
        "GC Superintendent",
        "Urgent, schedule-impacting field condition: ball-in-court shifts to the "
        "superintendent to direct interim means/methods pending the design team's answer "
        "(AIA A201 §3.2.4 does not suspend the Contractor's duty to proceed safely).",
    )


def _decide_discipline_default(rfi_type, primary_discipline, cost_impact, schedule_impact, rank) -> tuple[str, str]:
    reviewer = DISCIPLINE_REVIEWER.get(primary_discipline, "Architect of Record")
    rationale = (
        f"Default discipline routing under AIA A201 §3.2.4: the design professional of "
        f"record for {primary_discipline} interprets and clarifies the Contract Documents."
    )
    return reviewer, rationale


_DECIDERS: dict[str, Callable[[str, str, bool, bool, int], tuple[str, str]]] = {
    "R1_code_compliance": _decide_code_compliance,
    "R2_substitution_cost": _decide_substitution_cost,
    "R3_field_condition_schedule_urgent": _decide_field_condition_urgent,
    "R4_discipline_default": _decide_discipline_default,
}

RULES: tuple[PolicyRule, ...] = (
    PolicyRule("R1_code_compliance", "rfi_type is code_compliance_question", _fires_code_compliance),
    PolicyRule("R2_substitution_cost", "rfi_type is substitution_request and cost_impact is true", _fires_substitution_cost),
    PolicyRule(
        "R3_field_condition_schedule_urgent",
        "rfi_type is field_condition_conflict and schedule_impact is true and urgency is urgent or higher",
        _fires_field_condition_urgent,
    ),
    PolicyRule("R4_discipline_default", "no higher-priority rule matched; route by primary_discipline", _fires_discipline_default),
)


class NoRuleMatchedError(Exception):
    """Raised by route_with_rules() when a caller-supplied rule subset
    matches no rule for the given inputs (e.g. policy_audit.py's
    leave-one-rule-out ablation with R4_discipline_default removed). The
    default RULES tuple always has R4 as a catch-all, so route_rfi() itself
    can never raise this; a failure there would mean RULES was edited to
    drop its catch-all, which is still an AssertionError (an internal
    invariant break, not a caller-supplied-subset scenario)."""


def route_with_rules(
    rfi_type: str,
    primary_discipline: str,
    cost_impact: bool,
    schedule_impact: bool,
    urgency: str,
    rules: tuple[PolicyRule, ...] = RULES,
) -> RoutingDecision:
    """The decision table, parameterized over which rules are in play.
    Evaluated top to bottom; the first matching rule wins. `urgency` must be
    a value from vocabulary.URGENCY_TIERS — an out-of-vocabulary tier fails
    loudly via urgency_rank() rather than silently defaulting to "not
    escalated". Exists so policy_audit.py's rule-contribution ablation
    (scoring the policy with one rule removed) can call this directly
    instead of monkeypatching the module-global RULES."""
    rank = urgency_rank(urgency)
    escalation = escalation_for(cost_impact, schedule_impact, urgency)

    for rule in rules:
        if rule.predicate(rfi_type, primary_discipline, cost_impact, schedule_impact, rank):
            reviewer, rationale = _DECIDERS[rule.rule_id](
                rfi_type, primary_discipline, cost_impact, schedule_impact, rank
            )
            return RoutingDecision(
                assigned_reviewer=reviewer,
                routing_rationale=rationale,
                escalation=escalation,
                rule_id=rule.rule_id,
            )

    if rules is RULES:
        raise AssertionError("no rule matched; R4_discipline_default must be a catch-all")
    raise NoRuleMatchedError(
        f"no rule in {[r.rule_id for r in rules]} matched rfi_type={rfi_type!r}, "
        f"primary_discipline={primary_discipline!r}, cost_impact={cost_impact!r}, "
        f"schedule_impact={schedule_impact!r}, urgency={urgency!r}"
    )


def route_rfi(
    rfi_type: str,
    primary_discipline: str,
    cost_impact: bool,
    schedule_impact: bool,
    urgency: str,
) -> RoutingDecision:
    """The decision table. Evaluated top to bottom (RULES); the first
    matching rule wins. `urgency` must be a value from
    vocabulary.URGENCY_TIERS — an out-of-vocabulary tier fails loudly via
    urgency_rank() rather than silently defaulting to "not escalated"."""
    return route_with_rules(rfi_type, primary_discipline, cost_impact, schedule_impact, urgency, rules=RULES)


def reachable_reviewers() -> set[str]:
    """Enumerate the full 768-cell input grid and return the set of
    assigned_reviewer values route_rfi() can actually produce. Lives here
    (not in policy_audit.py) because routing_eval.py's accuracy-ceiling
    computation needs it too, and the two modules must not cross-import."""
    reviewers: set[str] = set()
    for rfi_type in RFI_TYPES:
        for discipline in DISCIPLINES:
            for cost_impact in (False, True):
                for schedule_impact in (False, True):
                    for urgency in URGENCY_TIERS:
                        decision = route_rfi(rfi_type, discipline, cost_impact, schedule_impact, urgency)
                        reviewers.add(decision.assigned_reviewer)
    return reviewers


def route_from_fields(label) -> RoutingDecision:
    """Convenience wrapper for routing_eval.py's Composed/Oracle arms: takes
    anything with the four classification-block attributes route_rfi()
    needs (an RFILabel, predicted or gold, satisfies this)."""
    return route_rfi(
        rfi_type=label.rfi_type,
        primary_discipline=label.primary_discipline,
        cost_impact=label.cost_impact,
        schedule_impact=label.schedule_impact,
        urgency=label.urgency,
    )
