"""Confidence gating for RFI classification/routing labels. Never deletes an
item — a missed classification is worse than a low-confidence one a human can
triage — but separates a `review_queue` of labels that are low-confidence,
missing a required field, or carry an escalation flag, so downstream
consumers (routing_eval.py, policy_audit.py, and scoring) can choose whether
to include gated items. Mirrors the sibling meeting-action-extraction
paper's gating.py; see that module's docstring for the same rationale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from schema import RFILabel

# escalation is already the RFI-domain "high stakes" signal: routing_policy's
# route_rfi() sets it when urgency is urgent/critical or when cost_impact and
# schedule_impact are both true — exactly the RFIs where a wrong or
# low-confidence routing decision is most costly to leave ungated.
REQUIRED_NONEMPTY_FIELDS = ("question_summary", "rfi_type", "assigned_reviewer")

# Only rfi_type and assigned_reviewer are whitespace-stripped by
# RFILabel.__post_init__ (schema.py); question_summary is not, and none of
# the three has its case folded there. Compared case-folded and stripped so
# an LLM's "N/A"/"none"/"tbd"/an en dash, or incidental whitespace, is
# treated the same as the canonical "—" placeholder instead of silently
# escaping the gate.
_PLACEHOLDER_VALUES = frozenset({"", "—", "–", "-", "--", "n/a", "na", "none", "null", "tbd"})


@dataclass
class GateReason:
    thread_id: str
    annotator: str
    reasons: list[str]


def gate(labels: list[RFILabel], confidence_threshold: float = 0.6) -> tuple[list[RFILabel], list[GateReason]]:
    flagged: list[GateReason] = []
    for label in labels:
        reasons = []
        if not math.isfinite(label.confidence) or not (0.0 <= label.confidence <= 1.0):
            reasons.append(f"confidence {label.confidence} out of [0, 1]")
        elif label.confidence < confidence_threshold:
            reasons.append(f"confidence {label.confidence:.4g} < {confidence_threshold:.4g}")
        for field_name in REQUIRED_NONEMPTY_FIELDS:
            value = getattr(label, field_name)
            normalized = value.strip().casefold() if isinstance(value, str) else value
            if not normalized or normalized in _PLACEHOLDER_VALUES:
                reasons.append(f"missing required field: {field_name}")
        if label.escalation:
            reasons.append("escalation flagged by routing policy")
        if reasons:
            flagged.append(GateReason(thread_id=label.thread_id, annotator=label.annotator, reasons=reasons))
    return labels, flagged
