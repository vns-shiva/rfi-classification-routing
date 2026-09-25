"""Closed vocabularies for the RFI classification/routing label schema
(schema.py's RFILabel). Four independent controlled fields:

- RFI_TYPES: what kind of question the RFI is asking.
- DISCIPLINES: the fixed discipline taxonomy used for both primary_discipline
  (single) and secondary_disciplines (multi-label) in the classification
  block, plus a many-to-many mapping to CSI MasterFormat 2016 group-level
  divisions (a discipline typically spans several divisions, and a handful
  of divisions are legitimately shared by two disciplines — e.g. fire
  suppression sits under both Plumbing and Fire_Protection scopes on most
  project teams — so the map is many-to-many in both directions rather than
  a single lookup dict).
- URGENCY_TIERS: an ordinal (not nominal) scale — routing_policy.py and
  agreement_metrics.py's quadratic-weighted-kappa agreement/scoring code both
  depend on this list's order being the true severity ordering, not just a
  set of labels.
- REVIEWER_ROLES: the fixed 10-role taxonomy routing_policy.py routes an RFI
  to. Fixed rather than free-text so routing is a closed-set classification
  problem instead of open-ended entity resolution.
"""
from __future__ import annotations

RFI_TYPES = {
    "design_clarification",
    "document_discrepancy",
    "field_condition_conflict",
    "substitution_request",
    "coordination_conflict",
    "code_compliance_question",
}

DISCIPLINES = {
    "Architectural",
    "Structural",
    "Mechanical",
    "Electrical",
    "Plumbing",
    "Fire_Protection",
    "Civil",
    "General",
}

# Ordered low -> high severity. Index position IS the ordinal value used by
# quadratic-weighted Cohen's kappa (agreement_metrics.py) and
# routing_policy.py's escalation rule — do not reorder without re-deriving
# every metric that depends on this ordering.
URGENCY_TIERS = ["routine", "priority", "urgent", "critical"]

REVIEWER_ROLES = [
    "Architect of Record",
    "Structural Engineer",
    "Mechanical Engineer",
    "Electrical Engineer",
    "Plumbing/Fire Protection Engineer",
    "Civil Engineer",
    "GC Superintendent",
    "Owner's Representative",
    "Cost/Change-Order Manager",
    "Code Official/AHJ Liaison",
]

# CSI MasterFormat 2016 group-level divisions actually exercised by the
# synthetic corpus (generate_rfis.py's scope note: extending pilot coverage
# means adding entries here, not rearchitecting — same convention as the
# prior paper's ROLE_TO_CSI).
DISCIPLINE_TO_CSI: dict[str, set[str]] = {
    "Structural": {"03 00 00 (Concrete)", "04 00 00 (Masonry)", "05 00 00 (Metals)", "31 00 00 (Earthwork)"},
    "Architectural": {"06 00 00 (Wood, Plastics, Composites)", "07 00 00 (Thermal and Moisture Protection)",
                      "08 00 00 (Openings)", "09 00 00 (Finishes)"},
    "Mechanical": {"23 00 00 (HVAC)"},
    "Electrical": {"26 00 00 (Electrical)", "27 00 00 (Communications)", "28 00 00 (Electronic Safety and Security)"},
    "Plumbing": {"21 00 00 (Fire Suppression)", "22 00 00 (Plumbing)"},
    "Fire_Protection": {"21 00 00 (Fire Suppression)"},
    "Civil": {"31 00 00 (Earthwork)", "32 00 00 (Exterior Improvements)", "33 00 00 (Utilities)"},
    "General": set(),
}


def csi_divisions_for_discipline(discipline: str) -> set[str]:
    return set(DISCIPLINE_TO_CSI.get(discipline, set()))


def disciplines_for_csi(csi_division: str) -> set[str]:
    return {d for d, divisions in DISCIPLINE_TO_CSI.items() if csi_division in divisions}


def urgency_rank(tier: str) -> int:
    """Ordinal position for QWK / routing escalation. Raises on an
    out-of-vocabulary tier rather than silently defaulting — a typo'd tier
    should fail loudly, not get miscoded as 'routine'."""
    return URGENCY_TIERS.index(tier)
