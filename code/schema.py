"""RFI thread + label schema. Plays the same role vocabulary.py's sibling
schema.py played for the prior paper (shared by generation, annotation,
scoring, and the pipeline so there is one definition instead of several
drifting apart) but stores records as JSON rather than a markdown pipe-table:
RFILabel's extraction block has list- and bool-typed fields (referenced
documents, cost_impact, schedule_impact, answer_in_documents) that don't fit
a flat table cell without inventing an escaping convention JSON already
solves.

RFILabel is deliberately one flat dataclass covering all three label
blocks (classification / extraction / routing) rather than three nested
objects — every consumer (annotator_agreement.py, classify.py, routing_eval.py)
needs fields from more than one block at once, and a flat record with a
field-order convention documented here is easier to keep in sync across
those consumers than three dataclasses plus a container.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from vocabulary import (
    DISCIPLINES,
    REVIEWER_ROLES,
    RFI_TYPES,
    URGENCY_TIERS,
    csi_divisions_for_discipline,
)

# deadlines.py's resolve_deadline() only ever produces these two statuses,
# always paired with the matching deadline_iso shape (see validate_label's
# cross-field check below) — anything else means a hand-written or
# LLM-produced label diverged from that contract.
_DEADLINE_RESOLUTION_STATUSES = {"resolved_to_date", "unresolved"}


@dataclass
class RFIMessage:
    message_id: int
    sender_role: str
    sender_name: str
    timestamp: str  # ISO date, e.g. "2025-03-14"
    message_type: str  # initial_question | response | clarification | resubmittal
    text: str


@dataclass
class RFIThread:
    thread_id: str
    project: str
    rfi_number: str
    date_submitted: str  # ISO date
    submitted_by_role: str
    subject: str
    messages: list[RFIMessage]

    def render(self) -> str:
        lines = [f"Subject: {self.subject}", f"Submitted by: {self.submitted_by_role} on {self.date_submitted}", ""]
        for m in self.messages:
            lines.append(f"[M{m.message_id:02d}] ({m.message_type}) {m.sender_role} ({m.sender_name}), {m.timestamp}:")
            lines.append(m.text)
            lines.append("")
        return "\n".join(lines)


@dataclass
class RFILabel:
    thread_id: str

    # --- classification block ---
    rfi_type: str
    primary_discipline: str
    secondary_disciplines: list[str]
    csi_division: str  # "—" for General/no-CSI-applicable RFIs
    urgency: str

    # --- extraction block ---
    question_summary: str
    referenced_documents: list[str]
    proposed_solution: str  # "—" if the submitter proposed none
    cost_impact: bool
    schedule_impact: bool
    answer_in_documents: bool  # true if referenced_documents already answer the question
    deadline_text: str  # verbatim phrase, "none stated" if absent

    # --- routing block ---
    assigned_reviewer: str
    routing_rationale: str
    escalation: bool

    # --- metadata (not part of any label block; pipeline/annotation bookkeeping) ---
    confidence: float = 0.5
    status: str = "open"
    annotator: str = ""  # "" as produced by llm_client; run_pipeline.py stamps "pipeline" (see --annotator-id); "gold"/"annotator_b" for hand labels

    deadline_iso: str | None = field(default=None)
    deadline_resolution_status: str = "unresolved"

    def __post_init__(self) -> None:
        # Normalizes incidental whitespace (a trailing newline is the common
        # LLM-output case) on every field validate_label treats as a closed
        # vocabulary, at construction time, so the *stored* value — not just
        # a locally-stripped copy inside validate_label — is what
        # routing_policy.route_from_fields() and any gold-vs-predicted
        # exact-match comparison actually see. Without this, a
        # validate_label-approved but still-padded value can miss every
        # DISCIPLINE_REVIEWER key (silent misroute) or blow up
        # urgency_rank() (KeyError/ValueError) downstream. Case is
        # deliberately left alone; see validate_label's docstring.
        self.rfi_type = self.rfi_type.strip()
        self.primary_discipline = self.primary_discipline.strip()
        self.secondary_disciplines = [d.strip() for d in self.secondary_disciplines]
        self.csi_division = self.csi_division.strip()
        self.urgency = self.urgency.strip()
        self.assigned_reviewer = self.assigned_reviewer.strip()
        self.deadline_resolution_status = self.deadline_resolution_status.strip()


def validate_label(label: RFILabel) -> list[str]:
    """Returns a list of validation error strings; empty list == valid.
    Never raises — generate_rfis.py and classify.py both need to record a
    still-invalid label (with its errors attached) rather than crash, since
    an LLM's raw output failing validation is itself a result worth keeping
    (see gating.py's REQUIRED_NONEMPTY_FIELDS precedent)."""
    errors: list[str] = []

    # RFILabel.__post_init__ already strips incidental whitespace (a
    # trailing newline is the common LLM-output case) from every field
    # checked below, so these names are just aliases onto the stored,
    # already-normalized values. Case is NOT folded — these vocabularies
    # mix case meaningfully (e.g. "Fire_Protection" vs.
    # "design_clarification"), so a case mismatch is a real error, not noise.
    rfi_type = label.rfi_type
    primary_discipline = label.primary_discipline
    secondary_disciplines = label.secondary_disciplines
    urgency = label.urgency
    csi_division = label.csi_division
    assigned_reviewer = label.assigned_reviewer

    if rfi_type not in RFI_TYPES:
        errors.append(f"rfi_type {label.rfi_type!r} not in RFI_TYPES")

    if primary_discipline not in DISCIPLINES:
        errors.append(f"primary_discipline {label.primary_discipline!r} not in DISCIPLINES")

    for original, d in zip(label.secondary_disciplines, secondary_disciplines):
        if d not in DISCIPLINES:
            errors.append(f"secondary_disciplines contains {original!r}, not in DISCIPLINES")
    if primary_discipline in secondary_disciplines:
        errors.append("primary_discipline must not also appear in secondary_disciplines")

    if urgency not in URGENCY_TIERS:
        errors.append(f"urgency {label.urgency!r} not in URGENCY_TIERS")

    if primary_discipline == "General":
        if csi_division != "—":
            errors.append("primary_discipline is 'General' but csi_division is not '—'")
    elif csi_division == "—":
        errors.append("csi_division is '—' but primary_discipline is not 'General'")
    else:
        valid_csi = csi_divisions_for_discipline(primary_discipline)
        if valid_csi and csi_division not in valid_csi:
            errors.append(
                f"csi_division {label.csi_division!r} not among CSI divisions mapped to "
                f"primary_discipline {label.primary_discipline!r}: {sorted(valid_csi)}"
            )

    if assigned_reviewer not in REVIEWER_ROLES:
        errors.append(f"assigned_reviewer {label.assigned_reviewer!r} not in REVIEWER_ROLES")

    if not (0.0 <= label.confidence <= 1.0):
        errors.append(f"confidence {label.confidence} out of [0,1]")

    if not label.question_summary.strip():
        errors.append("question_summary is empty")

    if not label.deadline_text.strip():
        errors.append("deadline_text is empty (use 'none stated')")

    # deadlines.py's resolve_deadline() always pairs "resolved_to_date" with
    # a concrete ISO date and "unresolved" with None — a label whose two
    # fields disagree with that contract (e.g. hand-edited, or produced by
    # code that doesn't go through resolve_deadline()) should fail loudly
    # rather than silently carry a nonsensical deadline into scoring.
    if label.deadline_resolution_status not in _DEADLINE_RESOLUTION_STATUSES:
        errors.append(
            f"deadline_resolution_status {label.deadline_resolution_status!r} "
            f"not in {sorted(_DEADLINE_RESOLUTION_STATUSES)}"
        )
    elif label.deadline_resolution_status == "resolved_to_date":
        deadline_iso = (label.deadline_iso or "").strip()
        if not deadline_iso:
            errors.append("deadline_resolution_status is 'resolved_to_date' but deadline_iso is empty")
        else:
            try:
                date.fromisoformat(deadline_iso)
            except ValueError:
                errors.append(f"deadline_iso {label.deadline_iso!r} is not a valid ISO date")
    elif label.deadline_resolution_status == "unresolved" and label.deadline_iso is not None:
        errors.append(f"deadline_resolution_status is 'unresolved' but deadline_iso is {label.deadline_iso!r}")

    return errors


def thread_to_dict(thread: RFIThread) -> dict:
    return asdict(thread)


def thread_from_dict(data: dict) -> RFIThread:
    messages = [RFIMessage(**m) for m in data["messages"]]
    return RFIThread(
        thread_id=data["thread_id"],
        project=data["project"],
        rfi_number=data["rfi_number"],
        date_submitted=data["date_submitted"],
        submitted_by_role=data["submitted_by_role"],
        subject=data["subject"],
        messages=messages,
    )


def write_thread(thread: RFIThread, path: Path) -> None:
    path.write_text(json.dumps(thread_to_dict(thread), indent=2), encoding="utf-8")


def read_thread(path: Path) -> RFIThread:
    return thread_from_dict(json.loads(path.read_text(encoding="utf-8")))


def label_to_dict(label: RFILabel) -> dict:
    return asdict(label)


def label_from_dict(data: dict) -> RFILabel:
    return RFILabel(**data)


def write_label(label: RFILabel, path: Path) -> None:
    path.write_text(json.dumps(label_to_dict(label), indent=2), encoding="utf-8")


def read_label(path: Path) -> RFILabel:
    return label_from_dict(json.loads(path.read_text(encoding="utf-8")))
