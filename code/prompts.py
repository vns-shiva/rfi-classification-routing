"""Prompt assembly for llm_client.py's ChatLLMClient. PROMPT_VERSION is
bumped and recorded in every prediction file's provenance banner (see
run_pipeline.py) whenever any prompt text below changes, so a re-run can be
told apart from a like-for-like repeat.

Three conditions (llm_client.CONDITIONS), one system-prompt each:
  - "bare":   schema description only.
  - "vocab":  + the closed vocabularies (RFI_TYPES, DISCIPLINES, CSI map,
              URGENCY_TIERS, REVIEWER_ROLES).
  - "policy": + a plain-language rendering of routing_policy.py's decision
              table, so routing_rationale can cite the same rule the
              Composed/Oracle scoring arms use — without ever letting the
              model see routing_policy.py's code (only its documented
              rules), and without changing the gold-label process, which
              stays independent of this policy (see routing_policy.py's
              module docstring).
"""
from __future__ import annotations

from llm_client import CONDITIONS
from schema import RFIThread
from vocabulary import (
    DISCIPLINE_TO_CSI,
    DISCIPLINES,
    REVIEWER_ROLES,
    RFI_TYPES,
    URGENCY_TIERS,
)

PROMPT_VERSION = "v1"
REPAIR_PROMPT_VERSION = "v1"

_LABEL_FIELDS_BLOCK = """\
Return a single JSON object (no prose, no markdown fences) with exactly these fields:

{
  "rfi_type": string,
  "primary_discipline": string,
  "secondary_disciplines": [string, ...],
  "csi_division": string,
  "urgency": string,
  "question_summary": string,
  "referenced_documents": [string, ...],
  "proposed_solution": string,
  "cost_impact": boolean,
  "schedule_impact": boolean,
  "answer_in_documents": boolean,
  "deadline_text": string,
  "assigned_reviewer": string,
  "routing_rationale": string,
  "escalation": boolean,
  "confidence": number
}

Field notes:
- secondary_disciplines: other disciplines materially involved, excluding primary_discipline. Empty list if none.
- csi_division: the CSI MasterFormat 2016 group-level division the RFI's subject matter falls under, or the
  literal string "—" if primary_discipline has no associated CSI division.
- proposed_solution: the submitter's proposed answer if they gave one, else the literal string "—".
- answer_in_documents: true only if the referenced documents already contain the answer to the question asked
  (the submitter should not have needed to ask).
- deadline_text: the verbatim phrase describing when a response is needed, or "none stated" if none is given.
- routing_rationale: one sentence explaining the assigned_reviewer choice.
- confidence: your confidence in this classification as a whole, in [0, 1].
"""


def build_vocabulary_block() -> str:
    csi_lines = "\n".join(
        f"  - {discipline}: {', '.join(sorted(divisions)) if divisions else '(no CSI division; use \"—\")'}"
        for discipline, divisions in DISCIPLINE_TO_CSI.items()
    )
    return f"""\
Use only these closed vocabularies:

rfi_type (exactly one): {', '.join(sorted(RFI_TYPES))}

primary_discipline / secondary_disciplines (from this set): {', '.join(sorted(DISCIPLINES))}

csi_division must be consistent with primary_discipline, from:
{csi_lines}

urgency (exactly one, low to high severity): {', '.join(URGENCY_TIERS)}

assigned_reviewer (exactly one): {', '.join(REVIEWER_ROLES)}
"""


def build_policy_block() -> str:
    return """\
When choosing assigned_reviewer, apply this routing policy (adapted from AIA A201 \
§3.2.4 / ConsensusDocs 200 §12.2 ball-in-court practice):

1. If rfi_type is "code_compliance_question": route to "Code Official/AHJ Liaison",
   regardless of discipline.
2. Else if rfi_type is "substitution_request" and cost_impact is true: route to
   "Cost/Change-Order Manager".
3. Else if rfi_type is "field_condition_conflict" and schedule_impact is true and
   urgency is "urgent" or "critical": route to "GC Superintendent".
4. Otherwise: route to the design professional of record for primary_discipline
   (Architect of Record for Architectural or General; the matching discipline
   engineer otherwise; Plumbing/Fire Protection Engineer for both Plumbing and
   Fire_Protection).

Set escalation to true if urgency is "urgent" or "critical", or if both cost_impact
and schedule_impact are true.
"""


def build_classification_system_prompt(condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r} (expected one of {CONDITIONS})")
    parts = [
        "You are an experienced construction project engineer classifying and routing "
        "a Request for Information (RFI) thread for a commercial construction project.",
        _LABEL_FIELDS_BLOCK,
    ]
    if condition in ("vocab", "policy"):
        parts.append(build_vocabulary_block())
    if condition == "policy":
        parts.append(build_policy_block())
    return "\n".join(parts)


def build_classification_prompt(thread: RFIThread) -> str:
    return f"""\
Project: {thread.project}
RFI number: {thread.rfi_number}

{thread.render()}

Classify and route this RFI thread. Respond with the JSON object only.
"""


def build_repair_instruction(previous_payload: str, errors: str, max_payload_chars: int = 1200) -> str:
    """Appended to the *user* prompt (never the system prompt) for
    classify.py's one bounded repair retry after a
    llm_client.ClassificationParseError. Deliberately changes the user-prompt
    text rather than resending the identical prompt: ChatLLMClient's response
    cache is keyed on (system, user), and a malformed-but-non-empty payload
    IS cached (see _cached_complete's empty-payload-only skip), so an
    unmodified retry would just replay the same cached failure forever.
    previous_payload is truncated so a wildly long malformed response can't
    blow up retry token cost."""
    truncated = previous_payload[:max_payload_chars]
    if len(previous_payload) > max_payload_chars:
        truncated += "... [truncated]"
    return f"""

Your previous response could not be parsed into the required JSON schema.

Previous response:
{truncated}

Problem: {errors}

Return ONLY a corrected, single JSON object matching the schema above — no prose, no markdown fences.
"""
