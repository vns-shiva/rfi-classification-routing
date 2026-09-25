"""Renders stratification.build_plan()'s CorpusPlan into concrete RFIThread /
RFILabel pairs.

Two anti-fabrication contracts drive the design here, both enforced by
construction rather than by post-hoc filtering:

  - RFILabel.deadline_text must be a span that actually appears in the
    rendered thread. A cell's deadline_family selects which deadline_phrase
    text *would* apply, but that phrase is only used as deadline_text when
    the chosen initial_question body_variant literally contains the
    {deadline_phrase} placeholder. When it doesn't (a known, tracked
    diversity gap -- see templates.eligible_body_variants), deadline_text is
    honestly "none stated" rather than a fabricated span. This mirrors
    templates.py's stated rationale for eligible_body_variants() exactly.
    For resolvable families, deadline_text is further narrowed by
    _extract_deadline_span() to the short resolvable sub-span within that
    sentence (e.g. "within five days" out of a longer sentence), not the
    whole templated sentence -- still always a genuine substring of it.

  - RFILabel.referenced_documents only lists a document if its rendered name
    literally appears somewhere in the thread text, never just because
    {doc_a}/{doc_b} were drawn for the substitution dict.

Determinism: every random draw for a thread comes from a named substream of
entry.seed (already a blake2b-derived per-thread seed computed once by
stratification.build_plan() -- this module must never recompute it, only
consume it, per PlanEntry's contract).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from hashlib import blake2b
from pathlib import Path
from random import Random
from typing import Any

from deadlines import resolve_deadline
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
from stratification import Cell, CorpusPlan, PlanEntry, build_plan, coverage_report, format_coverage_markdown
from templates import (
    ADVERSARIAL_PHRASE_FIELDS,
    AXIS_NAMES,
    BOOL_PHRASE_KEYS,
    DEADLINE_FAMILIES,
    SHAPE_REQUIRED_MESSAGE_TYPES,
    UNRESOLVABLE_FAMILIES,
    SharedPools,
    Template,
    eligible_body_variants,
    gold_row_for_cell,
    load_shared_pools,
    load_templates,
    validate_all,
)

BASE_SUBMIT_DATE = date(2025, 3, 3)
DATE_SPREAD_DAYS = 240

# deadlines.resolve_deadline() only recognizes short canonical phrases
# ("within 5 days", "this week", "by end of day", ...), not the full
# natural-language sentences templates store in deadline_phrasing. These
# patterns pull the resolvable span out of that sentence so deadline_text
# stays both a genuine verbatim substring (anti-fabrication contract) and
# something resolve_deadline can actually parse. Families with no
# resolvable date concept (none/event_anchored/ambiguous) have no entry --
# their phrase text is left as-is and correctly resolves to "unresolved".
_DEADLINE_SPAN_PATTERNS: dict[str, re.Pattern[str]] = {
    # (?:[\s-]+\w+){0,3} bounds the gap between "within" and the unit word to
    # at most 3 short tokens (e.g. "within five business days"), so a phrase
    # that never reaches a days/weeks unit within that span fails closed
    # (returns None) instead of a stray unbounded .*? swallowing an entire
    # unrelated clause up to some later "day"/"week" word.
    "relative_days": re.compile(r"\bwithin\b(?:[\s-]+\w+){0,3}[\s-]+\b(?:days?|weeks?)\b", re.IGNORECASE),
    "relative_weeks": re.compile(r"\bwithin\b(?:[\s-]+\w+){0,3}[\s-]+\b(?:days?|weeks?)\b", re.IGNORECASE),
    "business_days": re.compile(r"\bwithin\b(?:[\s-]+\w+){0,3}[\s-]+\b(?:days?|weeks?)\b", re.IGNORECASE),
    "weekday": re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.IGNORECASE),
    "this_next_week": re.compile(r"\b(?:this week|next week)\b", re.IGNORECASE),
    "end_of": re.compile(r"\bby (?:the )?end of (?:the )?(?:day|week)\b", re.IGNORECASE),
}

# Fails loudly at import time (rather than silently degrading to the
# pre-fix "always unresolved" bug for one family) if a resolvable family
# ever loses -- or a typo drops -- its extraction pattern.
_RESOLVABLE_FAMILIES = set(DEADLINE_FAMILIES) - set(UNRESOLVABLE_FAMILIES)
assert set(_DEADLINE_SPAN_PATTERNS) == _RESOLVABLE_FAMILIES, (
    f"_DEADLINE_SPAN_PATTERNS must cover exactly the resolvable deadline "
    f"families; got {set(_DEADLINE_SPAN_PATTERNS)}, expected {_RESOLVABLE_FAMILIES}"
)


def _extract_deadline_span(family: str, phrase_text: str) -> str | None:
    pattern = _DEADLINE_SPAN_PATTERNS.get(family)
    if pattern is None:
        return None
    match = pattern.search(phrase_text)
    return match.group(0) if match else None

__all__ = [
    "Provenance", "GeneratedRecord",
    "generate_one", "generate_corpus",
    "validate_record", "write_corpus", "read_corpus", "corpus_stats",
    "main",
]


def _substream(seed: int, name: str) -> Random:
    digest = blake2b(f"{seed}:{name}".encode("utf-8"), digest_size=8).digest()
    return Random(int.from_bytes(digest, "big"))


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Provenance / result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Provenance:
    thread_id: str
    template_id: str
    cell: dict
    variant_index: int
    seed: int
    body_variant_index: dict[str, int]
    question_summary_variant_index: int
    proposed_solution_variant_index: int | None
    project: str
    doc_a: str
    doc_b: str
    date_submitted: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Provenance":
        return Provenance(**d)


@dataclass(frozen=True)
class GeneratedRecord:
    thread: RFIThread
    label: RFILabel
    provenance: Provenance


# ---------------------------------------------------------------------------
# Per-thread value resolution
# ---------------------------------------------------------------------------

def _resolve_slots(template: Template, pools: SharedPools, seed: int) -> dict[str, str]:
    values: dict[str, str] = {}
    for slot_name, options in template.slots.items():
        values[slot_name] = _substream(seed, f"slot:{slot_name}").choice(list(options))
    for slot_name, options in pools.slot_pools.items():
        if slot_name in values:
            continue
        values[slot_name] = _substream(seed, f"slot:{slot_name}").choice(list(options))
    return values


def _resolve_docs(template: Template, cell: Cell, pools: SharedPools, seed: int) -> tuple[str, str]:
    pool = list(pools.doc_refs.get(cell.primary_discipline, ()))
    if len(pool) < 2:
        return "", ""
    rng = _substream(seed, "docs")
    a, b = rng.sample(pool, 2)
    return a, b


def _resolve_phrases(template: Template, cell: Cell, seed: int) -> dict[str, str]:
    def pick(pool_name: str, options) -> str:
        options = list(options)
        if not options:
            return ""
        return _substream(seed, pool_name).choice(options)

    phrases = {
        "urgency_phrase": pick("urgency_phrase", template.urgency_phrasing.get(cell.urgency, ())),
        "cost_phrase": pick("cost_phrase", template.cost_phrasing.get(BOOL_PHRASE_KEYS[cell.cost_impact], ())),
        "schedule_phrase": pick(
            "schedule_phrase", template.schedule_phrasing.get(BOOL_PHRASE_KEYS[cell.schedule_impact], ())
        ),
        "deadline_phrase": pick("deadline_phrase", template.deadline_phrasing.get(cell.deadline_family, ())),
        "decoy_phrase": "",
        "boundary_phrase": "",
        "ambiguity_phrase": "",
        "unresolvable_deadline_phrase": "",
    }
    field = ADVERSARIAL_PHRASE_FIELDS.get(cell.adversarial)
    if field is not None:
        pool_attr, phrase_name = field
        phrases[phrase_name] = pick(phrase_name, getattr(template, pool_attr))
    return phrases


def _select_initial_question_variant(template: Template, cell: Cell, variant_index: int) -> int:
    desired: list[str] = []
    if cell.deadline_family != "none":
        desired.append("deadline_phrase")
    desired.append("urgency_phrase")
    field = ADVERSARIAL_PHRASE_FIELDS.get(cell.adversarial)
    if field is not None:
        pool_attr, phrase_name = field
        if getattr(template, pool_attr):
            desired.append(phrase_name)

    while desired:
        eligible = eligible_body_variants(template, "initial_question", tuple(desired))
        if eligible:
            return eligible[variant_index % len(eligible)]
        desired.pop(0)
    eligible = eligible_body_variants(template, "initial_question", ())
    return eligible[variant_index % len(eligible)]


def _select_variant(template: Template, message_type: str, variant_index: int) -> int:
    eligible = eligible_body_variants(template, message_type, ())
    return eligible[variant_index % len(eligible)]


def _message_for_type(template: Template, message_type: str):
    return next(m for m in template.messages if m.message_type == message_type)


def _submission_date(seed: int) -> date:
    offset = _substream(seed, "date_submitted").randrange(0, DATE_SPREAD_DAYS)
    return BASE_SUBMIT_DATE + timedelta(days=offset)


def _timestamp_for(d: date, seed: int, name: str) -> str:
    rng = _substream(seed, name)
    hour = rng.randrange(7, 18)
    minute = rng.randrange(0, 60)
    return f"{d.isoformat()}T{hour:02d}:{minute:02d}"


def _rfi_number(thread_id: str) -> str:
    return f"RFI-{thread_id.split('-')[-1]}"


def _derive_subject(question_summary: str, max_len: int = 90) -> str:
    text = question_summary.strip()
    match = re.search(r"[.?!]", text)
    first = text[: match.start()] if match else text
    first = first.strip().rstrip(".?!,;: ")
    if len(first) > max_len:
        first = first[:max_len].rsplit(" ", 1)[0].rstrip(".,;: ")
    return first or "RFI"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_one(entry: PlanEntry, templates: dict[str, Template], pools: SharedPools) -> GeneratedRecord:
    template = templates[entry.template_id]
    cell = entry.cell
    seed = entry.seed

    slot_values = _resolve_slots(template, pools, seed)
    doc_a, doc_b = _resolve_docs(template, cell, pools, seed)
    project = _substream(seed, "project").choice(list(pools.projects))
    phrases = _resolve_phrases(template, cell, seed)
    date_submitted = _submission_date(seed)
    rfi_number = _rfi_number(entry.thread_id)

    values: dict[str, Any] = {}
    values.update(slot_values)
    values.update(phrases)
    values["doc_a"] = doc_a
    values["doc_b"] = doc_b
    values["project"] = project["project"]
    values["project_short"] = project["project_short"]
    values["rfi_number"] = rfi_number
    values["date_submitted"] = date_submitted.isoformat()
    values["submitter"] = ""
    values["responder"] = ""

    message_types = SHAPE_REQUIRED_MESSAGE_TYPES[cell.thread_shape]
    body_variant_index: dict[str, int] = {}
    rendered_messages: list[RFIMessage] = []
    submitted_by_role = ""
    current_date = date_submitted

    for i, message_type in enumerate(message_types):
        msg = _message_for_type(template, message_type)
        if message_type == "initial_question":
            idx = _select_initial_question_variant(template, cell, entry.variant_index)
        else:
            idx = _select_variant(template, message_type, entry.variant_index)
        body_variant_index[message_type] = idx
        raw_body = msg.body_variants[idx]

        role = _substream(seed, f"sender_role:{message_type}").choice(list(msg.sender_role_pool))
        name_options = pools.people.get(role, ())
        if not name_options:
            raise ValueError(f"{template.template_id}: no people.json entries for role {role!r}")
        name = _substream(seed, f"sender_name:{message_type}").choice(list(name_options))

        if message_type == "initial_question":
            submitted_by_role = role
            values["submitter"] = name
            timestamp = _timestamp_for(current_date, seed, f"timestamp:{message_type}")
        else:
            if message_type == "response":
                values["responder"] = name
            current_date = current_date + timedelta(days=_substream(seed, f"gap:{message_type}").randint(1, 5))
            timestamp = _timestamp_for(current_date, seed, f"timestamp:{message_type}")

        text = _collapse(raw_body.format(**values))
        rendered_messages.append(
            RFIMessage(
                message_id=i + 1,
                sender_role=role,
                sender_name=name,
                timestamp=timestamp,
                message_type=message_type,
                text=text,
            )
        )

    gold = template.gold
    question_summary_variants = gold["question_summary_variants"]
    qs_idx = entry.variant_index % len(question_summary_variants)
    question_summary = _collapse(question_summary_variants[qs_idx].format(**values))

    ps_idx: int | None = None
    if "proposed_solution_variants" in gold:
        proposed_solution_variants = gold["proposed_solution_variants"]
        ps_idx = entry.variant_index % len(proposed_solution_variants)
        proposed_solution = _collapse(proposed_solution_variants[ps_idx].format(**values))
    elif "proposed_solution" in gold:
        proposed_solution = _collapse(str(gold["proposed_solution"]).format(**values))
    else:
        proposed_solution = "—"

    subject = _derive_subject(question_summary)

    thread = RFIThread(
        thread_id=entry.thread_id,
        project=values["project"],
        rfi_number=rfi_number,
        date_submitted=date_submitted.isoformat(),
        submitted_by_role=submitted_by_role,
        subject=subject,
        messages=rendered_messages,
    )
    full_text = thread.render()

    raw_initial_question = _message_for_type(template, "initial_question").body_variants[
        body_variant_index["initial_question"]
    ]
    deadline_phrase = values.get("deadline_phrase", "")
    if "{deadline_phrase}" in raw_initial_question and deadline_phrase and deadline_phrase in full_text:
        span = _extract_deadline_span(cell.deadline_family, deadline_phrase)
        deadline_text = span if span is not None else deadline_phrase.strip()
    else:
        deadline_text = "none stated"
    deadline_iso, deadline_resolution_status = resolve_deadline(deadline_text, date_submitted.isoformat())

    referenced_documents = [d for d in (doc_a, doc_b) if d and d in full_text]

    row = gold_row_for_cell(template, cell)
    if row is None:
        raise RuntimeError(
            f"{entry.thread_id}: no gold_routing_table row matches cell for template {template.template_id!r}"
        )

    label = RFILabel(
        thread_id=entry.thread_id,
        rfi_type=cell.rfi_type,
        primary_discipline=cell.primary_discipline,
        secondary_disciplines=list(template.secondary_by_adversarial.get(cell.adversarial, ())),
        csi_division=template.csi_by_discipline[cell.primary_discipline],
        urgency=cell.urgency,
        question_summary=question_summary,
        referenced_documents=referenced_documents,
        proposed_solution=proposed_solution,
        cost_impact=cell.cost_impact,
        schedule_impact=cell.schedule_impact,
        answer_in_documents=bool(gold.get("answer_in_documents", False)),
        deadline_text=deadline_text,
        assigned_reviewer=row.assigned_reviewer,
        routing_rationale=row.routing_rationale,
        escalation=row.escalation,
        confidence=1.0,
        status="open",
        annotator="gold",
        deadline_iso=deadline_iso,
        deadline_resolution_status=deadline_resolution_status,
    )

    provenance = Provenance(
        thread_id=entry.thread_id,
        template_id=entry.template_id,
        cell=cell.to_dict(),
        variant_index=entry.variant_index,
        seed=entry.seed,
        body_variant_index=body_variant_index,
        question_summary_variant_index=qs_idx,
        proposed_solution_variant_index=ps_idx,
        project=values["project"],
        doc_a=doc_a,
        doc_b=doc_b,
        date_submitted=date_submitted.isoformat(),
    )

    return GeneratedRecord(thread=thread, label=label, provenance=provenance)


def generate_corpus(
    plan: CorpusPlan, templates: dict[str, Template], pools: SharedPools
) -> list[GeneratedRecord]:
    return [generate_one(entry, templates, pools) for entry in plan.entries]


# ---------------------------------------------------------------------------
# Validation / IO
# ---------------------------------------------------------------------------

def validate_record(record: GeneratedRecord) -> list[str]:
    errors = validate_label(record.label)
    if record.thread.thread_id != record.label.thread_id:
        errors.append(
            f"thread_id mismatch: thread={record.thread.thread_id!r} label={record.label.thread_id!r}"
        )
    if record.provenance.thread_id != record.thread.thread_id:
        errors.append(
            f"thread_id mismatch: thread={record.thread.thread_id!r} provenance={record.provenance.thread_id!r}"
        )
    full_text = record.thread.render()
    for doc in record.label.referenced_documents:
        if doc not in full_text:
            errors.append(f"referenced_documents entry {doc!r} does not appear in rendered thread text")
    if record.label.deadline_text not in ("none stated",) and record.label.deadline_text not in full_text:
        errors.append(f"deadline_text {record.label.deadline_text!r} does not appear in rendered thread text")
    return errors


def write_corpus(records: list[GeneratedRecord], out_dir: Path) -> None:
    threads_dir = out_dir / "threads"
    labels_dir = out_dir / "labels"
    provenance_dir = out_dir / "provenance"
    for d in (threads_dir, labels_dir, provenance_dir):
        d.mkdir(parents=True, exist_ok=True)
    for record in records:
        write_thread(record.thread, threads_dir / f"{record.thread.thread_id}.json")
        write_label(record.label, labels_dir / f"{record.label.thread_id}.json")
        (provenance_dir / f"{record.provenance.thread_id}.json").write_text(
            json.dumps(record.provenance.to_dict(), indent=2), encoding="utf-8"
        )


def read_corpus(
    out_dir: Path, thread_ids: list[str] | None = None, labels_subdir: str = "labels"
) -> list[GeneratedRecord]:
    threads_dir = out_dir / "threads"
    labels_dir = out_dir / labels_subdir
    provenance_dir = out_dir / "provenance"
    if thread_ids is None:
        thread_ids = sorted(p.stem for p in threads_dir.glob("*.json"))
    records = []
    for tid in thread_ids:
        thread = read_thread(threads_dir / f"{tid}.json")
        label = read_label(labels_dir / f"{tid}.json")
        provenance = Provenance.from_dict(json.loads((provenance_dir / f"{tid}.json").read_text(encoding="utf-8")))
        records.append(GeneratedRecord(thread=thread, label=label, provenance=provenance))
    return records


def corpus_stats(records: list[GeneratedRecord]) -> dict:
    total = len(records)
    axis_counts: dict[str, dict[str, int]] = {axis: {} for axis in AXIS_NAMES}
    template_counts: dict[str, int] = {}
    deadline_status_counts: dict[str, int] = {}
    non_none_deadline = 0
    non_none_deadline_resolved_text = 0

    for r in records:
        cell = r.provenance.cell
        for axis in AXIS_NAMES:
            key = str(cell[axis])
            axis_counts[axis][key] = axis_counts[axis].get(key, 0) + 1
        template_counts[r.provenance.template_id] = template_counts.get(r.provenance.template_id, 0) + 1
        deadline_status_counts[r.label.deadline_resolution_status] = (
            deadline_status_counts.get(r.label.deadline_resolution_status, 0) + 1
        )
        if cell["deadline_family"] != "none":
            non_none_deadline += 1
            if r.label.deadline_text != "none stated":
                non_none_deadline_resolved_text += 1

    return {
        "total": total,
        "axis_counts": axis_counts,
        "template_counts": dict(sorted(template_counts.items())),
        "deadline_resolution_status_counts": deadline_status_counts,
        "deadline_phrase_evidence_rate": (
            non_none_deadline_resolved_text / non_none_deadline if non_none_deadline else None
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic RFI corpus from templates + stratification plan.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "pilot-data" / "corpus")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    templates = load_templates()
    pools = load_shared_pools()

    schema_errors = validate_all(templates, pools)
    if schema_errors:
        print("template validation errors -- aborting:", file=sys.stderr)
        for e in schema_errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    kwargs = {"strict": args.strict}
    if args.seed is not None:
        kwargs["seed"] = args.seed
    plan = build_plan(templates, **kwargs)
    if plan.shortfalls:
        print("build_plan shortfalls:", file=sys.stderr)
        for s in plan.shortfalls:
            print(f"  - {s}", file=sys.stderr)

    records = generate_corpus(plan, templates, pools)

    record_errors: list[str] = []
    for r in records:
        for e in validate_record(r):
            record_errors.append(f"{r.provenance.thread_id}: {e}")
    if record_errors:
        print("record validation errors -- aborting without writing:", file=sys.stderr)
        for e in record_errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    write_corpus(records, args.out)

    print(format_coverage_markdown(coverage_report(plan, templates)))
    stats = corpus_stats(records)
    print(f"\ngenerated {stats['total']} threads -> {args.out}")
    print(f"deadline_phrase_evidence_rate: {stats['deadline_phrase_evidence_rate']}")


if __name__ == "__main__":
    main()
