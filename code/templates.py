"""RFI scenario templates: the only place a human writes RFI prose and gold
routing decisions. Kept in its own module (not inside stratification.py or
generate_rfis.py) because both of those need the Template type without
importing each other — stratification.py checks axis compatibility at
plan-build time, generate_rfis.py renders the prose and resolves gold labels
at generation time, and neither should import the other.

Every Template carries its own gold_routing_table rather than deferring to
routing_policy.route_rfi(): routing_policy.py's module docstring forbids ever
using it as the source of gold routing labels (that would make the "does the
model learn the policy" evaluation circular). Agreement between gold and the
policy is measured after generation, in policy_audit.py.

Deliberate deviation from the plan sketch: the deadline-family, adversarial,
and thread-shape enums conceptually "belong" to stratification.py (they
describe corpus cells, not RFI labels), but validate_template() needs them
and stratification.py already depends on this module for the Template type.
Defining them here (and having stratification.py import and re-export them)
keeps the dependency graph one-directional: templates.py has no dependency on
stratification.py.

Naming/structure conventions carried over from schema.py and routing_policy.py:
flat frozen dataclasses, from __future__ import annotations, validation
functions that return list[str] and never raise (mirrors
schema.validate_label), and build_*/load_* factory functions rather than
classmethods.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vocabulary import DISCIPLINES, REVIEWER_ROLES, RFI_TYPES, URGENCY_TIERS, csi_divisions_for_discipline

if TYPE_CHECKING:
    from stratification import Cell

TEMPLATE_SCHEMA_VERSION = 1
DEV_POOL = "dev"
EVAL_POOL = "eval"
POOLS = (DEV_POOL, EVAL_POOL)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Axes every Template must declare, in the order downstream cross-product
# checks enumerate them.
AXIS_NAMES = (
    "rfi_type",
    "primary_discipline",
    "urgency",
    "cost_impact",
    "schedule_impact",
    "deadline_family",
    "adversarial",
    "thread_shape",
)

# These conceptually belong to stratification.py (they describe corpus
# cells, not RFI labels) but live here to avoid a backwards import -- see
# the module docstring. stratification.py imports and re-exports them.
DEADLINE_FAMILIES = (
    "none", "relative_days", "relative_weeks", "business_days", "weekday",
    "this_next_week", "end_of", "event_anchored", "ambiguous",
)
UNRESOLVABLE_FAMILIES = ("none", "event_anchored", "ambiguous")
ADVERSARIAL_FAMILIES = (
    "none", "multi_discipline", "boundary_urgency", "unresolvable_deadline",
    "answer_in_documents", "distractor_cost_language", "type_ambiguity",
)
THREAD_SHAPES = ("single_message", "two_message_qa", "multi_turn_clarification", "resubmittal")

# Sorted explicitly rather than iterated straight off RFI_TYPES/DISCIPLINES:
# those two vocab entries are backed by Python `set`s, whose iteration order
# is PYTHONHASHSEED-dependent -- anything enumerated from this dict (e.g.
# stratification.py's cross-product over axis values) must see a stable
# order across processes.
AXIS_VOCAB: dict[str, tuple] = {
    "rfi_type": tuple(sorted(RFI_TYPES)),
    "primary_discipline": tuple(sorted(DISCIPLINES)),
    "urgency": URGENCY_TIERS,
    "cost_impact": (False, True),
    "schedule_impact": (False, True),
    "deadline_family": DEADLINE_FAMILIES,
    "adversarial": ADVERSARIAL_FAMILIES,
    "thread_shape": THREAD_SHAPES,
}

ROUTING_ROW_KEYS = ("rfi_type", "cost_impact", "schedule_impact", "urgency")

# Keys used by a template's cost_phrasing/schedule_phrasing dicts -- JSON has
# no bool keys, so these are authored as string literals "true"/"false".
BOOL_PHRASE_KEYS = {False: "false", True: "true"}

MESSAGE_TYPES = ("initial_question", "response", "clarification", "resubmittal")

# Which message_types a given thread_shape's rendering requires at least one
# message of. Mirrors RFIMessage.message_type's docstring in schema.py.
SHAPE_REQUIRED_MESSAGE_TYPES: dict[str, tuple[str, ...]] = {
    "single_message": ("initial_question",),
    "two_message_qa": ("initial_question", "response"),
    "multi_turn_clarification": ("initial_question", "response", "clarification"),
    "resubmittal": ("initial_question", "response", "resubmittal"),
}

# Placeholders substituted by generate_rfis.py itself (project/person/document
# identity, and the phrase blocks assembled from a template's own phrasing
# dicts) rather than looked up in a template's `slots` or the shared pools.
RESERVED_PLACEHOLDERS = frozenset({
    "project", "project_short", "submitter", "responder", "rfi_number",
    "date_submitted", "urgency_phrase", "cost_phrase", "schedule_phrase",
    "deadline_phrase", "decoy_phrase", "boundary_phrase", "ambiguity_phrase",
    "unresolvable_deadline_phrase", "doc_a", "doc_b",
})

# Reserved placeholders generate_rfis.py substitutes with "" when a cell's
# axis value doesn't call for them (e.g. {decoy_phrase} when adversarial !=
# distractor_cost_language, {boundary_phrase} when adversarial != boundary_urgency,
# {ambiguity_phrase} when adversarial != type_ambiguity, {unresolvable_deadline_phrase}
# when adversarial != unresolvable_deadline). doc_a/doc_b/project/... are
# always non-empty and are deliberately excluded -- t01's "in {doc_a}."
# glued to punctuation must stay legal.
#
# generate_rfis.py's substitution contract: replace each of these with either
# real phrase text or "", then collapse runs of whitespace to a single space
# and strip the message body. The whitespace-delimitation check below only
# guarantees that check is *safe* for the immediate neighbors of a single
# placeholder -- it does not catch every orphaned-punctuation shape (e.g.
# "{urgency_phrase} {deadline_phrase} ." collapses to a lone "." and is not
# flagged). Keep body_variants such that dropping any subset of these
# placeholders still reads as a complete sentence.
SENTENCE_PHRASE_PLACEHOLDERS = frozenset({
    "urgency_phrase", "cost_phrase", "schedule_phrase", "deadline_phrase",
    "decoy_phrase", "boundary_phrase", "ambiguity_phrase",
    "unresolvable_deadline_phrase",
})
assert SENTENCE_PHRASE_PLACEHOLDERS <= RESERVED_PLACEHOLDERS, (
    "every sentence-phrase placeholder must also be reserved (both are substituted "
    "by generate_rfis.py itself, never looked up in slots/pools)"
)
_PHRASE_RE = re.compile(r"\{(" + "|".join(sorted(SENTENCE_PHRASE_PLACEHOLDERS)) + r")\}")

_MIN_BODY_VARIANTS = 3

# Adversarial values that need a dedicated phrasing pool + placeholder to stay
# textually distinguishable from a sibling cell that shares every other axis
# value but a different adversarial value (most commonly 'none'). Maps each
# such value to (pool attribute name, placeholder name). Consumed by
# validate_template() below and intended for generate_rfis.py's substitution
# step as well, so the two never drift out of sync the way three hand-written
# near-duplicate checks did.
ADVERSARIAL_PHRASE_FIELDS: dict[str, tuple[str, str]] = {
    "distractor_cost_language": ("decoy_phrasing", "decoy_phrase"),
    "boundary_urgency": ("boundary_phrasing", "boundary_phrase"),
    "type_ambiguity": ("ambiguity_phrasing", "ambiguity_phrase"),
    "unresolvable_deadline": ("unresolvable_deadline_phrasing", "unresolvable_deadline_phrase"),
}


@dataclass(frozen=True)
class GoldRoutingRow:
    when: dict[str, Any]
    assigned_reviewer: str
    routing_rationale: str
    escalation: bool
    policy_divergence: bool = False
    divergence_reason: str = ""

    def matches(self, values: dict[str, Any]) -> bool:
        for key, want in self.when.items():
            if want == "*":
                continue
            if values.get(key) != want:
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "when": self.when,
            "assigned_reviewer": self.assigned_reviewer,
            "routing_rationale": self.routing_rationale,
            "escalation": self.escalation,
            "policy_divergence": self.policy_divergence,
            "divergence_reason": self.divergence_reason,
        }

    @staticmethod
    def from_dict(d: dict) -> "GoldRoutingRow":
        return GoldRoutingRow(
            when=dict(d["when"]),
            assigned_reviewer=d["assigned_reviewer"],
            routing_rationale=d["routing_rationale"],
            escalation=bool(d["escalation"]),
            policy_divergence=bool(d.get("policy_divergence", False)),
            divergence_reason=d.get("divergence_reason", ""),
        )


@dataclass(frozen=True)
class TemplateMessage:
    message_type: str
    sender_role_pool: tuple[str, ...]
    shapes: tuple[str, ...]
    body_variants: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "message_type": self.message_type,
            "sender_role_pool": list(self.sender_role_pool),
            "shapes": list(self.shapes),
            "body_variants": list(self.body_variants),
        }

    @staticmethod
    def from_dict(d: dict) -> "TemplateMessage":
        return TemplateMessage(
            message_type=d["message_type"],
            sender_role_pool=tuple(d["sender_role_pool"]),
            shapes=tuple(d["shapes"]),
            body_variants=tuple(d["body_variants"]),
        )


@dataclass(frozen=True)
class Template:
    template_id: str
    pool: str
    axes: dict[str, tuple]
    csi_by_discipline: dict[str, str]
    secondary_by_adversarial: dict[str, tuple[str, ...]]
    slots: dict[str, tuple[str, ...]]
    urgency_phrasing: dict[str, tuple[str, ...]]
    cost_phrasing: dict[str, tuple[str, ...]]
    schedule_phrasing: dict[str, tuple[str, ...]]
    deadline_phrasing: dict[str, tuple[str, ...]]
    decoy_phrasing: tuple[str, ...]
    boundary_phrasing: tuple[str, ...]
    ambiguity_phrasing: tuple[str, ...]
    unresolvable_deadline_phrasing: tuple[str, ...]
    messages: tuple[TemplateMessage, ...]
    gold: dict
    gold_routing_table: tuple[GoldRoutingRow, ...]
    schema_version: int = TEMPLATE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "pool": self.pool,
            "axes": {k: list(v) for k, v in self.axes.items()},
            "csi_by_discipline": dict(self.csi_by_discipline),
            "secondary_by_adversarial": {k: list(v) for k, v in self.secondary_by_adversarial.items()},
            "slots": {k: list(v) for k, v in self.slots.items()},
            "urgency_phrasing": {k: list(v) for k, v in self.urgency_phrasing.items()},
            "cost_phrasing": {k: list(v) for k, v in self.cost_phrasing.items()},
            "schedule_phrasing": {k: list(v) for k, v in self.schedule_phrasing.items()},
            "deadline_phrasing": {k: list(v) for k, v in self.deadline_phrasing.items()},
            "decoy_phrasing": list(self.decoy_phrasing),
            "boundary_phrasing": list(self.boundary_phrasing),
            "ambiguity_phrasing": list(self.ambiguity_phrasing),
            "unresolvable_deadline_phrasing": list(self.unresolvable_deadline_phrasing),
            "messages": [m.to_dict() for m in self.messages],
            "gold": self.gold,
            "gold_routing_table": [r.to_dict() for r in self.gold_routing_table],
        }

    @staticmethod
    def from_dict(d: dict) -> "Template":
        return Template(
            schema_version=d.get("schema_version", TEMPLATE_SCHEMA_VERSION),
            template_id=d["template_id"],
            pool=d["pool"],
            axes={k: tuple(v) for k, v in d["axes"].items()},
            csi_by_discipline=dict(d["csi_by_discipline"]),
            secondary_by_adversarial={k: tuple(v) for k, v in d.get("secondary_by_adversarial", {}).items()},
            slots={k: tuple(v) for k, v in d.get("slots", {}).items()},
            urgency_phrasing={k: tuple(v) for k, v in d.get("urgency_phrasing", {}).items()},
            cost_phrasing={k: tuple(v) for k, v in d.get("cost_phrasing", {}).items()},
            schedule_phrasing={k: tuple(v) for k, v in d.get("schedule_phrasing", {}).items()},
            deadline_phrasing={k: tuple(v) for k, v in d.get("deadline_phrasing", {}).items()},
            decoy_phrasing=tuple(d.get("decoy_phrasing", ())),
            boundary_phrasing=tuple(d.get("boundary_phrasing", ())),
            ambiguity_phrasing=tuple(d.get("ambiguity_phrasing", ())),
            unresolvable_deadline_phrasing=tuple(d.get("unresolvable_deadline_phrasing", ())),
            messages=tuple(TemplateMessage.from_dict(m) for m in d["messages"]),
            gold=d["gold"],
            gold_routing_table=tuple(GoldRoutingRow.from_dict(r) for r in d["gold_routing_table"]),
        )


# Minimum number of gold text variants (question_summary_variants,
# proposed_solution_variants) a template must author -- mirrors
# _MIN_BODY_VARIANTS's rationale but for gold text: generate_rfis.py rotates
# through these the same way it rotates through body_variants, and a
# degenerate single-entry list would make every rendered thread for a given
# template repeat identical gold text regardless of which cell it fills.
_MIN_GOLD_VARIANTS = 2

# Minimum number of distinct document references a discipline's doc_refs pool
# must carry. generate_rfis.py draws two *different* values from this pool for
# a single rendered thread ({doc_a} and {doc_b} are always used together, one
# contradicting or cross-referencing the other) -- a pool with fewer than 2
# entries can never satisfy that.
_MIN_DOC_REFS = 2


@dataclass(frozen=True)
class SharedPools:
    projects: tuple[dict, ...]
    people: dict[str, tuple[str, ...]]
    slot_pools: dict[str, Any]
    doc_refs: dict[str, tuple[str, ...]]

    def slot_names(self) -> set[str]:
        names = set(self.slot_pools.keys())
        names.update(self.people.keys())
        return names


def _extract_placeholders(text: str) -> set[str]:
    return set(re.findall(r"\{([a-zA-Z0-9_]+)\}", text))


def _initial_question_variants_missing(t: Template, phrase: str) -> list[str]:
    """body_variants of the template's initial_question message(s) that lack
    {phrase}. Every thread shape includes an initial_question message, so a
    variant missing the marker here (unlike response/resubmittal variants,
    which not every shape draws) can render a cell textually identical to its
    differently-labeled sibling regardless of which shape/variant is drawn."""
    marker = "{" + phrase + "}"
    return [
        body
        for msg in t.messages
        if msg.message_type == "initial_question"
        for body in msg.body_variants
        if marker not in body
    ]


def eligible_body_variants(t: Template, message_type: str, required_phrases: tuple[str, ...]) -> list[int]:
    """Indices of `message_type`'s body_variants that contain every phrase in
    `required_phrases`. generate_rfis.py (not yet built) must restrict variant
    selection to these indices for any cell whose gold label depends on that
    phrase being rendered -- e.g. a cell with a non-'none' deadline_family
    needs a variant carrying {deadline_phrase}, because schema.py's
    deadline_text field is a verbatim span from the rendered thread ('none
    stated' only when the family truly is 'none'). Selecting a variant outside
    this set would force generation to either fabricate a deadline_text span
    that isn't in the text, or emit a deadline_text that contradicts the
    declared deadline_family -- there is no non-fabricating fallback.
    validate_all() does not enforce that this set is non-empty for every
    template/message_type/phrase combination that needs it; see the scope
    note above."""
    msg = next((m for m in t.messages if m.message_type == message_type), None)
    if msg is None:
        raise ValueError(f"{t.template_id}: no {message_type!r} message")
    return [
        i for i, body in enumerate(msg.body_variants)
        if all(f"{{{phrase}}}" in body for phrase in required_phrases)
    ]


def _all_template_strings(t: Template) -> list[str]:
    strings: list[str] = (
        list(t.decoy_phrasing)
        + list(t.boundary_phrasing)
        + list(t.ambiguity_phrasing)
        + list(t.unresolvable_deadline_phrasing)
    )
    for d in (t.urgency_phrasing, t.cost_phrasing, t.schedule_phrasing, t.deadline_phrasing):
        for variants in d.values():
            strings.extend(variants)
    for m in t.messages:
        strings.extend(m.body_variants)
    gold = t.gold
    for key in ("question_summary_variants", "proposed_solution_variants"):
        strings.extend(gold.get(key, []) or [])
    if isinstance(gold.get("proposed_solution"), str):
        strings.append(gold["proposed_solution"])
    return strings


def _non_body_variant_strings(t: Template) -> list[tuple[str, str]]:
    """(name, text) pairs for every author-facing string that generate_rfis.py
    never runs the reserved sentence-phrase substitution over -- i.e.
    everything _all_template_strings() covers except message body_variants,
    plus slots (a {placeholder} left inside a slot value would survive
    verbatim into the rendered body instead of being substituted)."""
    pairs: list[tuple[str, str]] = []
    for slot_name, variants in t.slots.items():
        pairs.extend((f"slots[{slot_name!r}]", v) for v in variants)
    for dict_name, d in (
        ("urgency_phrasing", t.urgency_phrasing),
        ("cost_phrasing", t.cost_phrasing),
        ("schedule_phrasing", t.schedule_phrasing),
        ("deadline_phrasing", t.deadline_phrasing),
    ):
        pairs.extend((f"{dict_name}[{k!r}]", v) for k, vs in d.items() for v in vs)
    pairs.extend(("decoy_phrasing", v) for v in t.decoy_phrasing)
    pairs.extend(("boundary_phrasing", v) for v in t.boundary_phrasing)
    pairs.extend(("ambiguity_phrasing", v) for v in t.ambiguity_phrasing)
    pairs.extend(("unresolvable_deadline_phrasing", v) for v in t.unresolvable_deadline_phrasing)
    gold = t.gold
    for key in ("question_summary_variants", "proposed_solution_variants"):
        value = gold.get(key)
        if isinstance(value, list):
            pairs.extend((f"gold.{key}", v) for v in value)
    if isinstance(gold.get("proposed_solution"), str):
        pairs.append(("gold.proposed_solution", gold["proposed_solution"]))
    return pairs


def load_templates(dir: Path = TEMPLATES_DIR) -> dict[str, Template]:
    templates: dict[str, Template] = {}
    for path in sorted(dir.glob("t*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        templates[data["template_id"]] = Template.from_dict(data)
    return templates


def load_shared_pools(dir: Path = TEMPLATES_DIR) -> SharedPools:
    projects = json.loads((dir / "projects.json").read_text(encoding="utf-8"))
    people = json.loads((dir / "people.json").read_text(encoding="utf-8"))
    slot_pools = json.loads((dir / "slot_pools.json").read_text(encoding="utf-8"))
    # doc_refs is discipline-keyed document identity (backs the reserved
    # {doc_a}/{doc_b} placeholders -- see RESERVED_PLACEHOLDERS), not a named
    # slot placeholder itself, so it is split out of slot_pools rather than
    # left as a key inside it: leaving it in would make slot_names() report
    # a spurious {doc_refs} slot that no template ever declares.
    doc_refs = slot_pools.pop("doc_refs", {})
    return SharedPools(
        projects=tuple(projects),
        people={k: tuple(v) for k, v in people.items()},
        slot_pools=slot_pools,
        doc_refs={k: tuple(v) for k, v in doc_refs.items()},
    )


def validate_template(t: Template, pools: SharedPools | None = None) -> list[str]:
    errors: list[str] = []

    if t.schema_version != TEMPLATE_SCHEMA_VERSION:
        errors.append(f"{t.template_id}: schema_version {t.schema_version} != {TEMPLATE_SCHEMA_VERSION}")

    if t.pool not in POOLS:
        errors.append(f"{t.template_id}: pool {t.pool!r} not in {POOLS}")

    for axis in AXIS_NAMES:
        if axis not in t.axes:
            errors.append(f"{t.template_id}: missing axis {axis!r}")
            continue
        vocab = AXIS_VOCAB[axis]
        values = t.axes[axis]
        if not values:
            errors.append(f"{t.template_id}: axis {axis!r} is declared but has no values")
        if len(set(values)) != len(values):
            errors.append(f"{t.template_id}: axis {axis!r} has duplicate values {list(values)!r}")
        for value in values:
            if value not in vocab:
                errors.append(f"{t.template_id}: axis {axis!r} has invalid value {value!r}")

    for discipline in t.axes.get("primary_discipline", ()):
        csi = t.csi_by_discipline.get(discipline)
        if csi is None:
            errors.append(f"{t.template_id}: csi_by_discipline missing entry for {discipline!r}")
            continue
        valid_csi = csi_divisions_for_discipline(discipline)
        if discipline == "General":
            if csi != "—":
                errors.append(f"{t.template_id}: General discipline must map to '—', got {csi!r}")
        elif csi not in valid_csi:
            errors.append(f"{t.template_id}: csi {csi!r} not valid for discipline {discipline!r}")

    disciplines = tuple(t.axes.get("primary_discipline", ()))
    csi_division = t.gold.get("csi_division")
    if csi_division is not None:
        if len(set(disciplines)) > 1:
            errors.append(
                f"{t.template_id}: gold['csi_division'] must be omitted when primary_discipline "
                f"declares {len(set(disciplines))} values {list(disciplines)!r}; resolve "
                f"csi_by_discipline[cell.primary_discipline] instead"
            )
        elif disciplines:
            mapped = t.csi_by_discipline.get(disciplines[0])
            # mapped is None means csi_by_discipline is missing an entry for this
            # discipline, already reported by the loop above -- don't double-report.
            if mapped is not None and csi_division != mapped:
                errors.append(
                    f"{t.template_id}: gold['csi_division'] {csi_division!r} != "
                    f"csi_by_discipline[{disciplines[0]!r}] {mapped!r}"
                )

    for i, row in enumerate(t.gold_routing_table):
        if row.assigned_reviewer not in REVIEWER_ROLES:
            errors.append(f"{t.template_id}: routing row {i} assigned_reviewer {row.assigned_reviewer!r} not in REVIEWER_ROLES")
        if row.policy_divergence and not row.divergence_reason.strip():
            errors.append(f"{t.template_id}: routing row {i} policy_divergence=True but divergence_reason is empty")
        if not row.policy_divergence and row.divergence_reason.strip():
            errors.append(f"{t.template_id}: routing row {i} divergence_reason set but policy_divergence=False")
        for key in row.when:
            if key not in ROUTING_ROW_KEYS:
                errors.append(f"{t.template_id}: routing row {i} has unknown when-key {key!r}")

    errors.extend(_check_routing_coverage(t))

    for msg in t.messages:
        if len(msg.body_variants) < _MIN_BODY_VARIANTS:
            errors.append(
                f"{t.template_id}: message {msg.message_type!r} has {len(msg.body_variants)} "
                f"body_variants, need >= {_MIN_BODY_VARIANTS}"
            )
        for shape in msg.shapes:
            if shape not in THREAD_SHAPES:
                errors.append(f"{t.template_id}: message {msg.message_type!r} has invalid shape {shape!r}")

    declared_shapes = set(t.axes.get("thread_shape", ()))
    produced_shapes = {s for m in t.messages for s in m.shapes}
    for shape in declared_shapes - produced_shapes:
        errors.append(f"{t.template_id}: declared thread_shape {shape!r} produced by no message")

    for value in t.axes.get("urgency", ()):
        if value not in t.urgency_phrasing:
            errors.append(f"{t.template_id}: declared urgency {value!r} has no urgency_phrasing entry")
    for value in t.axes.get("deadline_family", ()):
        if value not in t.deadline_phrasing:
            errors.append(f"{t.template_id}: declared deadline_family {value!r} has no deadline_phrasing entry")

    for value in t.axes.get("cost_impact", ()):
        key = BOOL_PHRASE_KEYS[value]
        if key not in t.cost_phrasing:
            errors.append(f"{t.template_id}: declared cost_impact {value!r} has no cost_phrasing[{key!r}] entry")
    for value in t.axes.get("schedule_impact", ()):
        key = BOOL_PHRASE_KEYS[value]
        if key not in t.schedule_phrasing:
            errors.append(f"{t.template_id}: declared schedule_impact {value!r} has no schedule_phrasing[{key!r}] entry")

    for value in t.axes.get("adversarial", ()):
        if value not in t.secondary_by_adversarial:
            errors.append(f"{t.template_id}: declared adversarial {value!r} has no secondary_by_adversarial entry")

    # Adversarial values in ADVERSARIAL_PHRASE_FIELDS only need a dedicated
    # phrasing placeholder when the template also has other adversarial values
    # (e.g. 'none') to textually differentiate from. A sole-adversarial-value
    # template (e.g. t05, t07) has no sibling cells to distinguish -- the
    # phenomenon can be baked directly into its shared body_variants/
    # urgency_phrasing instead. Where it *is* required, the placeholder must
    # appear in every initial_question body_variant (not just "some" variant
    # somewhere in the template): initial_question is present in every thread
    # shape, so a variant missing it lets a cell render byte-identical to its
    # differently-labeled sibling whenever that variant happens to be drawn.
    _adversarial_values = t.axes.get("adversarial", ())
    _multi_adversarial = len(set(_adversarial_values)) > 1
    for _value, (_pool_attr, _phrase) in ADVERSARIAL_PHRASE_FIELDS.items():
        # A placeholder used anywhere with an empty backing pool is always a
        # bug (random.choice on an empty pool at generation time), regardless
        # of whether this adversarial value is sole or declared alongside
        # others -- so this half of the check is never gated on multiplicity.
        _placeholder_used = any(
            f"{{{_phrase}}}" in body for msg in t.messages for body in msg.body_variants
        )
        if _placeholder_used and not getattr(t, _pool_attr):
            errors.append(
                f"{t.template_id}: {{{_phrase}}} appears in body_variants but {_pool_attr} is empty"
            )
        if not _multi_adversarial or _value not in _adversarial_values:
            continue
        if not getattr(t, _pool_attr):
            errors.append(
                f"{t.template_id}: adversarial value {_value!r} declared alongside other "
                f"adversarial values but {_pool_attr} is empty"
            )
        _missing = _initial_question_variants_missing(t, _phrase)
        if _missing:
            errors.append(
                f"{t.template_id}: adversarial value {_value!r} declared alongside other "
                f"adversarial values but {{{_phrase}}} is missing from "
                f"{len(_missing)} initial_question body_variant(s): {_missing!r}"
            )

    # A template that ever declares 'boundary_urgency' relies on {urgency_phrase}
    # to carry the actual priority/urgent gold label -- both the sole-value case
    # (e.g. t05, which has no dedicated boundary_phrasing pool at all) and the
    # multi-value case (e.g. t09, where {boundary_phrase} differentiates from
    # 'none' but {urgency_phrase} is what differentiates priority from urgent
    # *within* the boundary_urgency family itself). A variant missing
    # {urgency_phrase} renders with zero textual basis for the urgency label,
    # so two boundary_urgency cells that differ only in urgency tier become
    # twins whenever that variant is drawn -- the same defect class this
    # mechanism exists to catch, just on the urgency axis instead of the
    # adversarial axis.
    if "boundary_urgency" in _adversarial_values:
        _missing_urgency = _initial_question_variants_missing(t, "urgency_phrase")
        if _missing_urgency:
            errors.append(
                f"{t.template_id}: adversarial value 'boundary_urgency' relies on "
                f"{{urgency_phrase}} but it is missing from {len(_missing_urgency)} "
                f"initial_question body_variant(s): {_missing_urgency!r}"
            )

    # Scope note: the same twin-cell argument above (a variant missing the
    # phrase that carries an axis's label makes two cells differing only on
    # that axis render identically) applies just as much to every template's
    # plain urgency axis (independent of 'boundary_urgency') and to
    # deadline_family whenever it declares any value other than 'none' --
    # not just to the boundary_urgency special case. This is NOT enforced
    # here as a hard validate_all() error, because doing so today would fail
    # all 21 templates: body_variants are intentionally written with uneven
    # phrase coverage for stylistic variety. But for {deadline_phrase} this
    # is not purely stylistic -- schema.py's deadline_text field is a
    # verbatim span from the rendered thread, so a generator that draws a
    # variant lacking {deadline_phrase} for a non-'none' deadline_family cell
    # has no non-fabricating way to populate deadline_text (it must either
    # invent a span that isn't in the text, or emit "none stated" against a
    # declared family that isn't 'none'). The eligible_body_variants() helper
    # above exists so generate_rfis.py (not yet built) is not left to
    # rediscover this: it MUST restrict variant selection to the indices that
    # function returns for any phrase a cell's gold label depends on. Note
    # this shrinks the effectively-selectable set below the
    # _MIN_BODY_VARIANTS = 3 the validator guarantees on the *authored* list
    # (every template currently has exactly 2 variants carrying both
    # {urgency_phrase} and {deadline_phrase}) -- that diversity gap is real
    # and tracked by test_generate_rfis.py's variant-eligibility-floor test,
    # not silently assumed away.

    adversarial_values = tuple(t.axes.get("adversarial", ()))
    declared_aid = "answer_in_documents" in adversarial_values
    if declared_aid and len(set(adversarial_values)) > 1:
        errors.append(
            f"{t.template_id}: adversarial value 'answer_in_documents' must be the sole adversarial "
            f"axis value (body_variants cannot vary per adversarial value), got {list(adversarial_values)!r}"
        )
    declared_md = "multi_discipline" in adversarial_values
    if declared_md and len(set(adversarial_values)) > 1:
        errors.append(
            f"{t.template_id}: adversarial value 'multi_discipline' must be the sole adversarial "
            f"axis value (its gold difference lives only in secondary_by_adversarial, with no "
            f"dedicated phrasing pool to differentiate it textually), got {list(adversarial_values)!r}"
        )
    if "answer_in_documents" not in t.gold:
        errors.append(f"{t.template_id}: gold is missing required bool 'answer_in_documents'")
    elif not isinstance(t.gold["answer_in_documents"], bool):
        errors.append(f"{t.template_id}: gold['answer_in_documents'] must be a bool, got {t.gold['answer_in_documents']!r}")
    elif t.gold["answer_in_documents"] != declared_aid:
        errors.append(
            f"{t.template_id}: gold['answer_in_documents']={t.gold['answer_in_documents']!r} disagrees with "
            f"adversarial axis {list(adversarial_values)!r} (must be True iff 'answer_in_documents' is declared)"
        )

    for key in ("question_summary_variants", "proposed_solution_variants"):
        value = t.gold.get(key)
        if value is not None and not isinstance(value, list):
            errors.append(f"{t.template_id}: gold[{key!r}] must be a list, got {type(value).__name__}")
        elif isinstance(value, list) and len(value) < _MIN_GOLD_VARIANTS:
            errors.append(
                f"{t.template_id}: gold[{key!r}] has {len(value)} variant(s), need >= {_MIN_GOLD_VARIANTS}"
            )

    _uses_doc_refs = any(
        "{doc_a}" in body or "{doc_b}" in body for msg in t.messages for body in msg.body_variants
    )
    if _uses_doc_refs and pools is not None:
        for discipline in t.axes.get("primary_discipline", ()):
            available = pools.doc_refs.get(discipline, ())
            if len(available) < _MIN_DOC_REFS:
                errors.append(
                    f"{t.template_id}: {{doc_a}}/{{doc_b}} appears in body_variants and declares "
                    f"primary_discipline {discipline!r}, but pools.doc_refs[{discipline!r}] has "
                    f"{len(available)} entries, need >= {_MIN_DOC_REFS}"
                )

    used_phrases: set[str] = set()
    for msg in t.messages:
        for body in msg.body_variants:
            if not body.strip():
                errors.append(f"{t.template_id}: message {msg.message_type!r} has an empty body_variant")
                continue
            for m in _PHRASE_RE.finditer(body):
                used_phrases.add(m.group(1))
                s, e = m.span()
                if (s > 0 and not body[s - 1].isspace()) or (e < len(body) and not body[e].isspace()):
                    errors.append(
                        f"{t.template_id}: message {msg.message_type!r} body_variant has "
                        f"{{{m.group(1)}}} not whitespace-delimited (empty substitution would orphan "
                        f"adjacent text/punctuation) in {body!r}"
                    )
            if not _PHRASE_RE.sub("", body).strip():
                errors.append(
                    f"{t.template_id}: message {msg.message_type!r} body_variant is only phrase "
                    f"placeholders and could render empty: {body!r}"
                )

    for axis, phrase in (
        ("urgency", "urgency_phrase"),
        ("cost_impact", "cost_phrase"),
        ("schedule_impact", "schedule_phrase"),
        ("deadline_family", "deadline_phrase"),
    ):
        if t.axes.get(axis) and phrase not in used_phrases:
            errors.append(
                f"{t.template_id}: axis {axis!r} is declared but {{{phrase}}} never appears in a body_variant"
            )
    for name, text in _non_body_variant_strings(t):
        for m in _PHRASE_RE.finditer(text):
            errors.append(
                f"{t.template_id}: {{{m.group(1)}}} is only substituted in message body_variants, "
                f"found in {name} {text!r}"
            )

    for msg in t.messages:
        if msg.message_type not in MESSAGE_TYPES:
            errors.append(f"{t.template_id}: message has invalid message_type {msg.message_type!r}")

    for shape in declared_shapes:
        required = SHAPE_REQUIRED_MESSAGE_TYPES.get(shape, ())
        for message_type in required:
            if not any(m.message_type == message_type and shape in m.shapes for m in t.messages):
                errors.append(
                    f"{t.template_id}: thread_shape {shape!r} requires a {message_type!r} message "
                    f"whose shapes include {shape!r}, none found"
                )

    known_slots = set(t.slots.keys())
    if pools is not None:
        known_slots |= pools.slot_names()
    for text in _all_template_strings(t):
        for placeholder in _extract_placeholders(text):
            if placeholder in RESERVED_PLACEHOLDERS:
                continue
            if placeholder not in known_slots:
                errors.append(f"{t.template_id}: unresolved placeholder {{{placeholder}}} in {text!r}")

    return errors


def _check_routing_coverage(t: Template) -> list[str]:
    errors: list[str] = []
    rfi_types = t.axes.get("rfi_type", ())
    cost_values = t.axes.get("cost_impact", ())
    schedule_values = t.axes.get("schedule_impact", ())
    urgencies = t.axes.get("urgency", ())
    for rfi_type in rfi_types:
        for cost in cost_values:
            for schedule in schedule_values:
                for urgency in urgencies:
                    values = {
                        "rfi_type": rfi_type,
                        "cost_impact": cost,
                        "schedule_impact": schedule,
                        "urgency": urgency,
                    }
                    if not any(row.matches(values) for row in t.gold_routing_table):
                        errors.append(
                            f"{t.template_id}: no gold_routing_table row matches "
                            f"rfi_type={rfi_type!r}, cost_impact={cost!r}, "
                            f"schedule_impact={schedule!r}, urgency={urgency!r}"
                        )
    return errors


def validate_all(templates: dict[str, Template], pools: SharedPools | None = None) -> list[str]:
    errors: list[str] = []
    for template_id, t in templates.items():
        if template_id != t.template_id:
            errors.append(f"template dict key {template_id!r} != template_id {t.template_id!r}")
        errors.extend(validate_template(t, pools))

    dev_ids = {tid for tid, t in templates.items() if t.pool == DEV_POOL}
    eval_ids = {tid for tid, t in templates.items() if t.pool == EVAL_POOL}
    overlap = dev_ids & eval_ids
    if overlap:
        errors.append(f"dev and eval share template ids: {sorted(overlap)}")

    return errors


def is_compatible(t: Template, cell: "Cell") -> bool:
    if t.pool != cell.split:
        return False
    checks = (
        ("rfi_type", cell.rfi_type),
        ("primary_discipline", cell.primary_discipline),
        ("urgency", cell.urgency),
        ("cost_impact", cell.cost_impact),
        ("schedule_impact", cell.schedule_impact),
        ("deadline_family", cell.deadline_family),
        ("adversarial", cell.adversarial),
        ("thread_shape", cell.thread_shape),
    )
    return all(value in t.axes.get(axis, ()) for axis, value in checks)


def templates_for_cell(templates: dict[str, Template], cell: "Cell") -> list[str]:
    return sorted(tid for tid, t in templates.items() if is_compatible(t, cell))


def gold_row_for_cell(t: Template, cell: "Cell") -> GoldRoutingRow | None:
    """Resolves the gold routing row a cell's axis values match, using the
    same ROUTING_ROW_KEYS-keyed lookup _check_routing_coverage() validates
    coverage for. The single shared source stratification.py and
    generate_rfis.py should both use instead of duplicating this lookup --
    per the module docstring, this reads a template's own gold_routing_table,
    never routing_policy.route_rfi()."""
    values = {key: getattr(cell, key) for key in ROUTING_ROW_KEYS}
    for row in t.gold_routing_table:
        if row.matches(values):
            return row
    return None
