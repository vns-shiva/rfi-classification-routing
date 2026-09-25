"""Corpus stratification: turns the templates.py universe (templates x their
declared axis values) into a concrete 240-thread generation plan with
guaranteed coverage of every axis value, before generate_rfis.py renders a
single thread.

Anti-circularity: this module must NEVER import routing_policy. Gold routing
labels come only from a template's own gold_routing_table (via
templates.gold_row_for_cell); agreement between gold and routing_policy.
route_rfi() is measured after generation, in policy_audit.py. Importing
routing_policy here to inform sampling would make that agreement measurement
circular.

Two concerns are deliberately kept separate:
  - AXIS_WEIGHTS is a realism *prior* (a product-of-independent-axes
    simplification of how often a given combination would occur on a real
    project). It is never reported as ground truth -- only the plan's
    realized, post-hoc counts (coverage_report) go in the paper.
  - QUOTAS are hard per-split per-axis-value minimum counts. Weights alone
    cannot guarantee that a rare-but-important combination (e.g. urgency=
    critical) appears at all; quotas are the actual coverage guarantee.

build_plan() is a two-phase greedy/weighted sampler:
  Phase A closes every quota with the least-used matching (template, cell)
    pair it can find, recording a shortfall (never raising, unless
    strict=True) when no expressible pair can satisfy a quota with the
    templates currently authored.
  Phase B fills the remainder of the split with weighted_choice() draws from
    the currently-least-used expressible pairs, so that pairs get spread
    across the corpus instead of a few pairs dominating it.
  The chosen (template, cell) pairs are then shuffled (per-split) before
    thread_ids are assigned, so a thread's position/date has no relationship
    to how "adversarial" it is -- an unshuffled plan would put every
    quota-closing (often adversarial) thread at the start of the split.

Cell deliberately does NOT enforce coherence (see is_coherent) in
__post_init__: it only enforces that every field is a real vocabulary value.
Coherence is a *combination* property (e.g. "distractor_cost_language only
makes sense when cost_impact=False"), checked separately so that
templates.is_compatible() and cells_for_template() can each decide what to
do with an incoherent-but-vocabulary-valid combination.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any

from templates import (
    ADVERSARIAL_FAMILIES,
    AXIS_NAMES,
    AXIS_VOCAB,
    DEADLINE_FAMILIES,
    DEV_POOL,
    EVAL_POOL,
    GoldRoutingRow,
    POOLS,
    ROUTING_ROW_KEYS,
    SharedPools,
    Template,
    TemplateMessage,
    THREAD_SHAPES,
    UNRESOLVABLE_FAMILIES,
    gold_row_for_cell,
    is_compatible,
    load_shared_pools,
    load_templates,
    templates_for_cell,
    validate_all,
    validate_template,
)
from vocabulary import REVIEWER_ROLES

__all__ = [
    "POOLS", "DEV_POOL", "EVAL_POOL", "AXIS_NAMES", "AXIS_VOCAB",
    "DEADLINE_FAMILIES", "UNRESOLVABLE_FAMILIES", "ADVERSARIAL_FAMILIES",
    "THREAD_SHAPES", "ROUTING_ROW_KEYS", "REVIEWER_ROLES",
    "Template", "TemplateMessage", "GoldRoutingRow", "SharedPools",
    "load_templates", "load_shared_pools", "validate_template", "validate_all",
    "is_compatible", "templates_for_cell", "gold_row_for_cell",
    "Cell", "is_coherent", "cell_incoherence_reasons", "is_expressible",
    "cells_for_template", "expressible_pairs", "FEATURE_NAMES", "cell_features",
    "weighted_choice", "AXIS_WEIGHTS", "cell_weight",
    "QUOTAS", "DIVERGENCE_QUOTAS", "REVIEWER_QUOTAS", "validate_quotas",
    "CORPUS_SIZE", "SPLIT_SIZES", "SPLIT_ORDER", "OVERLAP_SIZE",
    "DEFAULT_SEED", "MAX_TEMPLATE_SHARE",
    "PlanEntry", "CorpusPlan", "CoverageError",
    "build_plan", "unreachable_quotas", "cells_by_split",
    "plan_to_dict", "plan_from_dict", "write_plan", "read_plan",
    "coverage_report", "coverage_shortfalls", "assert_coverage",
    "format_coverage_markdown", "select_overlap", "main",
]

CORPUS_SIZE = 240
SPLIT_SIZES: dict[str, int] = {DEV_POOL: 40, EVAL_POOL: 200}
SPLIT_ORDER: tuple[str, ...] = (DEV_POOL, EVAL_POOL)
OVERLAP_SIZE = 60
DEFAULT_SEED = 20260101

# No single template may supply more than this share of its split, once
# enough templates exist to make that a meaningful constraint. With only one
# template authored for a split, _template_cap() relaxes this automatically
# (see its docstring) -- otherwise no plan could be built at all yet.
MAX_TEMPLATE_SHARE = 0.20

FEATURE_NAMES = AXIS_NAMES


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    split: str
    rfi_type: str
    primary_discipline: str
    urgency: str
    cost_impact: bool
    schedule_impact: bool
    deadline_family: str
    adversarial: str
    thread_shape: str

    def __post_init__(self) -> None:
        if self.split not in POOLS:
            raise ValueError(f"Cell.split {self.split!r} not in {POOLS}")
        for axis in AXIS_NAMES:
            value = getattr(self, axis)
            if axis in ("cost_impact", "schedule_impact") and not isinstance(value, bool):
                raise ValueError(f"Cell.{axis} must be a bool, got {value!r}")
            vocab = AXIS_VOCAB[axis]
            if value not in vocab:
                raise ValueError(f"Cell.{axis} {value!r} not in {vocab}")

    def to_dict(self) -> dict:
        return {"split": self.split, **{axis: getattr(self, axis) for axis in AXIS_NAMES}}

    @staticmethod
    def from_dict(d: dict) -> "Cell":
        return Cell(split=d["split"], **{axis: d[axis] for axis in AXIS_NAMES})


def _cell_sort_key(cell: Cell) -> tuple:
    return tuple(getattr(cell, axis) for axis in AXIS_NAMES)


# ---------------------------------------------------------------------------
# Coherence: a cell can be a valid cross-product member yet describe a
# scenario that makes no narrative sense. These rules catch that, kept
# separate from Cell.__post_init__'s pure vocabulary-membership check.
# ---------------------------------------------------------------------------

def cell_incoherence_reasons(cell: Cell) -> list[str]:
    reasons: list[str] = []

    # C1: an "unresolvable deadline" adversarial cell needs a deadline_family
    # that is actually unresolvable -- pairing it with e.g. "weekday" (a
    # perfectly resolvable family) contradicts the adversarial label.
    if cell.adversarial == "unresolvable_deadline" and cell.deadline_family not in UNRESOLVABLE_FAMILIES:
        reasons.append(
            f"adversarial='unresolvable_deadline' requires deadline_family in "
            f"{UNRESOLVABLE_FAMILIES}, got {cell.deadline_family!r}"
        )

    # C2: "distractor cost language" means the thread mentions cost-sounding
    # language as a red herring while the real answer is cost_impact=False;
    # pairing it with cost_impact=True removes the distractor -- there's
    # nothing to be distracted from.
    if cell.adversarial == "distractor_cost_language" and cell.cost_impact is not False:
        reasons.append("adversarial='distractor_cost_language' requires cost_impact=False")

    # C3: critical urgency with zero cost or schedule impact is incoherent on
    # a construction project -- nothing is "critical" in a vacuum.
    if cell.urgency == "critical" and not (cell.cost_impact or cell.schedule_impact):
        reasons.append("urgency='critical' requires cost_impact or schedule_impact to be True")

    # C4: "boundary urgency" cells are meant to probe the routine/priority and
    # urgent/critical boundaries, not sit at either extreme of the scale.
    if cell.adversarial == "boundary_urgency" and cell.urgency not in ("priority", "urgent"):
        reasons.append("adversarial='boundary_urgency' requires urgency in ('priority', 'urgent')")

    return reasons


def is_coherent(cell: Cell) -> bool:
    return not cell_incoherence_reasons(cell)


def is_expressible(t: Template, cell: Cell) -> bool:
    """A cell a template can actually produce: axis-compatible, coherent, and
    resolvable to a gold routing row via the template's own
    gold_routing_table (never routing_policy.route_rfi())."""
    if not is_compatible(t, cell):
        return False
    if not is_coherent(cell):
        return False
    return gold_row_for_cell(t, cell) is not None


def cells_for_template(t: Template) -> list[Cell]:
    """Every coherent cell in t's declared axis cross-product."""
    cells: list[Cell] = []
    for rfi_type in t.axes.get("rfi_type", ()):
        for primary_discipline in t.axes.get("primary_discipline", ()):
            for urgency in t.axes.get("urgency", ()):
                for cost_impact in t.axes.get("cost_impact", ()):
                    for schedule_impact in t.axes.get("schedule_impact", ()):
                        for deadline_family in t.axes.get("deadline_family", ()):
                            for adversarial in t.axes.get("adversarial", ()):
                                for thread_shape in t.axes.get("thread_shape", ()):
                                    cell = Cell(
                                        split=t.pool,
                                        rfi_type=rfi_type,
                                        primary_discipline=primary_discipline,
                                        urgency=urgency,
                                        cost_impact=cost_impact,
                                        schedule_impact=schedule_impact,
                                        deadline_family=deadline_family,
                                        adversarial=adversarial,
                                        thread_shape=thread_shape,
                                    )
                                    if is_coherent(cell):
                                        cells.append(cell)
    return cells


def expressible_pairs(templates: dict[str, Template]) -> list[tuple[str, Cell]]:
    """Sorted, deduplicated (template_id, Cell) pairs across every template."""
    pairs: set[tuple[str, Cell]] = set()
    for tid, t in templates.items():
        for cell in cells_for_template(t):
            if is_expressible(t, cell):
                pairs.add((tid, cell))
    return sorted(pairs, key=lambda p: (p[0], _cell_sort_key(p[1])))


def cell_features(cell: Cell) -> frozenset[tuple[str, Any]]:
    return frozenset((axis, getattr(cell, axis)) for axis in AXIS_NAMES)


def cells_by_split(templates: dict[str, Template], split: str) -> list[tuple[str, Cell]]:
    return [(tid, cell) for tid, cell in expressible_pairs(templates) if cell.split == split]


# ---------------------------------------------------------------------------
# Weighted sampling
# ---------------------------------------------------------------------------

def weighted_choice(rng: random.Random, values: list[Any], weights: dict[Any, float]) -> Any:
    """Deterministic (given rng's state) weighted draw from `values`. Every
    value must have a non-negative entry in `weights`; a zero-weight value is
    never returned. Raises ValueError on empty `values`, a negative weight,
    or a non-positive total weight, and KeyError if a value in `values` has
    no entry in `weights`."""
    if not values:
        raise ValueError("weighted_choice requires at least one value")

    weighed: list[tuple[Any, float]] = []
    total = 0.0
    for v in values:
        if v not in weights:
            raise KeyError(f"weighted_choice: no weight given for value {v!r}")
        w = weights[v]
        if w < 0:
            raise ValueError(f"weighted_choice: negative weight for value {v!r}: {w}")
        weighed.append((v, w))
        total += w

    if total <= 0:
        raise ValueError("weighted_choice requires a positive total weight")

    r = rng.uniform(0.0, total)
    upto = 0.0
    for v, w in weighed:
        if w == 0:
            continue
        upto += w
        if upto >= r:
            return v
    # Float rounding can leave r fractionally above the true total; fall back
    # to the last non-zero-weight value rather than ever returning nothing.
    for v, w in reversed(weighed):
        if w > 0:
            return v
    raise ValueError("weighted_choice: unreachable -- no positive-weight value found")


# A realism prior (product-of-independent-axes simplification of how often a
# combination would occur on a real project), used only to bias Phase B's
# fill order. Never reported as ground truth -- see the module docstring.
AXIS_WEIGHTS: dict[str, dict[Any, float]] = {
    "rfi_type": {
        "code_compliance_question": 0.08,
        "coordination_conflict": 0.12,
        "design_clarification": 0.30,
        "document_discrepancy": 0.18,
        "field_condition_conflict": 0.20,
        "substitution_request": 0.12,
    },
    "primary_discipline": {
        "Architectural": 0.22,
        "Civil": 0.08,
        "Electrical": 0.12,
        "Fire_Protection": 0.06,
        "General": 0.08,
        "Mechanical": 0.14,
        "Plumbing": 0.10,
        "Structural": 0.20,
    },
    "urgency": {
        "routine": 0.40,
        "priority": 0.35,
        "urgent": 0.18,
        "critical": 0.07,
    },
    "cost_impact": {False: 0.55, True: 0.45},
    "schedule_impact": {False: 0.55, True: 0.45},
    "deadline_family": {
        "none": 0.20,
        "relative_days": 0.16,
        "relative_weeks": 0.10,
        "business_days": 0.14,
        "weekday": 0.10,
        "this_next_week": 0.10,
        "end_of": 0.08,
        "event_anchored": 0.07,
        "ambiguous": 0.05,
    },
    "adversarial": {
        "none": 0.55,
        "multi_discipline": 0.09,
        "boundary_urgency": 0.09,
        "unresolvable_deadline": 0.07,
        "answer_in_documents": 0.08,
        "distractor_cost_language": 0.06,
        "type_ambiguity": 0.06,
    },
    "thread_shape": {
        "single_message": 0.30,
        "two_message_qa": 0.35,
        "multi_turn_clarification": 0.25,
        "resubmittal": 0.10,
    },
}


def cell_weight(cell: Cell) -> float:
    weight = 1.0
    for axis in AXIS_NAMES:
        weight *= AXIS_WEIGHTS[axis][getattr(cell, axis)]
    return weight


# ---------------------------------------------------------------------------
# Quotas: hard per-split per-axis-value minimum counts. These are the actual
# coverage guarantee -- AXIS_WEIGHTS alone would let a rare-but-important
# value (e.g. urgency=critical) appear zero times by chance.
# ---------------------------------------------------------------------------

QUOTAS: dict[str, dict[str, dict[Any, int]]] = {
    DEV_POOL: {
        "rfi_type": {v: 3 for v in AXIS_VOCAB["rfi_type"]},
        "primary_discipline": {v: 3 for v in AXIS_VOCAB["primary_discipline"]},
        "urgency": {"routine": 6, "priority": 6, "urgent": 3, "critical": 2},
        "deadline_family": {v: 1 for v in AXIS_VOCAB["deadline_family"]},
        "adversarial": {v: 1 for v in ADVERSARIAL_FAMILIES if v != "none"},
        "thread_shape": {v: 3 for v in AXIS_VOCAB["thread_shape"]},
    },
    EVAL_POOL: {
        "rfi_type": {v: 15 for v in AXIS_VOCAB["rfi_type"]},
        "primary_discipline": {v: 12 for v in AXIS_VOCAB["primary_discipline"]},
        "urgency": {"routine": 30, "priority": 40, "urgent": 20, "critical": 10},
        "deadline_family": {v: 6 for v in AXIS_VOCAB["deadline_family"]},
        "adversarial": {v: 6 for v in ADVERSARIAL_FAMILIES if v != "none"},
        "thread_shape": {v: 20 for v in AXIS_VOCAB["thread_shape"]},
    },
}

# Minimum count, per split, of gold_routing_table rows with
# policy_divergence=True -- the corpus needs enough of these for
# policy_audit.py's gold-vs-route_rfi() agreement rate to be measured on a
# non-trivial sample of the cases where they actually disagree.
DIVERGENCE_QUOTAS: dict[str, int] = {DEV_POOL: 3, EVAL_POOL: 16}

# Minimum count, per split, of threads assigned to *each* REVIEWER_ROLES value
# -- without this, Phase A's deterministic tie-break (least-used pair, then
# alphabetically-first cell) can leave a reviewer role with zero threads even
# though it is expressible, because it never happens to be the alphabetically
# -- or otherwise -- preferred candidate for any quota-closing draw.
REVIEWER_QUOTAS: dict[str, int] = {DEV_POOL: 1, EVAL_POOL: 5}


def validate_quotas() -> list[str]:
    errors: list[str] = []
    for split, axis_quotas in QUOTAS.items():
        if split not in POOLS:
            errors.append(f"QUOTAS has unknown split {split!r}")
            continue
        split_size = SPLIT_SIZES[split]
        for axis, value_quotas in axis_quotas.items():
            if axis not in AXIS_NAMES:
                errors.append(f"QUOTAS[{split!r}] has unknown axis {axis!r}")
                continue
            vocab = AXIS_VOCAB[axis]
            total = 0
            for value, min_count in value_quotas.items():
                if value not in vocab:
                    errors.append(f"QUOTAS[{split!r}][{axis!r}] has unknown value {value!r}")
                if min_count < 0:
                    errors.append(f"QUOTAS[{split!r}][{axis!r}][{value!r}] is negative")
                total += min_count
            if total > split_size:
                errors.append(
                    f"QUOTAS[{split!r}][{axis!r}] minimums sum to {total}, exceeding split size {split_size}"
                )
    for split, min_divergent in DIVERGENCE_QUOTAS.items():
        if split not in POOLS:
            errors.append(f"DIVERGENCE_QUOTAS has unknown split {split!r}")
        elif min_divergent > SPLIT_SIZES[split]:
            errors.append(f"DIVERGENCE_QUOTAS[{split!r}]={min_divergent} exceeds split size {SPLIT_SIZES[split]}")
    for split, min_per_reviewer in REVIEWER_QUOTAS.items():
        if split not in POOLS:
            errors.append(f"REVIEWER_QUOTAS has unknown split {split!r}")
            continue
        if min_per_reviewer < 0:
            errors.append(f"REVIEWER_QUOTAS[{split!r}] is negative")
        total = min_per_reviewer * len(REVIEWER_ROLES)
        if total > SPLIT_SIZES[split]:
            errors.append(
                f"REVIEWER_QUOTAS[{split!r}]={min_per_reviewer} * {len(REVIEWER_ROLES)} roles = {total}, "
                f"exceeding split size {SPLIT_SIZES[split]}"
            )
    return errors


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class CoverageError(Exception):
    pass


@dataclass(frozen=True)
class PlanEntry:
    thread_id: str
    split: str
    template_id: str
    cell: Cell
    variant_index: int
    seed: int

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "split": self.split,
            "template_id": self.template_id,
            "cell": self.cell.to_dict(),
            "variant_index": self.variant_index,
            "seed": self.seed,
        }

    @staticmethod
    def from_dict(d: dict) -> "PlanEntry":
        return PlanEntry(
            thread_id=d["thread_id"],
            split=d["split"],
            template_id=d["template_id"],
            cell=Cell.from_dict(d["cell"]),
            variant_index=d["variant_index"],
            seed=d["seed"],
        )


@dataclass(frozen=True)
class CorpusPlan:
    entries: tuple[PlanEntry, ...]
    seed: int
    shortfalls: tuple[str, ...] = ()

    def by_split(self, split: str) -> list[PlanEntry]:
        return [e for e in self.entries if e.split == split]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "shortfalls": list(self.shortfalls),
            "entries": [e.to_dict() for e in self.entries],
        }

    @staticmethod
    def from_dict(d: dict) -> "CorpusPlan":
        return CorpusPlan(
            seed=d["seed"],
            shortfalls=tuple(d.get("shortfalls", ())),
            entries=tuple(PlanEntry.from_dict(e) for e in d["entries"]),
        )


def _thread_seed(top_seed: int, thread_id: str) -> int:
    digest = blake2b(f"{top_seed}:{thread_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _template_cap(split_size: int, n_templates: int) -> int:
    """Per-template usage cap for a split. With few templates authored, a
    strict MAX_TEMPLATE_SHARE cap would make the split unfillable (one
    template would have to supply 100% of it) -- so the cap is relaxed to
    whichever is larger: the intended share, or an even split across however
    many templates currently exist."""
    if n_templates <= 0:
        return 0
    by_share = math.ceil(split_size * MAX_TEMPLATE_SHARE)
    by_spread = math.ceil(split_size / n_templates)
    return max(by_share, by_spread)


def build_plan(templates: dict[str, Template], *, seed: int = DEFAULT_SEED, strict: bool = False) -> CorpusPlan:
    rng = random.Random(seed)
    all_pairs = expressible_pairs(templates)
    entries_by_split: dict[str, list[PlanEntry]] = {s: [] for s in SPLIT_ORDER}
    shortfalls: list[str] = []

    for split in SPLIT_ORDER:
        split_size = SPLIT_SIZES[split]
        split_pairs = [p for p in all_pairs if p[1].split == split]
        split_templates = sorted({tid for tid, _ in split_pairs})
        if not split_templates:
            raise CoverageError(f"{split}: no expressible (template, cell) pairs -- no templates for this split")

        cap = _template_cap(split_size, len(split_templates))
        use_count: dict[tuple[str, Cell], int] = {p: 0 for p in split_pairs}
        template_use_count: dict[str, int] = {tid: 0 for tid in split_templates}
        chosen: list[tuple[str, Cell]] = []

        def _take(pair: tuple[str, Cell]) -> None:
            chosen.append(pair)
            use_count[pair] += 1
            template_use_count[pair[0]] += 1

        def _best_under_cap(candidates: list[tuple[str, Cell]]) -> tuple[str, Cell]:
            ranked = sorted(
                candidates,
                key=lambda p: (use_count[p], template_use_count[p[0]], p[0], _cell_sort_key(p[1])),
            )
            under_cap = [p for p in ranked if template_use_count[p[0]] < cap]
            return under_cap[0] if under_cap else ranked[0]

        # Phase A: greedily close every quota.
        for axis, value_quotas in QUOTAS.get(split, {}).items():
            for value, min_count in value_quotas.items():
                candidates = [p for p in split_pairs if getattr(p[1], axis) == value]
                have = sum(1 for p in chosen if getattr(p[1], axis) == value)
                if not candidates:
                    if min_count > 0:
                        shortfalls.append(
                            f"{split}: quota {axis}={value!r} (need {min_count}) unreachable -- "
                            f"no expressible pair supplies it"
                        )
                    continue
                while have < min_count and len(chosen) < split_size:
                    _take(_best_under_cap(candidates))
                    have += 1
                if have < min_count:
                    shortfalls.append(
                        f"{split}: quota {axis}={value!r} needs {min_count}, only reached {have}"
                    )

        min_divergent = DIVERGENCE_QUOTAS.get(split, 0)
        if min_divergent:
            divergent_candidates = [
                p for p in split_pairs
                if (row := gold_row_for_cell(templates[p[0]], p[1])) is not None and row.policy_divergence
            ]
            have = sum(
                1 for p in chosen
                if (row := gold_row_for_cell(templates[p[0]], p[1])) is not None and row.policy_divergence
            )
            if not divergent_candidates and min_divergent > 0:
                shortfalls.append(
                    f"{split}: divergence quota (need {min_divergent}) unreachable -- "
                    f"no expressible pair has policy_divergence=True"
                )
            while have < min_divergent and len(chosen) < split_size and divergent_candidates:
                _take(_best_under_cap(divergent_candidates))
                have += 1
            if have < min_divergent:
                shortfalls.append(f"{split}: divergence quota needs {min_divergent}, only reached {have}")

        min_per_reviewer = REVIEWER_QUOTAS.get(split, 0)
        if min_per_reviewer:
            for role in REVIEWER_ROLES:
                role_candidates = [
                    p for p in split_pairs
                    if (row := gold_row_for_cell(templates[p[0]], p[1])) is not None
                    and row.assigned_reviewer == role
                ]
                have = sum(
                    1 for p in chosen
                    if (row := gold_row_for_cell(templates[p[0]], p[1])) is not None
                    and row.assigned_reviewer == role
                )
                if not role_candidates:
                    if min_per_reviewer > 0:
                        shortfalls.append(
                            f"{split}: reviewer quota {role!r} (need {min_per_reviewer}) unreachable -- "
                            f"no expressible pair assigns it"
                        )
                    continue
                while have < min_per_reviewer and len(chosen) < split_size:
                    _take(_best_under_cap(role_candidates))
                    have += 1
                if have < min_per_reviewer:
                    shortfalls.append(
                        f"{split}: reviewer quota {role!r} needs {min_per_reviewer}, only reached {have}"
                    )

        # Phase B: weighted fill from the currently-least-used tier.
        while len(chosen) < split_size:
            min_use = min(use_count.values())
            tier = [p for p in split_pairs if use_count[p] == min_use and template_use_count[p[0]] < cap]
            if not tier:
                tier = [p for p in split_pairs if use_count[p] == min_use]
            weights = {p: cell_weight(p[1]) for p in tier}
            _take(weighted_choice(rng, tier, weights))

        # variant_index counts repeats of the same (template, cell) pair, in
        # the order Phase A/B chose them (before the shuffle below).
        seen_counts: dict[tuple[str, Cell], int] = {}
        thread_entries: list[tuple[str, Cell, int]] = []
        for tid, cell in chosen:
            variant_index = seen_counts.get((tid, cell), 0)
            seen_counts[(tid, cell)] = variant_index + 1
            thread_entries.append((tid, cell, variant_index))

        # Shuffle so a thread's position (and later, its assigned date) has
        # no relationship to how adversarial-heavy the quota-closing phase
        # made the front of the list.
        rng.shuffle(thread_entries)

        for i, (tid, cell, variant_index) in enumerate(thread_entries, start=1):
            thread_id = f"{split}-{i:04d}"
            entries_by_split[split].append(
                PlanEntry(
                    thread_id=thread_id,
                    split=split,
                    template_id=tid,
                    cell=cell,
                    variant_index=variant_index,
                    seed=_thread_seed(seed, thread_id),
                )
            )

    if shortfalls and strict:
        raise CoverageError("build_plan(strict=True) coverage shortfalls:\n" + "\n".join(sorted(set(shortfalls))))

    all_entries = tuple(e for split in SPLIT_ORDER for e in entries_by_split[split])
    return CorpusPlan(entries=all_entries, seed=seed, shortfalls=tuple(sorted(set(shortfalls))))


def unreachable_quotas(templates: dict[str, Template]) -> list[str]:
    """Authoring-time diagnostic: which QUOTAS/DIVERGENCE_QUOTAS entries have
    *zero* supplying expressible pairs given the templates authored so far
    (a structural gap -- more templates are needed -- as opposed to
    build_plan's own shortfalls, which can also mean the split simply filled
    up before a quota was fully closed)."""
    problems: list[str] = []
    pairs = expressible_pairs(templates)
    for split in SPLIT_ORDER:
        split_pairs = [p for p in pairs if p[1].split == split]
        available: dict[str, set] = {axis: set() for axis in AXIS_NAMES}
        for _, cell in split_pairs:
            for axis in AXIS_NAMES:
                available[axis].add(getattr(cell, axis))
        for axis, value_quotas in QUOTAS.get(split, {}).items():
            for value, min_count in value_quotas.items():
                if min_count > 0 and value not in available[axis]:
                    problems.append(
                        f"{split}: quota requires {axis}={value!r} but no expressible pair supplies it"
                    )
        min_divergent = DIVERGENCE_QUOTAS.get(split, 0)
        if min_divergent > 0:
            has_divergent = any(
                (row := gold_row_for_cell(templates[tid], cell)) is not None and row.policy_divergence
                for tid, cell in split_pairs
            )
            if not has_divergent:
                problems.append(f"{split}: divergence quota requires policy_divergence=True rows but none exist")
        min_per_reviewer = REVIEWER_QUOTAS.get(split, 0)
        if min_per_reviewer > 0:
            reachable_reviewers = {
                row.assigned_reviewer
                for tid, cell in split_pairs
                if (row := gold_row_for_cell(templates[tid], cell)) is not None
            }
            for role in REVIEWER_ROLES:
                if role not in reachable_reviewers:
                    problems.append(f"{split}: reviewer quota requires {role!r} but no expressible pair supplies it")
    return sorted(problems)


def plan_to_dict(plan: CorpusPlan) -> dict:
    return plan.to_dict()


def plan_from_dict(d: dict) -> CorpusPlan:
    return CorpusPlan.from_dict(d)


def write_plan(plan: CorpusPlan, path: Path) -> None:
    path.write_text(json.dumps(plan_to_dict(plan), indent=2), encoding="utf-8")


def read_plan(path: Path) -> CorpusPlan:
    return plan_from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------

def coverage_report(plan: CorpusPlan, templates: dict[str, Template] | None = None) -> dict:
    report: dict[str, Any] = {"corpus_size": len(plan.entries), "splits": {}}
    for split in SPLIT_ORDER:
        entries = plan.by_split(split)
        axis_counts = {axis: {value: 0 for value in AXIS_VOCAB[axis]} for axis in AXIS_NAMES}
        template_counts: dict[str, int] = {}
        for e in entries:
            for axis in AXIS_NAMES:
                axis_counts[axis][getattr(e.cell, axis)] += 1
            template_counts[e.template_id] = template_counts.get(e.template_id, 0) + 1

        split_report: dict[str, Any] = {
            "thread_count": len(entries),
            "axis_counts": {axis: {str(v): c for v, c in values.items()} for axis, values in axis_counts.items()},
            "template_counts": dict(sorted(template_counts.items())),
        }

        if templates is not None:
            reviewer_counts: dict[str, int] = {role: 0 for role in REVIEWER_ROLES}
            divergence_count = 0
            for e in entries:
                row = gold_row_for_cell(templates[e.template_id], e.cell)
                if row is not None:
                    reviewer_counts[row.assigned_reviewer] += 1
                    if row.policy_divergence:
                        divergence_count += 1
            split_report["reviewer_counts"] = reviewer_counts
            split_report["policy_divergence_count"] = divergence_count

        report["splits"][split] = split_report
    return report


def coverage_shortfalls(report: dict) -> list[str]:
    shortfalls: list[str] = []
    for split in SPLIT_ORDER:
        split_report = report["splits"].get(split)
        if split_report is None:
            continue
        axis_counts = split_report["axis_counts"]
        for axis, value_quotas in QUOTAS.get(split, {}).items():
            for value, min_count in value_quotas.items():
                have = axis_counts.get(axis, {}).get(str(value), 0)
                if have < min_count:
                    shortfalls.append(f"{split}: axis {axis!r} value {value!r} has {have}, needs >= {min_count}")
        min_divergent = DIVERGENCE_QUOTAS.get(split, 0)
        if min_divergent and "policy_divergence_count" in split_report:
            have = split_report["policy_divergence_count"]
            if have < min_divergent:
                shortfalls.append(f"{split}: policy_divergence_count has {have}, needs >= {min_divergent}")
        min_per_reviewer = REVIEWER_QUOTAS.get(split, 0)
        if min_per_reviewer and "reviewer_counts" in split_report:
            for role, count in split_report["reviewer_counts"].items():
                if count < min_per_reviewer:
                    shortfalls.append(f"{split}: reviewer {role!r} has {count}, needs >= {min_per_reviewer}")
    return shortfalls


def assert_coverage(plan: CorpusPlan, templates: dict[str, Template] | None = None) -> None:
    report = coverage_report(plan, templates)
    shortfalls = coverage_shortfalls(report)
    if shortfalls:
        raise CoverageError("coverage shortfalls:\n" + "\n".join(shortfalls))


def format_coverage_markdown(report: dict) -> str:
    lines = ["# Corpus coverage report", "", f"Total threads: {report['corpus_size']}", ""]
    for split in SPLIT_ORDER:
        split_report = report["splits"].get(split)
        if split_report is None:
            continue
        lines.append(f"## {split} ({split_report['thread_count']} threads)")
        lines.append("")
        for axis in AXIS_NAMES:
            lines.append(f"### {axis}")
            lines.append("")
            lines.append("| value | count |")
            lines.append("|---|---|")
            for value, count in split_report["axis_counts"][axis].items():
                lines.append(f"| {value} | {count} |")
            lines.append("")
        if "reviewer_counts" in split_report:
            lines.append("### reviewer")
            lines.append("")
            lines.append("| reviewer | count |")
            lines.append("|---|---|")
            for reviewer, count in split_report["reviewer_counts"].items():
                lines.append(f"| {reviewer} | {count} |")
            lines.append("")
            lines.append(f"policy_divergence_count: {split_report['policy_divergence_count']}")
            lines.append("")
    return "\n".join(lines)


def select_overlap(plan: CorpusPlan, *, seed: int = DEFAULT_SEED, n: int = OVERLAP_SIZE) -> list[str]:
    """Stratified-by-rfi_type selection of n eval-split thread_ids for double
    annotation (60/200 = 30%)."""
    eval_entries = plan.by_split(EVAL_POOL)
    if n > len(eval_entries):
        raise ValueError(f"cannot select {n} overlap threads from {len(eval_entries)} eval threads")

    by_type: dict[str, list[PlanEntry]] = {}
    for e in eval_entries:
        by_type.setdefault(e.cell.rfi_type, []).append(e)
    for group in by_type.values():
        group.sort(key=lambda e: e.thread_id)

    types_sorted = sorted(by_type)
    total = len(eval_entries)
    exact = {t: n * len(by_type[t]) / total for t in types_sorted}
    quota = {t: int(exact[t]) for t in types_sorted}
    remainder = n - sum(quota.values())
    by_fraction = sorted(types_sorted, key=lambda t: (-(exact[t] - quota[t]), t))
    for t in by_fraction[:remainder]:
        quota[t] += 1

    rng = random.Random(seed)
    selected: list[str] = []
    for t in types_sorted:
        pool = by_type[t]
        k = min(quota[t], len(pool))
        selected.extend(e.thread_id for e in rng.sample(pool, k))

    if len(selected) < n:
        leftover = [e for e in eval_entries if e.thread_id not in selected]
        rng.shuffle(leftover)
        for e in leftover:
            if len(selected) >= n:
                break
            selected.append(e.thread_id)

    return sorted(selected)


def main() -> None:
    templates = load_templates()
    problems = unreachable_quotas(templates)
    if problems:
        print("unreachable quotas (author more templates to close these):")
        for p in problems:
            print(f"  - {p}")
        print()
    plan = build_plan(templates, strict=False)
    if plan.shortfalls:
        print("build_plan shortfalls:")
        for s in plan.shortfalls:
            print(f"  - {s}")
        print()
    report = coverage_report(plan, templates)
    print(format_coverage_markdown(report))


if __name__ == "__main__":
    main()
