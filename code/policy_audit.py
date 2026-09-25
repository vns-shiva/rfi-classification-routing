"""Corpus-level audit of routing_policy.py against the generated gold-label
corpus (pilot-data/corpus/labels + provenance), independent of any LLM
prediction.

This is the corpus-level counterpart to code/tests/test_policy_agreement.py,
which performs the same policy_divergence-vs-route_rfi() comparison at
*template-authoring* time, over every expressible (template, Cell) pair.
That test's docstring explicitly assigns this module the job of running the
same kind of check against actually-generated RFILabel rows, once
generate_rfis.py has produced them. Section S5 (divergence_census) below
mirrors that test's comparison logic at the corpus level; the corpus-level
gold labels are expected to agree with it exactly, since generate_rfis.py
sets assigned_reviewer/routing_rationale/escalation directly from each
cell's gold_row_for_cell() row and never from routing_policy.route_rfi()
(see templates.py's and routing_policy.py's module docstrings) -- any
disagreement this module finds is either a generation bug or a template row
whose policy_divergence flag has drifted, not evidence the policy is
"wrong".

Import-decoupling contract: this module may only import project modules
from {schema, gating, routing_policy, vocabulary, templates, stratification}
(stratification re-exports the templates.py symbols this module needs). It
must never import routing_eval -- the two are independent passes over the
same corpus (this one audits the deterministic policy against gold; that
one scores prediction arms against gold), so either can be run and checked
without the other.

Run as a script:

    python code/policy_audit.py --corpus pilot-data/corpus \
        --out pilot-data/policy_audit --strict
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from gating import gate
from routing_policy import (
    RULE_IDS,
    RULES,
    NoRuleMatchedError,
    escalation_for,
    route_rfi,
    route_with_rules,
)
from schema import RFILabel, read_label, validate_label
from stratification import Cell, gold_row_for_cell, load_templates
from vocabulary import DISCIPLINES, RFI_TYPES, URGENCY_TIERS

MIN_RULE_SUPPORT = 10


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusThread:
    thread_id: str
    template_id: str
    cell: Cell
    label: RFILabel


def load_corpus(corpus_dir: Path) -> list[CorpusThread]:
    """Loads every gold label (labels/*.json) and cross-references its
    provenance record for the originating template_id and Cell. Only the
    gold annotator is in scope here -- this module audits routing_policy.py
    against how the corpus was generated, not inter-annotator agreement
    (that is annotator_agreement.py's job)."""
    labels_dir = corpus_dir / "labels"
    provenance_dir = corpus_dir / "provenance"
    threads: list[CorpusThread] = []
    for path in sorted(labels_dir.glob("*.json")):
        label = read_label(path)
        errors = validate_label(label)
        if errors:
            raise ValueError(f"{path}: invalid gold label: {errors}")
        prov_path = provenance_dir / f"{label.thread_id}.json"
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        cell = Cell.from_dict(prov["cell"])
        threads.append(
            CorpusThread(
                thread_id=label.thread_id,
                template_id=prov["template_id"],
                cell=cell,
                label=label,
            )
        )
    return threads


# ---------------------------------------------------------------------------
# Report shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditSection:
    name: str
    summary: dict[str, Any]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DivergenceCase:
    thread_id: str
    template_id: str
    cell: dict
    gold_assigned_reviewer: str
    gold_escalation: bool
    policy_assigned_reviewer: str
    policy_escalation: bool
    row_policy_divergence: bool
    actual_divergence: bool
    kind: str  # "false_positive_divergence_flag" | "missed_divergence_flag"


@dataclass(frozen=True)
class AuditReport:
    corpus_dir: str
    n_threads: int
    sections: dict[str, AuditSection]
    divergence_cases: list[DivergenceCase]
    strict_failures: list[str]


# ---------------------------------------------------------------------------
# S1: structural rule inventory / reachability (no corpus needed)
# ---------------------------------------------------------------------------


def rule_inventory_and_reachability() -> AuditSection:
    """Enumerates the full RFI_TYPES x DISCIPLINES x cost_impact x
    schedule_impact x URGENCY_TIERS grid (768 cells, the same enumeration
    routing_policy.reachable_reviewers() uses) and tallies which rule_id
    fires and which reviewers are reachable. Purely structural -- does not
    touch the corpus -- so it also serves as a standalone sanity check that
    every rule in RULE_IDS actually fires somewhere."""
    rule_fire_counts: Counter[str] = Counter()
    reviewers_by_rule: dict[str, set[str]] = {rid: set() for rid in RULE_IDS}
    reviewers: set[str] = set()
    n_cells = 0
    for rfi_type in RFI_TYPES:
        for discipline in DISCIPLINES:
            for cost_impact in (False, True):
                for schedule_impact in (False, True):
                    for urgency in URGENCY_TIERS:
                        n_cells += 1
                        decision = route_rfi(rfi_type, discipline, cost_impact, schedule_impact, urgency)
                        rule_fire_counts[decision.rule_id] += 1
                        reviewers_by_rule[decision.rule_id].add(decision.assigned_reviewer)
                        reviewers.add(decision.assigned_reviewer)

    unreachable_rules = [rid for rid in RULE_IDS if rule_fire_counts[rid] == 0]
    return AuditSection(
        name="rule_inventory_and_reachability",
        summary={
            "n_cells": n_cells,
            "rule_fire_counts": dict(rule_fire_counts),
            "unreachable_rules": unreachable_rules,
            "n_reachable_reviewers": len(reviewers),
            "reachable_reviewers": sorted(reviewers),
        },
        details={"reviewers_by_rule": {rid: sorted(rs) for rid, rs in reviewers_by_rule.items()}},
    )


# ---------------------------------------------------------------------------
# S2: rule hit counts over the actual corpus
# ---------------------------------------------------------------------------


def rule_hit_counts(threads: list[CorpusThread]) -> AuditSection:
    """For each corpus thread, routes the *gold label's own* classification
    fields through route_rfi() and tallies which rule fires. This is a
    validity check on the deterministic policy applied to the corpus's real
    field distribution, not a comparison to the gold routing decision --
    see policy_agreement() in annotator_agreement.py for that discipline
    applied the same way to annotator fields."""
    counts: Counter[str] = Counter()
    by_rfi_type: dict[str, Counter[str]] = {t: Counter() for t in RFI_TYPES}
    for t in threads:
        decision = route_rfi(
            rfi_type=t.label.rfi_type,
            primary_discipline=t.label.primary_discipline,
            cost_impact=t.label.cost_impact,
            schedule_impact=t.label.schedule_impact,
            urgency=t.label.urgency,
        )
        counts[decision.rule_id] += 1
        by_rfi_type.setdefault(t.label.rfi_type, Counter())[decision.rule_id] += 1

    return AuditSection(
        name="rule_hit_counts",
        summary={"n_threads": len(threads), "rule_fire_counts": dict(counts)},
        details={"rule_fire_counts_by_rfi_type": {k: dict(v) for k, v in by_rfi_type.items()}},
    )


# ---------------------------------------------------------------------------
# S3: escalation-formula consistency
# ---------------------------------------------------------------------------


def escalation_consistency(threads: list[CorpusThread]) -> AuditSection:
    """Checks each gold label's escalation flag against escalation_for()
    computed from that same label's own cost_impact/schedule_impact/urgency
    -- the one formula routing_policy.route_rfi() and this module both use,
    per escalation_for()'s docstring. Every mismatch here is either a
    generation bug or a template row whose gold escalation was authored
    inconsistently with the shared formula."""
    n = len(threads)
    n_match = 0
    mismatches: list[dict[str, Any]] = []
    for t in threads:
        expected = escalation_for(t.label.cost_impact, t.label.schedule_impact, t.label.urgency)
        if expected == t.label.escalation:
            n_match += 1
        else:
            mismatches.append(
                {
                    "thread_id": t.thread_id,
                    "gold_escalation": t.label.escalation,
                    "formula_escalation": expected,
                    "cost_impact": t.label.cost_impact,
                    "schedule_impact": t.label.schedule_impact,
                    "urgency": t.label.urgency,
                }
            )

    return AuditSection(
        name="escalation_consistency",
        summary={
            "n_threads": n,
            "n_match": n_match,
            "n_mismatch": n - n_match,
            "match_rate": n_match / n if n else float("nan"),
        },
        details={"mismatches": mismatches},
    )


# ---------------------------------------------------------------------------
# S4: leave-one-rule-out contribution ablation
# ---------------------------------------------------------------------------


def rule_contribution(threads: list[CorpusThread]) -> AuditSection:
    """For each rule, removes it from RULES and re-routes every corpus
    thread's own classification fields, counting how many threads would get
    a different assigned_reviewer (or become unroutable, when ablating the
    R4_discipline_default catch-all leaves some cells matching no rule --
    route_with_rules() raises NoRuleMatchedError there instead of silently
    misrouting). n_support -- how many corpus threads the *full* ruleset
    actually routes via this rule -- gates interpretation: a rule with fewer
    than MIN_RULE_SUPPORT supporting threads has too little corpus evidence
    for its contribution estimate to be meaningful."""
    full_decisions = [
        route_rfi(t.label.rfi_type, t.label.primary_discipline, t.label.cost_impact, t.label.schedule_impact, t.label.urgency)
        for t in threads
    ]

    per_rule: dict[str, dict[str, Any]] = {}
    for rule_id in RULE_IDS:
        reduced = tuple(r for r in RULES if r.rule_id != rule_id)
        n_support = sum(1 for d in full_decisions if d.rule_id == rule_id)
        n_changed = 0
        n_unroutable = 0

        for t, full in zip(threads, full_decisions):
            try:
                ablated = route_with_rules(
                    t.label.rfi_type,
                    t.label.primary_discipline,
                    t.label.cost_impact,
                    t.label.schedule_impact,
                    t.label.urgency,
                    rules=reduced,
                )
            except NoRuleMatchedError:
                n_unroutable += 1
                if full.rule_id == rule_id:
                    n_changed += 1
                continue
            if ablated.assigned_reviewer != full.assigned_reviewer:
                n_changed += 1

        per_rule[rule_id] = {
            "n_support": n_support,
            "n_changed_if_removed": n_changed,
            "n_unroutable_if_removed": n_unroutable,
            "meets_min_support": n_support >= MIN_RULE_SUPPORT,
        }

    return AuditSection(
        name="rule_contribution",
        summary={"n_threads": len(threads), "min_rule_support": MIN_RULE_SUPPORT},
        details={"per_rule": per_rule},
    )


# ---------------------------------------------------------------------------
# S5: corpus-level divergence census
# ---------------------------------------------------------------------------


def divergence_census(
    threads: list[CorpusThread], templates: dict[str, Any]
) -> tuple[AuditSection, list[DivergenceCase]]:
    """The corpus-level counterpart to
    test_policy_agreement.py::test_policy_divergence_flag_agrees_with_route_rfi,
    run against actually-generated threads instead of every expressible
    (template, Cell) pair. For each thread: looks up the template row via
    gold_row_for_cell(), computes what route_rfi() would do on that same
    Cell's fields, and classifies the row's authored policy_divergence flag
    as a false positive (flagged divergent but gold and policy actually
    agree) or a miss (not flagged, but gold and policy actually disagree).
    Both should be empty on a healthy corpus -- test_policy_agreement.py
    already guarantees this at the template level, so a nonempty result
    here points at a generation bug, not a template-authoring one."""
    n_unresolvable = 0
    n_agrees_with_policy = 0
    n_diverges_from_policy = 0
    n_template_declares_divergence = 0
    n_consistent = 0
    cases: list[DivergenceCase] = []

    for t in threads:
        template = templates.get(t.template_id)
        row = gold_row_for_cell(template, t.cell) if template is not None else None
        if row is None:
            n_unresolvable += 1
            continue

        if row.policy_divergence:
            n_template_declares_divergence += 1

        policy = route_rfi(
            rfi_type=t.cell.rfi_type,
            primary_discipline=t.cell.primary_discipline,
            cost_impact=t.cell.cost_impact,
            schedule_impact=t.cell.schedule_impact,
            urgency=t.cell.urgency,
        )
        reviewer_matches = t.label.assigned_reviewer == policy.assigned_reviewer
        escalation_matches = t.label.escalation == policy.escalation
        actual_divergence = not (reviewer_matches and escalation_matches)

        if actual_divergence:
            n_diverges_from_policy += 1
        else:
            n_agrees_with_policy += 1

        if row.policy_divergence and not actual_divergence:
            kind = "false_positive_divergence_flag"
        elif not row.policy_divergence and actual_divergence:
            kind = "missed_divergence_flag"
        else:
            n_consistent += 1
            continue

        cases.append(
            DivergenceCase(
                thread_id=t.thread_id,
                template_id=t.template_id,
                cell=t.cell.to_dict(),
                gold_assigned_reviewer=t.label.assigned_reviewer,
                gold_escalation=t.label.escalation,
                policy_assigned_reviewer=policy.assigned_reviewer,
                policy_escalation=policy.escalation,
                row_policy_divergence=row.policy_divergence,
                actual_divergence=actual_divergence,
                kind=kind,
            )
        )

    n = len(threads)
    n_false_positive = sum(1 for c in cases if c.kind == "false_positive_divergence_flag")
    n_missed = sum(1 for c in cases if c.kind == "missed_divergence_flag")

    section = AuditSection(
        name="divergence_census",
        summary={
            "n_threads": n,
            "n_unresolvable": n_unresolvable,
            "n_agrees_with_policy": n_agrees_with_policy,
            "n_diverges_from_policy": n_diverges_from_policy,
            "policy_agreement_rate": n_agrees_with_policy / n if n else float("nan"),
            "n_template_declares_divergence": n_template_declares_divergence,
            "n_consistent_with_template_flag": n_consistent,
            "n_false_positive": n_false_positive,
            "n_missed": n_missed,
        },
    )
    return section, cases


# ---------------------------------------------------------------------------
# S6: confidence-gate interaction
# ---------------------------------------------------------------------------


def gate_interaction(threads: list[CorpusThread]) -> AuditSection:
    """Runs gating.gate() over the corpus's gold labels and cross-tabs the
    result against the routing policy: which rule fired for each gated
    thread, and -- since gate() always flags every escalation=True label
    (see gating.py's REQUIRED_NONEMPTY_FIELDS/escalation check) -- whether
    every escalated gold label was in fact gated. A nonzero
    escalated_but_not_flagged is an invariant break in gating.gate() itself,
    not a corpus data-quality issue, and is checked under --strict."""
    labels = [t.label for t in threads]
    _, flagged = gate(labels)
    flagged_by_thread = {f.thread_id: f for f in flagged}

    rule_id_by_thread: dict[str, str] = {}
    for t in threads:
        decision = route_rfi(
            t.label.rfi_type, t.label.primary_discipline, t.label.cost_impact, t.label.schedule_impact, t.label.urgency
        )
        rule_id_by_thread[t.thread_id] = decision.rule_id

    flagged_rule_counts: Counter[str] = Counter(rule_id_by_thread[tid] for tid in flagged_by_thread)

    escalated_threads = [t for t in threads if t.label.escalation]
    n_escalated = len(escalated_threads)
    n_escalated_and_flagged = sum(1 for t in escalated_threads if t.thread_id in flagged_by_thread)

    return AuditSection(
        name="gate_interaction",
        summary={
            "n_threads": len(threads),
            "n_flagged": len(flagged),
            "n_flagged_for_escalation": sum(
                1 for f in flagged if any("escalation flagged" in r for r in f.reasons)
            ),
            "n_escalated": n_escalated,
            "n_escalated_and_flagged": n_escalated_and_flagged,
            "escalated_but_not_flagged": n_escalated - n_escalated_and_flagged,
        },
        details={"flagged_rule_counts": dict(flagged_rule_counts)},
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_audit(corpus_dir: Path) -> AuditReport:
    threads = load_corpus(corpus_dir)
    templates = load_templates()

    s1 = rule_inventory_and_reachability()
    s2 = rule_hit_counts(threads)
    s3 = escalation_consistency(threads)
    s4 = rule_contribution(threads)
    s5, divergence_cases = divergence_census(threads, templates)
    s6 = gate_interaction(threads)

    sections = {s.name: s for s in (s1, s2, s3, s4, s5, s6)}

    strict_failures: list[str] = []
    if s1.summary["unreachable_rules"]:
        strict_failures.append(
            f"rule_inventory_and_reachability: unreachable rules {s1.summary['unreachable_rules']}"
        )
    if s3.summary["n_mismatch"]:
        strict_failures.append(
            f"escalation_consistency: {s3.summary['n_mismatch']} gold labels disagree with escalation_for()"
        )
    if s5.summary["n_unresolvable"]:
        strict_failures.append(
            f"divergence_census: {s5.summary['n_unresolvable']} threads had no resolvable gold_row_for_cell()"
        )
    if s5.summary["n_false_positive"] or s5.summary["n_missed"]:
        strict_failures.append(
            f"divergence_census: {s5.summary['n_false_positive']} false-positive and "
            f"{s5.summary['n_missed']} missed policy_divergence flags"
        )
    if s6.summary["escalated_but_not_flagged"]:
        strict_failures.append(
            f"gate_interaction: {s6.summary['escalated_but_not_flagged']} escalated gold labels were not gated"
        )

    return AuditReport(
        corpus_dir=str(corpus_dir),
        n_threads=len(threads),
        sections=sections,
        divergence_cases=divergence_cases,
        strict_failures=strict_failures,
    )


def _to_jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (frozenset, set)):
        return sorted(_to_jsonable(v) for v in obj)
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    return obj


def write_json_report(report: AuditReport, path: Path) -> None:
    path.write_text(json.dumps(_to_jsonable(report), indent=2, default=str), encoding="utf-8")


def _fmt(x: object) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if x != x:
            return "n/a"
        return f"{x:.4f}"
    return str(x)


def write_markdown_report(report: AuditReport, path: Path) -> None:
    lines = [
        "# Routing Policy Audit Report",
        "",
        f"Corpus: `{report.corpus_dir}` | Threads: {report.n_threads}",
        "",
    ]
    for name, section in report.sections.items():
        lines.append(f"## {name}")
        lines.append("")
        for k, v in section.summary.items():
            if isinstance(v, dict):
                lines.append(f"- **{k}**: {v}")
            else:
                lines.append(f"- **{k}**: {_fmt(v)}")
        lines.append("")

    lines.append(f"## Divergence cases needing review: {len(report.divergence_cases)}")
    lines.append("")
    for c in report.divergence_cases:
        lines.append(
            f"- {c.thread_id} ({c.template_id}, {c.kind}): "
            f"gold=({c.gold_assigned_reviewer!r}, escalation={c.gold_escalation!r}) vs. "
            f"policy=({c.policy_assigned_reviewer!r}, escalation={c.policy_escalation!r}) "
            f"[row.policy_divergence={c.row_policy_divergence!r}]"
        )
    lines.append("")

    if report.strict_failures:
        lines.append("## Strict-mode failures")
        for f in report.strict_failures:
            lines.append(f"- {f}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit routing_policy.py against the generated gold-label corpus.")
    parser.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parent.parent / "pilot-data" / "corpus")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "pilot-data" / "policy_audit")
    parser.add_argument("--strict", action="store_true", help="exit nonzero if any invariant check fails")
    args = parser.parse_args(argv)

    report = run_audit(args.corpus)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json_report(report, args.out / "policy_audit_report.json")
    write_markdown_report(report, args.out / "policy_audit_report.md")
    print(
        f"Wrote policy audit report ({report.n_threads} threads, "
        f"{len(report.divergence_cases)} divergence cases, {len(report.strict_failures)} strict failures) to {args.out}"
    )

    if args.strict and report.strict_failures:
        for f in report.strict_failures:
            print(f"STRICT FAILURE: {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
