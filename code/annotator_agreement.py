"""Cross-annotator (or annotator-vs-pipeline) agreement for RFI labels.

Aligns two or more label sets on shared thread_ids, then scores agreement
per RFILabel field using the kind of metric that field's type actually
supports: Krippendorff's alpha (nominal/ordinal) for closed-vocabulary
fields, MASI/set-PRF for list-valued fields, cosine similarity over
TF-IDF/BGE embeddings for free text, and day-offset for the resolved
deadline date. Never uses routing_policy.route_rfi() as a stand-in for gold
(see that module's docstring) — policy_agreement() instead compares each
annotator's own values against what the policy deterministically derives
from that same annotator's own fields, which is a validity/consistency
check, not a substitute label source.

Run as a script:

    python code/annotator_agreement.py --corpus pilot-data/corpus \
        --out pilot-data/agreement
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from agreement_metrics import (
    AlphaResult,
    krippendorff_alpha,
    masi_distance,
    percent_agreement,
    set_micro_prf,
)
from embeddings import build_similarity
from routing_policy import route_from_fields
from schema import RFILabel, validate_label
from vocabulary import URGENCY_TIERS

_SENTINEL = "—"

_POLICY_INPUT_FIELDS = ("rfi_type", "primary_discipline", "cost_impact", "schedule_impact", "urgency")


# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str  # bookkeeping | nominal | ordinal | boolean | set | text | date | constant
    reportable: bool = True
    note: str = ""


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("thread_id", "bookkeeping", reportable=False, note="join key, not a label"),
    FieldSpec("rfi_type", "nominal"),
    FieldSpec("primary_discipline", "nominal"),
    FieldSpec("secondary_disciplines", "set"),
    FieldSpec("csi_division", "nominal", note="conditional: units where both annotators say General/'—' are excluded as uninformative"),
    FieldSpec("urgency", "ordinal", note="krippendorff_alpha(metric='ordinal', categories=URGENCY_TIERS)"),
    FieldSpec("question_summary", "text"),
    FieldSpec("referenced_documents", "set"),
    FieldSpec("proposed_solution", "text", note="sentinel '—' split from cosine similarity among mutually-non-sentinel units"),
    FieldSpec("cost_impact", "boolean"),
    FieldSpec("schedule_impact", "boolean"),
    FieldSpec("answer_in_documents", "boolean"),
    FieldSpec("deadline_text", "text"),
    FieldSpec("assigned_reviewer", "nominal"),
    FieldSpec("routing_rationale", "text"),
    FieldSpec("escalation", "boolean"),
    FieldSpec("confidence", "constant", reportable=False, note="continuous pipeline/annotator confidence, not a label to score agreement on"),
    FieldSpec("status", "constant", reportable=False, note="workflow bookkeeping, not a label"),
    FieldSpec("annotator", "bookkeeping", reportable=False, note="identifies the rater, not a label"),
    FieldSpec("deadline_iso", "date", note="day-offset, conditional on both annotators resolving to a date"),
    FieldSpec("deadline_resolution_status", "nominal"),
)

_FIELD_SPEC_BY_NAME: dict[str, FieldSpec] = {s.name: s for s in FIELD_SPECS}

_rfilabel_field_names = {f.name for f in fields(RFILabel)}
if _rfilabel_field_names != set(_FIELD_SPEC_BY_NAME):
    missing = _rfilabel_field_names - set(_FIELD_SPEC_BY_NAME)
    extra = set(_FIELD_SPEC_BY_NAME) - _rfilabel_field_names
    raise RuntimeError(
        f"FIELD_SPECS is out of sync with RFILabel fields: missing={missing} extra={extra}"
    )


def _fmt(x: object) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and np.isnan(x):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignedCorpus:
    annotator_ids: list[str]
    thread_ids: list[str]
    labels: dict[str, dict[str, RFILabel]]
    framing: str


def discover_annotator_subdirs(corpus_dir: Path) -> dict[str, Path]:
    """Maps annotator id -> label directory. `labels/` is "gold"; a sibling
    `labels_<suffix>/` directory is annotator id `<suffix>` (e.g.
    `labels_annotator_b` -> "annotator_b", `labels_pipeline` -> "pipeline")."""
    result: dict[str, Path] = {}
    labels_dir = corpus_dir / "labels"
    if labels_dir.is_dir():
        result["gold"] = labels_dir
    for candidate in sorted(corpus_dir.glob("labels_*")):
        if candidate.is_dir():
            annotator_id = candidate.name[len("labels_"):]
            result[annotator_id] = candidate
    if not result:
        raise FileNotFoundError(f"no labels/ or labels_*/ directory found under {corpus_dir}")
    return result


def load_annotator_labels(label_dir: Path, expected_annotator: str) -> dict[str, RFILabel]:
    """Reads every *.json label file in label_dir. Hard-errors if a label's
    own .annotator field disagrees with the directory it was found in
    (silently trusting the directory would let a copy-pasted label from the
    wrong annotator pass as this annotator's data), or if the label fails
    validate_label() (an invalid label can't be trusted for agreement math:
    e.g. a mis-vocabularied urgency value would blow up urgency_rank)."""
    labels: dict[str, RFILabel] = {}
    for path in sorted(label_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        label = RFILabel(**data)
        if label.annotator != expected_annotator:
            raise ValueError(
                f"{path}: label.annotator={label.annotator!r} does not match "
                f"expected annotator {expected_annotator!r} for directory {label_dir}"
            )
        errors = validate_label(label)
        if errors:
            raise ValueError(f"{path}: invalid label for annotator {expected_annotator!r}: {errors}")
        labels[label.thread_id] = label
    return labels


def _determine_framing(annotator_ids: list[str]) -> str:
    if len(annotator_ids) == 2 and ("pipeline" in annotator_ids or "predictions" in annotator_ids):
        return "recoverability"
    return "inter_annotator"


def align_annotators(corpus_dir: Path) -> AlignedCorpus:
    subdirs = discover_annotator_subdirs(corpus_dir)
    annotator_ids = sorted(subdirs)
    per_annotator_labels = {aid: load_annotator_labels(subdirs[aid], aid) for aid in annotator_ids}

    thread_id_sets = [set(labels) for labels in per_annotator_labels.values()]
    shared_thread_ids = sorted(set.intersection(*thread_id_sets)) if thread_id_sets else []

    labels = {
        aid: {tid: per_annotator_labels[aid][tid] for tid in shared_thread_ids}
        for aid in annotator_ids
    }

    return AlignedCorpus(
        annotator_ids=annotator_ids,
        thread_ids=shared_thread_ids,
        labels=labels,
        framing=_determine_framing(annotator_ids),
    )


def _subset(aligned: AlignedCorpus, thread_ids: list[str]) -> AlignedCorpus:
    labels = {aid: {tid: aligned.labels[aid][tid] for tid in thread_ids} for aid in aligned.annotator_ids}
    return AlignedCorpus(
        annotator_ids=aligned.annotator_ids,
        thread_ids=list(thread_ids),
        labels=labels,
        framing=aligned.framing,
    )


def field_values(aligned: AlignedCorpus, field_name: str) -> dict[str, list]:
    return {
        aid: [getattr(aligned.labels[aid][tid], field_name) for tid in aligned.thread_ids]
        for aid in aligned.annotator_ids
    }


# ---------------------------------------------------------------------------
# Field-level agreement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldAgreementResult:
    field: str
    kind: str
    reportable: bool
    n_units: int
    n_applicable: int
    summary: dict
    note: str = ""


def _units_for_krippendorff(aligned: AlignedCorpus, values: dict[str, list]) -> list[list]:
    return [[values[aid][i] for aid in aligned.annotator_ids] for i in range(len(aligned.thread_ids))]


def _nominal_field_agreement(aligned: AlignedCorpus, spec: FieldSpec) -> FieldAgreementResult:
    values = field_values(aligned, spec.name)
    units = _units_for_krippendorff(aligned, values)
    alpha_res = krippendorff_alpha(units, metric="nominal")
    pct = None
    if len(aligned.annotator_ids) == 2:
        a_id, b_id = aligned.annotator_ids
        pct = percent_agreement(values[a_id], values[b_id])
    return FieldAgreementResult(
        field=spec.name, kind=spec.kind, reportable=spec.reportable,
        n_units=len(aligned.thread_ids), n_applicable=alpha_res.n_units,
        summary={"alpha": alpha_res.alpha, "undefined_reason": alpha_res.undefined_reason, "percent_agreement": pct},
        note=spec.note,
    )


def _csi_division_agreement(aligned: AlignedCorpus, spec: FieldSpec) -> FieldAgreementResult:
    disc = field_values(aligned, "primary_discipline")
    csi = field_values(aligned, spec.name)
    units = []
    n_excluded = 0
    for i in range(len(aligned.thread_ids)):
        if all(disc[aid][i] == "General" for aid in aligned.annotator_ids):
            n_excluded += 1
            continue
        units.append([csi[aid][i] for aid in aligned.annotator_ids])
    alpha_res = krippendorff_alpha(units, metric="nominal")
    pct = None
    if len(aligned.annotator_ids) == 2 and units:
        pct = percent_agreement([u[0] for u in units], [u[1] for u in units])
    return FieldAgreementResult(
        field=spec.name, kind=spec.kind, reportable=spec.reportable,
        n_units=len(aligned.thread_ids), n_applicable=alpha_res.n_units,
        summary={
            "alpha": alpha_res.alpha, "undefined_reason": alpha_res.undefined_reason,
            "percent_agreement": pct, "n_excluded_general_general": n_excluded,
        },
        note=spec.note,
    )


def _ordinal_field_agreement(aligned: AlignedCorpus, spec: FieldSpec) -> FieldAgreementResult:
    values = field_values(aligned, spec.name)
    units = _units_for_krippendorff(aligned, values)
    alpha_res = krippendorff_alpha(units, metric="ordinal", categories=URGENCY_TIERS)
    pct = None
    if len(aligned.annotator_ids) == 2:
        a_id, b_id = aligned.annotator_ids
        pct = percent_agreement(values[a_id], values[b_id])
    return FieldAgreementResult(
        field=spec.name, kind=spec.kind, reportable=spec.reportable,
        n_units=len(aligned.thread_ids), n_applicable=alpha_res.n_units,
        summary={"alpha": alpha_res.alpha, "undefined_reason": alpha_res.undefined_reason, "percent_agreement": pct},
        note=spec.note,
    )


def _set_field_agreement(aligned: AlignedCorpus, spec: FieldSpec) -> FieldAgreementResult:
    if len(aligned.annotator_ids) != 2:
        raise NotImplementedError(f"set-valued field agreement ({spec.name}) only supports exactly 2 annotators")
    a_id, b_id = aligned.annotator_ids
    values = field_values(aligned, spec.name)
    pairs = [(frozenset(a), frozenset(b)) for a, b in zip(values[a_id], values[b_id])]
    masi = [masi_distance(a, b) for a, b in pairs]
    exact = sum(1 for a, b in pairs if a == b)
    prf = set_micro_prf(pairs)
    n = len(pairs)
    return FieldAgreementResult(
        field=spec.name, kind=spec.kind, reportable=spec.reportable,
        n_units=n, n_applicable=n,
        summary={
            "mean_masi": float(np.mean(masi)) if masi else float("nan"),
            "exact_match_rate": exact / n if n else float("nan"),
            "micro_precision": prf["precision"], "micro_recall": prf["recall"], "micro_f1": prf["f1"],
        },
        note=spec.note,
    )


def _text_field_agreement(aligned: AlignedCorpus, spec: FieldSpec, similarity_model: str) -> FieldAgreementResult:
    if len(aligned.annotator_ids) != 2:
        raise NotImplementedError(f"text field agreement ({spec.name}) only supports exactly 2 annotators")
    a_id, b_id = aligned.annotator_ids
    values = field_values(aligned, spec.name)
    sim_model = build_similarity(similarity_model)
    sims = [
        float(sim_model.pairwise([a_text], [b_text])[0][0])
        for a_text, b_text in zip(values[a_id], values[b_id])
    ]
    n = len(sims)
    return FieldAgreementResult(
        field=spec.name, kind=spec.kind, reportable=spec.reportable,
        n_units=n, n_applicable=n,
        summary={
            "mean_cosine_similarity": float(np.mean(sims)) if sims else float("nan"),
            "median_cosine_similarity": float(np.median(sims)) if sims else float("nan"),
        },
        note=spec.note,
    )


def _sentinel_text_field_agreement(aligned: AlignedCorpus, spec: FieldSpec, similarity_model: str) -> FieldAgreementResult:
    if len(aligned.annotator_ids) != 2:
        raise NotImplementedError(f"text field agreement ({spec.name}) only supports exactly 2 annotators")
    a_id, b_id = aligned.annotator_ids
    values = field_values(aligned, spec.name)
    n = len(aligned.thread_ids)
    both_sentinel = 0
    sentinel_mismatch = 0
    both_real: list[tuple[str, str]] = []
    for a_text, b_text in zip(values[a_id], values[b_id]):
        a_is, b_is = (a_text == _SENTINEL), (b_text == _SENTINEL)
        if a_is and b_is:
            both_sentinel += 1
        elif a_is != b_is:
            sentinel_mismatch += 1
        else:
            both_real.append((a_text, b_text))

    sim_model = build_similarity(similarity_model)
    sims = [float(sim_model.pairwise([a], [b])[0][0]) for a, b in both_real]
    return FieldAgreementResult(
        field=spec.name, kind=spec.kind, reportable=spec.reportable,
        n_units=n, n_applicable=len(both_real),
        summary={
            "sentinel_agreement_rate": (both_sentinel + len(both_real)) / n if n else float("nan"),
            "n_both_sentinel": both_sentinel,
            "n_sentinel_mismatch": sentinel_mismatch,
            "n_both_real": len(both_real),
            "mean_cosine_similarity_when_both_real": float(np.mean(sims)) if sims else float("nan"),
        },
        note=spec.note,
    )


def _date_field_agreement(aligned: AlignedCorpus, spec: FieldSpec) -> FieldAgreementResult:
    if len(aligned.annotator_ids) != 2:
        raise NotImplementedError(f"date field agreement ({spec.name}) only supports exactly 2 annotators")
    from datetime import date

    a_id, b_id = aligned.annotator_ids
    status = field_values(aligned, "deadline_resolution_status")
    iso = field_values(aligned, spec.name)
    n = len(aligned.thread_ids)
    offsets: list[int] = []
    n_status_mismatch = 0
    for i in range(n):
        sa, sb = status[a_id][i], status[b_id][i]
        if sa == "resolved_to_date" and sb == "resolved_to_date":
            da = date.fromisoformat(iso[a_id][i])
            db = date.fromisoformat(iso[b_id][i])
            offsets.append(abs((da - db).days))
        elif sa != sb:
            n_status_mismatch += 1
    return FieldAgreementResult(
        field=spec.name, kind=spec.kind, reportable=spec.reportable,
        n_units=n, n_applicable=len(offsets),
        summary={
            "mean_abs_day_offset": float(np.mean(offsets)) if offsets else float("nan"),
            "median_abs_day_offset": float(np.median(offsets)) if offsets else float("nan"),
            "exact_match_rate": (sum(1 for o in offsets if o == 0) / len(offsets)) if offsets else float("nan"),
            "n_status_mismatch": n_status_mismatch,
        },
        note=spec.note,
    )


def _constant_field_agreement(aligned: AlignedCorpus, spec: FieldSpec) -> FieldAgreementResult:
    values = field_values(aligned, spec.name)
    n = len(aligned.thread_ids)
    mismatches = 0
    for i in range(n):
        distinct = {values[aid][i] for aid in aligned.annotator_ids}
        if len(distinct) > 1:
            mismatches += 1
    note = spec.note
    if mismatches:
        note = (
            f"{spec.name} is bookkeeping, not scored for agreement, but differed across "
            f"annotators on {mismatches}/{n} units"
        )
    return FieldAgreementResult(
        field=spec.name, kind=spec.kind, reportable=False,
        n_units=n, n_applicable=n,
        summary={"n_mismatched": mismatches},
        note=note,
    )


def field_agreement(aligned: AlignedCorpus, field_name: str, *, similarity_model: str = "tfidf") -> FieldAgreementResult | None:
    spec = _FIELD_SPEC_BY_NAME[field_name]
    if spec.kind == "bookkeeping":
        return None
    if spec.kind in ("nominal", "boolean"):
        if spec.name == "csi_division":
            return _csi_division_agreement(aligned, spec)
        return _nominal_field_agreement(aligned, spec)
    if spec.kind == "ordinal":
        return _ordinal_field_agreement(aligned, spec)
    if spec.kind == "set":
        return _set_field_agreement(aligned, spec)
    if spec.kind == "text":
        if spec.name == "proposed_solution":
            return _sentinel_text_field_agreement(aligned, spec, similarity_model)
        return _text_field_agreement(aligned, spec, similarity_model)
    if spec.kind == "date":
        return _date_field_agreement(aligned, spec)
    if spec.kind == "constant":
        return _constant_field_agreement(aligned, spec)
    raise ValueError(f"unhandled field kind: {spec.kind!r}")


# ---------------------------------------------------------------------------
# Policy decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyAgreementResult:
    n_units: int
    input_agreement_rate: float
    reviewer_alpha_from_own_inputs: AlphaResult
    escalation_percent_agreement_from_own_inputs: float
    policy_validity_rate: dict[str, float]
    escalation_validity_rate: dict[str, float]
    n_input_agreement_units: int
    reviewer_alpha_given_input_agreement: AlphaResult


def policy_agreement(aligned: AlignedCorpus) -> PolicyAgreementResult:
    """Four-part decomposition of routing agreement, using
    routing_policy.route_from_fields() only as a deterministic function of
    each annotator's *own* fields (never as a gold-label substitute):
    (a) cross-annotator alpha on each annotator's policy-derived reviewer,
    computed from that annotator's own classification fields;
    (b) each annotator's policy-validity rate (their gold assigned_reviewer
    vs. what the policy derives from their own fields);
    (c) each annotator's escalation-validity rate (same idea for the
    escalation flag);
    (d) reviewer alpha restricted to units where both annotators agree on
    all five route_rfi inputs, isolating whether assigned_reviewer is still
    subjective given identical inputs."""
    if len(aligned.annotator_ids) != 2:
        raise NotImplementedError("policy_agreement only supports exactly 2 annotators")
    a_id, b_id = aligned.annotator_ids
    tids = aligned.thread_ids
    n = len(tids)

    input_agree_mask = []
    policy_decisions = {a_id: [], b_id: []}
    for tid in tids:
        label_a = aligned.labels[a_id][tid]
        label_b = aligned.labels[b_id][tid]
        inputs_a = tuple(getattr(label_a, f) for f in _POLICY_INPUT_FIELDS)
        inputs_b = tuple(getattr(label_b, f) for f in _POLICY_INPUT_FIELDS)
        input_agree_mask.append(inputs_a == inputs_b)
        policy_decisions[a_id].append(route_from_fields(label_a))
        policy_decisions[b_id].append(route_from_fields(label_b))

    input_agreement_rate = (sum(input_agree_mask) / n) if n else float("nan")

    reviewer_units = [
        [policy_decisions[a_id][i].assigned_reviewer, policy_decisions[b_id][i].assigned_reviewer]
        for i in range(n)
    ]
    reviewer_alpha = krippendorff_alpha(reviewer_units, metric="nominal")

    esc_a = [d.escalation for d in policy_decisions[a_id]]
    esc_b = [d.escalation for d in policy_decisions[b_id]]
    escalation_pct = percent_agreement(esc_a, esc_b)

    policy_validity_rate: dict[str, float] = {}
    escalation_validity_rate: dict[str, float] = {}
    for aid in (a_id, b_id):
        gold_reviewer = [aligned.labels[aid][tid].assigned_reviewer for tid in tids]
        policy_reviewer = [d.assigned_reviewer for d in policy_decisions[aid]]
        policy_validity_rate[aid] = percent_agreement(gold_reviewer, policy_reviewer)
        gold_escalation = [aligned.labels[aid][tid].escalation for tid in tids]
        policy_escalation = [d.escalation for d in policy_decisions[aid]]
        escalation_validity_rate[aid] = percent_agreement(gold_escalation, policy_escalation)

    conditional_units = [
        [aligned.labels[a_id][tids[i]].assigned_reviewer, aligned.labels[b_id][tids[i]].assigned_reviewer]
        for i in range(n) if input_agree_mask[i]
    ]
    reviewer_alpha_given_agreement = krippendorff_alpha(conditional_units, metric="nominal")

    return PolicyAgreementResult(
        n_units=n,
        input_agreement_rate=input_agreement_rate,
        reviewer_alpha_from_own_inputs=reviewer_alpha,
        escalation_percent_agreement_from_own_inputs=escalation_pct,
        policy_validity_rate=policy_validity_rate,
        escalation_validity_rate=escalation_validity_rate,
        n_input_agreement_units=len(conditional_units),
        reviewer_alpha_given_input_agreement=reviewer_alpha_given_agreement,
    )


# ---------------------------------------------------------------------------
# Stratified agreement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StratumResult:
    axis: str
    value: str
    n_units: int
    field_results: dict[str, FieldAgreementResult]


def _load_provenance_axis(corpus_dir: Path, thread_ids: list[str], axis: str) -> dict[str, str]:
    prov_dir = corpus_dir / "provenance"
    result = {}
    for tid in thread_ids:
        data = json.loads((prov_dir / f"{tid}.json").read_text(encoding="utf-8"))
        result[tid] = data["cell"][axis]
    return result


def stratified_agreement(
    aligned: AlignedCorpus,
    corpus_dir: Path,
    axis: str,
    fields_to_score: tuple[str, ...] = ("rfi_type", "urgency", "assigned_reviewer"),
) -> list[StratumResult]:
    """Groups units by a provenance-recorded generation axis (e.g.
    'adversarial', 'rfi_type') — not by any annotator's own labeled value —
    so a systematic weak spot (e.g. worse agreement specifically on
    unresolvable-deadline RFIs) shows up even if annotators themselves
    disagree about which stratum a thread belongs to."""
    axis_values = _load_provenance_axis(corpus_dir, aligned.thread_ids, axis)
    groups: dict[str, list[str]] = {}
    for tid, v in axis_values.items():
        groups.setdefault(v, []).append(tid)

    results = []
    for value, tids in sorted(groups.items()):
        sub = _subset(aligned, sorted(tids))
        field_results = {f: field_agreement(sub, f) for f in fields_to_score}
        results.append(StratumResult(axis=axis, value=value, n_units=len(tids), field_results=field_results))
    return results


# ---------------------------------------------------------------------------
# Disagreements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Disagreement:
    thread_id: str
    field: str
    values: dict[str, object]


def disagreements(aligned: AlignedCorpus, field_names: tuple[str, ...] | None = None) -> list[Disagreement]:
    if field_names is None:
        field_names = tuple(s.name for s in FIELD_SPECS if s.kind not in ("bookkeeping", "constant"))
    out: list[Disagreement] = []
    for tid in aligned.thread_ids:
        for f_name in field_names:
            spec = _FIELD_SPEC_BY_NAME[f_name]
            values = {aid: getattr(aligned.labels[aid][tid], f_name) for aid in aligned.annotator_ids}
            if spec.kind == "set":
                distinct = {frozenset(v) for v in values.values()}
            else:
                distinct = {repr(v) for v in values.values()}
            if len(distinct) > 1:
                out.append(Disagreement(thread_id=tid, field=f_name, values=values))
    return out


# ---------------------------------------------------------------------------
# Report assembly and I/O
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgreementReport:
    framing: str
    annotator_ids: list[str]
    n_units: int
    field_results: dict[str, FieldAgreementResult]
    policy: PolicyAgreementResult | None
    stratified: dict[str, list[StratumResult]]
    disagreements: list[Disagreement]
    constant_field_warnings: list[str]


def build_report(
    corpus_dir: Path,
    *,
    similarity_model: str = "tfidf",
    stratify_axes: tuple[str, ...] = ("adversarial", "rfi_type"),
) -> AgreementReport:
    aligned = align_annotators(corpus_dir)
    two_rater = len(aligned.annotator_ids) == 2

    field_results: dict[str, FieldAgreementResult] = {}
    warnings: list[str] = []
    for spec in FIELD_SPECS:
        if spec.kind == "bookkeeping":
            continue
        res = field_agreement(aligned, spec.name, similarity_model=similarity_model)
        field_results[spec.name] = res
        if spec.kind == "constant" and res.summary.get("n_mismatched"):
            warnings.append(res.note)

    policy = policy_agreement(aligned) if two_rater else None
    stratified = {axis: stratified_agreement(aligned, corpus_dir, axis) for axis in stratify_axes} if two_rater else {}
    diffs = disagreements(aligned) if two_rater else []

    return AgreementReport(
        framing=aligned.framing,
        annotator_ids=aligned.annotator_ids,
        n_units=len(aligned.thread_ids),
        field_results=field_results,
        policy=policy,
        stratified=stratified,
        disagreements=diffs,
        constant_field_warnings=warnings,
    )


def _to_jsonable(obj):
    if is_dataclass_instance(obj):
        return {k: _to_jsonable(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (frozenset, set)):
        return sorted(_to_jsonable(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def is_dataclass_instance(obj) -> bool:
    from dataclasses import is_dataclass
    return is_dataclass(obj) and not isinstance(obj, type)


def write_json_report(report: AgreementReport, path: Path) -> None:
    path.write_text(json.dumps(_to_jsonable(report), indent=2, default=str), encoding="utf-8")


def write_markdown_report(report: AgreementReport, path: Path) -> None:
    lines = [
        "# Annotator Agreement Report",
        "",
        f"Framing: **{report.framing}** | Annotators: {', '.join(report.annotator_ids)} | Units: {report.n_units}",
        "",
        "## Field-level agreement",
        "",
        "| Field | Kind | Reportable | n_applicable | Summary |",
        "|---|---|---|---|---|",
    ]
    for name, res in report.field_results.items():
        summary_str = "; ".join(f"{k}={_fmt(v)}" for k, v in res.summary.items())
        lines.append(f"| {name} | {res.kind} | {res.reportable} | {res.n_applicable} | {summary_str} |")
    lines.append("")

    if report.constant_field_warnings:
        lines.append("## Warnings")
        for w in report.constant_field_warnings:
            lines.append(f"- {w}")
        lines.append("")

    if report.policy:
        p = report.policy
        lines.append("## Policy-decomposition agreement")
        lines.append(f"- Input agreement rate (5 policy fields): {_fmt(p.input_agreement_rate)} ({p.n_units} units)")
        lines.append(f"- Reviewer alpha from each annotator's own inputs: {_fmt(p.reviewer_alpha_from_own_inputs.alpha)}")
        lines.append(f"- Escalation agreement from each annotator's own inputs: {_fmt(p.escalation_percent_agreement_from_own_inputs)}")
        for aid, v in p.policy_validity_rate.items():
            lines.append(f"- Policy-validity (gold vs. policy(own inputs)) for {aid}: {_fmt(v)}")
        for aid, v in p.escalation_validity_rate.items():
            lines.append(f"- Escalation-validity for {aid}: {_fmt(v)}")
        lines.append(
            f"- Reviewer alpha given input agreement ({p.n_input_agreement_units} units): "
            f"{_fmt(p.reviewer_alpha_given_input_agreement.alpha)}"
        )
        lines.append("")

    for axis, strata in report.stratified.items():
        lines.append(f"## Stratified by `{axis}`")
        lines.append("")
        for s in strata:
            lines.append(f"### {s.value} (n={s.n_units})")
            for fname, res in s.field_results.items():
                summary_str = "; ".join(f"{k}={_fmt(v)}" for k, v in res.summary.items())
                lines.append(f"- {fname}: {summary_str}")
            lines.append("")

    lines.append(f"## Disagreements: {len(report.disagreements)} field-level mismatches")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_adjudication_worklist(disagreement_list: list[Disagreement], path: Path) -> None:
    lines = [
        "# Adjudication Worklist",
        "",
        f"{len(disagreement_list)} field-level disagreements needing review.",
        "",
    ]
    by_thread: dict[str, list[Disagreement]] = {}
    for d in disagreement_list:
        by_thread.setdefault(d.thread_id, []).append(d)
    for tid, ds in sorted(by_thread.items()):
        lines.append(f"## {tid}")
        for d in ds:
            values_str = "; ".join(f"{aid}={v!r}" for aid, v in d.values.items())
            lines.append(f"- **{d.field}**: {values_str}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score cross-annotator agreement over an RFI label corpus.")
    parser.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parent.parent / "pilot-data" / "corpus")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "pilot-data" / "agreement")
    parser.add_argument("--similarity-model", choices=["tfidf", "bge"], default="tfidf")
    args = parser.parse_args()

    report = build_report(args.corpus, similarity_model=args.similarity_model)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json_report(report, args.out / "agreement_report.json")
    write_markdown_report(report, args.out / "agreement_report.md")
    write_adjudication_worklist(report.disagreements, args.out / "adjudication_worklist.md")
    print(f"Wrote agreement report ({report.framing}, {report.n_units} units) to {args.out}")


if __name__ == "__main__":
    main()
