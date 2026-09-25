"""Scores predicted RFI labels (an LLM pipeline run, or a hand-labeled
second annotator) against the gold-label corpus across five routing
evaluation arms, plus a confidence-gated accuracy breakdown.

This is the prediction-scoring counterpart to policy_audit.py, which audits
routing_policy.py against the *gold* corpus alone. The two must stay
independent passes over the same corpus so either can be run and checked
without the other -- see policy_audit.py's own module docstring.

Import-decoupling contract: this module may only import project modules from
{schema, gating, routing_policy, vocabulary, agreement_metrics}. It must
never import policy_audit or stratification. In particular, split
membership ("dev" vs "eval") is read directly from each thread's
provenance/*.json as a plain dict (cell["split"]) rather than via
stratification.Cell -- DEV_SPLIT/EVAL_SPLIT below are the same literal
strings as templates.DEV_POOL/EVAL_POOL, duplicated here (not imported) so
this module never has to import templates or stratification to know them.

Five evaluation arms, all built on routing_policy.route_from_fields()/the
label's own fields -- never a re-derivation of policy or escalation logic:

  - oracle:          route_from_fields(gold)      vs. gold   (the ceiling:
    how well the deterministic policy reproduces gold routing when given
    perfect classification fields; not expected to be 1.0, since some gold
    rows are deliberately policy-divergent -- see policy_audit.py's
    divergence_census. This arm's accuracy is the routing_eval analogue of
    that section's policy_agreement_rate, restricted to eval-split threads
    that have a matched prediction.)
  - composed:        route_from_fields(predicted) vs. gold   (the full
    pipeline: LLM classifies, policy routes)
  - direct:          predicted's own assigned_reviewer/escalation vs. gold
    (the LLM's raw routing guess, no policy involved)
  - policy_fidelity: route_from_fields(predicted) vs. route_from_fields(gold)
    (isolates classification-field noise from gold/policy divergence noise,
    by holding the policy law fixed on both sides)
  - majority:        a train/eval-separated naive baseline -- the single
    most common assigned_reviewer/escalation value among the DEV split's
    gold labels, applied uniformly to every EVAL split matched thread. Fit
    and scoring populations are disjoint, so this baseline is not
    circularly favorable.

Reviewer-field agreement is also reported as Cohen's kappa
(agreement_metrics.cohens_kappa), a chance-corrected metric, since raw
accuracy alone overstates agreement when one reviewer role dominates the
corpus (R4_discipline_default fires on 203/240 gold threads per
policy_audit_report.md). Escalation is imbalanced the same way (71/240
positive), so it gets a kappa too.

gating.gate() is run on the predicted labels for the composed and direct
arms (the two arms that actually consume LLM output) to split reviewer
accuracy into auto-approved vs. flagged subsets -- the human-in-the-loop
confidence-gating claim: flagged predictions should be *worse* on average,
which is the whole point of routing them to a human instead of auto-filing.

Run as a script:

    python code/routing_eval.py --corpus pilot-data/corpus \
        --predictions pilot-data/pipeline_run --annotator-id pipeline \
        --out pilot-data/routing_eval --strict
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path

from agreement_metrics import cohens_kappa
from gating import gate
from routing_policy import route_from_fields
from schema import RFILabel, read_label, validate_label

DEV_SPLIT = "dev"
EVAL_SPLIT = "eval"

GATE_THRESHOLD_DEFAULT = 0.6


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchedThread:
    thread_id: str
    split: str | None
    gold: RFILabel
    predicted: RFILabel


@dataclass(frozen=True)
class InvalidPrediction:
    """A predicted label that failed schema.validate_label() and was excluded
    from scoring rather than crashing the run. Gold labels are held to the
    stricter standard of raising immediately (see load_gold_labels) since
    gold is assumed already audited; predicted labels are untrusted LLM
    output, so an out-of-vocabulary field (e.g. urgency="high") is an
    expected, reportable outcome rather than a bug in this module."""

    thread_id: str
    path: str
    errors: list[str]


def load_gold_labels(corpus_dir: Path) -> dict[str, RFILabel]:
    """Gold labels are always the corpus's own labels/*.json -- the same
    ground truth policy_audit.py uses, never anything routing_policy.py
    derives (that would be circular). Raises if the directory is missing, a
    label fails schema.validate_label(), or two files share a thread_id --
    gold is assumed already audited, so any of these indicates a corpus bug
    that must not be silently absorbed into a "clean" report of zeros."""
    labels_dir = corpus_dir / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"gold labels directory not found: {labels_dir}")
    gold: dict[str, RFILabel] = {}
    for path in sorted(labels_dir.glob("*.json")):
        label = read_label(path)
        errors = validate_label(label)
        if errors:
            raise ValueError(f"invalid gold label at {path}: {errors}")
        if label.thread_id in gold:
            raise ValueError(f"duplicate gold thread_id {label.thread_id!r}: {path} collides with an earlier file")
        gold[label.thread_id] = label
    return gold


def load_predicted_labels(predictions_dir: Path, annotator_id: str) -> tuple[dict[str, RFILabel], list[InvalidPrediction]]:
    """Reads run_pipeline.py's output layout: <predictions_dir>/labels_{annotator_id}/*.json.
    Also usable to score a hand-labeled second annotator by pointing
    --predictions at that annotator's own output directory and
    --annotator-id at their id (e.g. "annotator_b").

    Unlike load_gold_labels, a predicted label that fails validate_label() --
    a plausible LLM output, e.g. an out-of-vocabulary urgency tier that would
    otherwise crash route_from_fields()'s urgency_rank() lookup deep inside
    scoring -- is excluded and reported in the returned InvalidPrediction
    list instead of raising, so one bad prediction doesn't abort the whole
    run. A duplicate thread_id across two files is reported the same way."""
    labels_dir = predictions_dir / f"labels_{annotator_id}"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"predictions directory not found: {labels_dir}")
    predicted: dict[str, RFILabel] = {}
    invalid: list[InvalidPrediction] = []
    for path in sorted(labels_dir.glob("*.json")):
        label = read_label(path)
        errors = validate_label(label)
        if errors:
            invalid.append(InvalidPrediction(thread_id=label.thread_id, path=str(path), errors=list(errors)))
            continue
        if label.thread_id in predicted:
            invalid.append(InvalidPrediction(thread_id=label.thread_id, path=str(path), errors=["duplicate thread_id"]))
            continue
        predicted[label.thread_id] = label
    return predicted, invalid


def load_splits(corpus_dir: Path) -> dict[str, str]:
    """Reads split membership ("dev"/"eval") straight out of each thread's
    provenance/*.json as a plain dict -- never via stratification.Cell, per
    this module's import-decoupling contract."""
    provenance_dir = corpus_dir / "provenance"
    splits: dict[str, str] = {}
    for path in sorted(provenance_dir.glob("*.json")):
        prov = json.loads(path.read_text(encoding="utf-8"))
        split = prov.get("cell", {}).get("split")
        if split is not None:
            splits[prov["thread_id"]] = split
    return splits


def match_threads(
    gold: dict[str, RFILabel], predicted: dict[str, RFILabel], splits: dict[str, str]
) -> tuple[list[MatchedThread], list[str], list[str]]:
    """Joins gold and predicted labels by thread_id. Threads present in gold
    but missing from predicted (e.g. the pipeline run only covered a subset)
    are reported as `missing`. Threads present in predicted but absent from
    gold (e.g. a stale prediction file, or --annotator-id pointed at the
    wrong run) are reported as `extra` -- the one-directional check this
    replaced silently dropped these with no signal at all."""
    matched: list[MatchedThread] = []
    missing: list[str] = []
    for thread_id, gold_label in gold.items():
        predicted_label = predicted.get(thread_id)
        if predicted_label is None:
            missing.append(thread_id)
            continue
        matched.append(MatchedThread(thread_id=thread_id, split=splits.get(thread_id), gold=gold_label, predicted=predicted_label))
    extra = sorted(set(predicted) - set(gold))
    return matched, sorted(missing), extra


# ---------------------------------------------------------------------------
# Report shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateBreakdown:
    n_gated: int
    n_gated_reviewer_correct: int
    gated_reviewer_accuracy: float
    n_ungated: int
    n_ungated_reviewer_correct: int
    ungated_reviewer_accuracy: float
    n_gated_by_category: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmResult:
    name: str
    n: int
    n_reviewer_correct: int
    reviewer_accuracy: float
    reviewer_kappa: float | None
    reviewer_kappa_undefined_reason: str | None
    n_escalation_correct: int
    escalation_accuracy: float
    escalation_kappa: float | None
    escalation_kappa_undefined_reason: str | None
    n_both_correct: int
    both_accuracy: float
    gate_breakdown: GateBreakdown | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalReport:
    corpus_dir: str
    predictions_dir: str
    annotator_id: str
    n_gold: int
    n_predicted: int
    n_matched: int
    n_eval_split_matched: int
    n_dev_split_gold: int
    missing_thread_ids: list[str]
    gate_threshold: float
    majority_reviewer: str | None
    majority_escalation: bool | None
    arms: dict[str, ArmResult]
    strict_failures: list[str]
    extra_thread_ids: list[str] = field(default_factory=list)
    n_invalid_predictions: int = 0
    invalid_prediction_thread_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------


def _safe_kappa(a: list, b: list, categories: list) -> tuple[float | None, str | None]:
    """Wraps cohens_kappa so a degenerate arm (too few threads, or every
    thread landing in the same category) is reported as an explicit
    undefined reason instead of raising or silently returning 0.0."""
    if len(a) < 2:
        return None, "insufficient_data"
    if len(set(categories)) < 2:
        return None, "no_variance_in_categories"
    kappa = cohens_kappa(a, b, categories)
    if kappa is None:
        return None, "undefined_by_cohens_kappa"
    return kappa, None


def _score_arm(
    name: str,
    reviewer_pairs: list[tuple[str, str]],
    escalation_pairs: list[tuple[bool, bool]],
    gate_breakdown: GateBreakdown | None = None,
    notes: list[str] | None = None,
) -> ArmResult:
    n = len(reviewer_pairs)
    n_reviewer_correct = sum(1 for pred, gold in reviewer_pairs if pred == gold)
    n_escalation_correct = sum(1 for pred, gold in escalation_pairs if pred == gold)
    n_both_correct = sum(
        1
        for (pred_r, gold_r), (pred_e, gold_e) in zip(reviewer_pairs, escalation_pairs)
        if pred_r == gold_r and pred_e == gold_e
    )

    reviewer_categories = sorted({r for pair in reviewer_pairs for r in pair})
    reviewer_kappa, reviewer_kappa_reason = _safe_kappa(
        [pred for pred, _ in reviewer_pairs], [gold for _, gold in reviewer_pairs], reviewer_categories
    )
    escalation_categories = sorted({e for pair in escalation_pairs for e in pair})
    escalation_kappa, escalation_kappa_reason = _safe_kappa(
        [pred for pred, _ in escalation_pairs], [gold for _, gold in escalation_pairs], escalation_categories
    )

    return ArmResult(
        name=name,
        n=n,
        n_reviewer_correct=n_reviewer_correct,
        reviewer_accuracy=n_reviewer_correct / n if n else float("nan"),
        reviewer_kappa=reviewer_kappa,
        reviewer_kappa_undefined_reason=reviewer_kappa_reason,
        n_escalation_correct=n_escalation_correct,
        escalation_accuracy=n_escalation_correct / n if n else float("nan"),
        escalation_kappa=escalation_kappa,
        escalation_kappa_undefined_reason=escalation_kappa_reason,
        n_both_correct=n_both_correct,
        both_accuracy=n_both_correct / n if n else float("nan"),
        gate_breakdown=gate_breakdown,
        notes=notes or [],
    )


def _reason_category(reason: str) -> str:
    """Classifies one gating.GateReason reason string into a coarse bucket,
    so the composed/direct gate_breakdown can show whether a flagged thread
    was flagged for low confidence vs. an escalation vs. a missing field --
    on the real pilot corpus every flag happens to be escalation-attributable
    (policy_audit_report.md: escalated_but_not_flagged=0, and confidence
    never independently triggers a flag there), so collapsing all reasons
    into one "gated" bucket would hide that the two are conflated."""
    if reason.startswith("confidence"):
        return "confidence"
    if reason.startswith("missing required field"):
        return "missing_field"
    if reason == "escalation flagged by routing policy":
        return "escalation"
    return "other"


def _gate_breakdown_for(
    threads: list[MatchedThread], reviewer_pairs: list[tuple[str, str]], gate_threshold: float
) -> GateBreakdown:
    """Runs gating.gate() over the arm's *predicted* labels (never gold --
    gating is a property of what the pipeline output looked like, not of the
    ground truth) and splits reviewer-field correctness by flagged vs.
    auto-approved.

    `reviewer_pairs` must be the caller's own already-computed (predicted,
    gold) reviewer pairs, aligned index-for-index with `threads` -- e.g.
    score_direct's raw predicted.assigned_reviewer or score_composed's
    route_from_fields(predicted).assigned_reviewer. Re-deriving correctness
    here from t.predicted.assigned_reviewer vs. t.gold.assigned_reviewer
    would silently score the *direct* comparison even when called from
    score_composed, breaking the invariant that
    n_gated_reviewer_correct + n_ungated_reviewer_correct == arm.n_reviewer_correct."""
    predicted_labels = [t.predicted for t in threads]
    _, flagged = gate(predicted_labels, confidence_threshold=gate_threshold)
    flagged_reasons = {f.thread_id: f.reasons for f in flagged}
    flagged_ids = set(flagged_reasons)

    gated_idx = [i for i, t in enumerate(threads) if t.thread_id in flagged_ids]
    ungated_idx = [i for i, t in enumerate(threads) if t.thread_id not in flagged_ids]
    n_gated_correct = sum(1 for i in gated_idx if reviewer_pairs[i][0] == reviewer_pairs[i][1])
    n_ungated_correct = sum(1 for i in ungated_idx if reviewer_pairs[i][0] == reviewer_pairs[i][1])

    n_gated_by_category: Counter[str] = Counter()
    for i in gated_idx:
        categories = {_reason_category(reason) for reason in flagged_reasons[threads[i].thread_id]}
        for category in categories:
            n_gated_by_category[category] += 1

    return GateBreakdown(
        n_gated=len(gated_idx),
        n_gated_reviewer_correct=n_gated_correct,
        gated_reviewer_accuracy=n_gated_correct / len(gated_idx) if gated_idx else float("nan"),
        n_ungated=len(ungated_idx),
        n_ungated_reviewer_correct=n_ungated_correct,
        ungated_reviewer_accuracy=n_ungated_correct / len(ungated_idx) if ungated_idx else float("nan"),
        n_gated_by_category=dict(n_gated_by_category),
    )


# ---------------------------------------------------------------------------
# The five arms
# ---------------------------------------------------------------------------


def score_oracle(threads: list[MatchedThread]) -> ArmResult:
    reviewer_pairs = [(route_from_fields(t.gold).assigned_reviewer, t.gold.assigned_reviewer) for t in threads]
    escalation_pairs = [(route_from_fields(t.gold).escalation, t.gold.escalation) for t in threads]
    return _score_arm("oracle", reviewer_pairs, escalation_pairs)


def score_composed(threads: list[MatchedThread], gate_threshold: float) -> ArmResult:
    reviewer_pairs = [(route_from_fields(t.predicted).assigned_reviewer, t.gold.assigned_reviewer) for t in threads]
    escalation_pairs = [(route_from_fields(t.predicted).escalation, t.gold.escalation) for t in threads]
    return _score_arm(
        "composed",
        reviewer_pairs,
        escalation_pairs,
        gate_breakdown=_gate_breakdown_for(threads, reviewer_pairs, gate_threshold),
    )


def score_direct(threads: list[MatchedThread], gate_threshold: float) -> ArmResult:
    reviewer_pairs = [(t.predicted.assigned_reviewer, t.gold.assigned_reviewer) for t in threads]
    escalation_pairs = [(t.predicted.escalation, t.gold.escalation) for t in threads]
    return _score_arm(
        "direct",
        reviewer_pairs,
        escalation_pairs,
        gate_breakdown=_gate_breakdown_for(threads, reviewer_pairs, gate_threshold),
    )


def score_policy_fidelity(threads: list[MatchedThread]) -> ArmResult:
    """Compares the policy's own decision on predicted fields against its
    decision on gold fields -- i.e. composed vs. oracle, not either against
    gold. Isolates how much classification-field noise moves the policy's
    output, holding the policy law fixed on both sides (so this arm is
    unaffected by gold/policy divergence, unlike composed and oracle)."""
    reviewer_pairs = [
        (route_from_fields(t.predicted).assigned_reviewer, route_from_fields(t.gold).assigned_reviewer) for t in threads
    ]
    escalation_pairs = [(route_from_fields(t.predicted).escalation, route_from_fields(t.gold).escalation) for t in threads]
    return _score_arm("policy_fidelity", reviewer_pairs, escalation_pairs)


def fit_majority_baseline(dev_gold_labels: list[RFILabel]) -> tuple[str | None, bool | None]:
    if not dev_gold_labels:
        return None, None
    reviewer = Counter(label.assigned_reviewer for label in dev_gold_labels).most_common(1)[0][0]
    escalation = Counter(label.escalation for label in dev_gold_labels).most_common(1)[0][0]
    return reviewer, escalation


def score_majority(threads: list[MatchedThread], majority_reviewer: str | None, majority_escalation: bool | None) -> ArmResult:
    if majority_reviewer is None or not threads:
        return _score_arm("majority", [], [], notes=["majority baseline undefined: no DEV split gold labels or no EVAL split matched threads"])
    reviewer_pairs = [(majority_reviewer, t.gold.assigned_reviewer) for t in threads]
    escalation_pairs = [(majority_escalation, t.gold.escalation) for t in threads]
    return _score_arm("majority", reviewer_pairs, escalation_pairs)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_eval(corpus_dir: Path, predictions_dir: Path, annotator_id: str, gate_threshold: float = GATE_THRESHOLD_DEFAULT) -> EvalReport:
    gold = load_gold_labels(corpus_dir)
    predicted, invalid_predictions = load_predicted_labels(predictions_dir, annotator_id)
    splits = load_splits(corpus_dir)
    matched, missing, extra = match_threads(gold, predicted, splits)

    eval_matched = [t for t in matched if t.split == EVAL_SPLIT]
    dev_gold_labels = [label for thread_id, label in gold.items() if splits.get(thread_id) == DEV_SPLIT]

    majority_reviewer, majority_escalation = fit_majority_baseline(dev_gold_labels)

    arms = {
        "oracle": score_oracle(eval_matched),
        "composed": score_composed(eval_matched, gate_threshold),
        "direct": score_direct(eval_matched, gate_threshold),
        "policy_fidelity": score_policy_fidelity(eval_matched),
        "majority": score_majority(eval_matched, majority_reviewer, majority_escalation),
    }

    strict_failures: list[str] = []
    if not eval_matched:
        strict_failures.append("no EVAL-split threads have both a gold label and a matched prediction")
    if not dev_gold_labels:
        strict_failures.append("no DEV-split gold labels found; majority baseline is undefined")
    if missing:
        eval_missing = [tid for tid in missing if splits.get(tid) == EVAL_SPLIT]
        if eval_missing:
            strict_failures.append(f"{len(eval_missing)} EVAL-split gold threads have no matched prediction")
    if invalid_predictions:
        strict_failures.append(f"{len(invalid_predictions)} predicted labels failed schema.validate_label() and were excluded from scoring")
    if extra:
        strict_failures.append(f"{len(extra)} predicted thread_ids have no matching gold label")

    return EvalReport(
        corpus_dir=str(corpus_dir),
        predictions_dir=str(predictions_dir),
        annotator_id=annotator_id,
        n_gold=len(gold),
        n_predicted=len(predicted),
        n_matched=len(matched),
        n_eval_split_matched=len(eval_matched),
        n_dev_split_gold=len(dev_gold_labels),
        missing_thread_ids=missing,
        gate_threshold=gate_threshold,
        majority_reviewer=majority_reviewer,
        majority_escalation=majority_escalation,
        arms=arms,
        strict_failures=strict_failures,
        extra_thread_ids=extra,
        n_invalid_predictions=len(invalid_predictions),
        invalid_prediction_thread_ids=[p.thread_id for p in invalid_predictions],
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


def write_json_report(report: EvalReport, path: Path) -> None:
    path.write_text(json.dumps(_to_jsonable(report), indent=2, default=str), encoding="utf-8")


def _fmt(x: object) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if x != x:
            return "n/a"
        return f"{x:.4f}"
    return str(x)


def write_markdown_report(report: EvalReport, path: Path) -> None:
    lines = [
        "# Routing Evaluation Report",
        "",
        f"Corpus: `{report.corpus_dir}` | Predictions: `{report.predictions_dir}` | Annotator: {report.annotator_id}",
        "",
        f"- **n_gold**: {report.n_gold}",
        f"- **n_predicted**: {report.n_predicted}",
        f"- **n_matched**: {report.n_matched}",
        f"- **n_eval_split_matched**: {report.n_eval_split_matched} (the scored population for every arm below)",
        f"- **n_dev_split_gold**: {report.n_dev_split_gold} (fits the majority baseline)",
        f"- **missing_thread_ids**: {len(report.missing_thread_ids)}",
        f"- **extra_thread_ids**: {len(report.extra_thread_ids)} (predicted, no matching gold label)",
        f"- **n_invalid_predictions**: {report.n_invalid_predictions} (failed schema.validate_label(), excluded from scoring)",
        f"- **gate_threshold**: {report.gate_threshold}",
        f"- **majority_reviewer**: {report.majority_reviewer!r}",
        f"- **majority_escalation**: {report.majority_escalation!r}",
        "",
    ]
    for name, arm in report.arms.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- **n**: {arm.n}")
        lines.append(f"- **reviewer_accuracy**: {_fmt(arm.reviewer_accuracy)} ({arm.n_reviewer_correct}/{arm.n})")
        lines.append(f"- **reviewer_kappa**: {_fmt(arm.reviewer_kappa)} (undefined_reason={arm.reviewer_kappa_undefined_reason})")
        lines.append(f"- **escalation_accuracy**: {_fmt(arm.escalation_accuracy)} ({arm.n_escalation_correct}/{arm.n})")
        lines.append(f"- **escalation_kappa**: {_fmt(arm.escalation_kappa)} (undefined_reason={arm.escalation_kappa_undefined_reason})")
        lines.append(f"- **both_accuracy**: {_fmt(arm.both_accuracy)} ({arm.n_both_correct}/{arm.n})")
        if arm.gate_breakdown is not None:
            gb = arm.gate_breakdown
            lines.append(
                f"- **gate_breakdown**: gated {_fmt(gb.gated_reviewer_accuracy)} ({gb.n_gated_reviewer_correct}/{gb.n_gated}) "
                f"vs. ungated {_fmt(gb.ungated_reviewer_accuracy)} ({gb.n_ungated_reviewer_correct}/{gb.n_ungated})"
            )
            lines.append(f"- **gate_breakdown.n_gated_by_category**: {dict(sorted(gb.n_gated_by_category.items()))}")
        for note in arm.notes:
            lines.append(f"- note: {note}")
        lines.append("")

    if report.missing_thread_ids:
        lines.append(f"## Missing predictions: {len(report.missing_thread_ids)}")
        for tid in report.missing_thread_ids:
            lines.append(f"- {tid}")
        lines.append("")

    if report.extra_thread_ids:
        lines.append(f"## Extra predictions (no matching gold label): {len(report.extra_thread_ids)}")
        for tid in report.extra_thread_ids:
            lines.append(f"- {tid}")
        lines.append("")

    if report.invalid_prediction_thread_ids:
        lines.append(f"## Invalid predictions (excluded from scoring): {report.n_invalid_predictions}")
        for tid in report.invalid_prediction_thread_ids:
            lines.append(f"- {tid}")
        lines.append("")

    if report.strict_failures:
        lines.append("## Strict-mode failures")
        for f in report.strict_failures:
            lines.append(f"- {f}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score predicted RFI labels against the gold corpus across five routing evaluation arms.")
    parser.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parent.parent / "pilot-data" / "corpus")
    parser.add_argument("--predictions", type=Path, required=True, help="directory containing labels_{annotator-id}/*.json")
    parser.add_argument("--annotator-id", default="pipeline")
    parser.add_argument("--gate-threshold", type=float, default=GATE_THRESHOLD_DEFAULT)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "pilot-data" / "routing_eval")
    parser.add_argument("--strict", action="store_true", help="exit nonzero if any invariant check fails")
    args = parser.parse_args(argv)

    report = run_eval(args.corpus, args.predictions, args.annotator_id, args.gate_threshold)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json_report(report, args.out / "routing_eval_report.json")
    write_markdown_report(report, args.out / "routing_eval_report.md")
    print(
        f"Wrote routing eval report ({report.n_eval_split_matched} eval-split matched threads, "
        f"{len(report.strict_failures)} strict failures) to {args.out}"
    )

    if args.strict and report.strict_failures:
        for f in report.strict_failures:
            print(f"STRICT FAILURE: {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
