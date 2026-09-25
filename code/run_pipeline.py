"""run_pipeline.py: classify a corpus split end to end and write a reviewable
prediction bundle, without ever touching pilot-data/corpus/ (the gold-label
tree).

For each selected thread: classify.classify_thread() (bounded repair retry +
deadline resolution), then gating.gate() once over the whole batch of
successfully-parsed labels (classify.py deliberately does not call gate()
itself — see its docstring). A client exception beyond the
ClassificationParseError classify.py already retries (a real API call
failing outright) is caught here per-thread and recorded as a "error"
outcome, so a real-provider run over hundreds of threads doesn't die mid-batch
and lose the budget already spent; --fail-fast disables that and re-raises
immediately for debugging.

Output goes to pilot-data/predicted-{client}-{condition}[-{split}]/ (or
--out): provenance.json, predictions.json, labels_pipeline/*.json (or
labels_{--annotator-id}/ if overridden), review_queue.json, raw/*.json, and a
human-readable summary.md.

    python code/run_pipeline.py --client stub --split dev --limit 5
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_SAFE_ANNOTATOR_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify import MAX_REPAIR_ATTEMPTS, ClassificationOutcome, classify_thread, outcome_to_dict, summarize_outcomes
from gating import GateReason, gate
from llm_client import LLMClient, build_client
from schema import RFIThread, read_thread, validate_label, write_label

DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_ANNOTATOR_ID = "pipeline"
SPLITS = ("all", "dev", "eval")

OWNED_FILES = ("provenance.json", "predictions.json", "review_queue.json", "summary.md")
OWNED_DIRS = ("raw",)  # the labels_{annotator_id} dir is added dynamically


class CorpusError(Exception):
    """Raised for anything wrong with the input side before a single client
    call is made: a missing corpus/threads directory, a --thread-id with no
    matching file, a thread file that fails to parse, or an empty selection.
    main() maps this to exit code 2."""


class OutputDirUnsafeError(Exception):
    """Raised by assert_out_dir_safe()/assert_cache_dir_safe() when --out or
    --cache-dir (or a derived default) would write into the gold-label corpus
    tree, by prepare_out_dir() when the output directory already holds a
    previous run's output and --force wasn't passed, and by the
    labels_{annotator_id} containment check in _existing_owned_paths()/
    write_run() as a defense-in-depth backstop behind validate_annotator_id().
    main() maps all of these to exit code 2."""


def select_thread_paths(
    corpus_dir: Path,
    split: str,
    thread_ids: list[str] | None,
    limit: int,
) -> list[Path]:
    threads_dir = corpus_dir / "threads"
    if not threads_dir.is_dir():
        raise CorpusError(f"no threads/ directory found under {corpus_dir}")

    all_paths = sorted(threads_dir.glob("*.json"))
    by_stem = {p.stem: p for p in all_paths}

    if thread_ids:
        missing = [tid for tid in thread_ids if tid not in by_stem]
        if missing:
            raise CorpusError(f"--thread-id not found in {threads_dir}: {missing}")
        # dict.fromkeys dedupes an accidentally-repeated --thread-id while
        # preserving first-occurrence order — load_threads() below treats two
        # paths producing the same thread_id as a corpus error, which a
        # literal repeat of the same id/path is not.
        deduped_ids = list(dict.fromkeys(thread_ids))
        return [by_stem[tid] for tid in deduped_ids]

    if split == "all":
        selected = all_paths
    else:
        selected = [p for p in all_paths if p.stem.startswith(f"{split}-")]

    if limit > 0:
        selected = selected[:limit]

    if not selected:
        raise CorpusError(
            f"no threads selected (corpus={corpus_dir}, split={split!r}, limit={limit}) — refusing to run an empty batch"
        )
    return selected


def load_threads(paths: list[Path]) -> list[RFIThread]:
    threads = []
    seen_ids: dict[str, Path] = {}
    for path in paths:
        try:
            thread = read_thread(path)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            raise CorpusError(f"{path}: could not be read as an RFIThread: {e}") from e
        if thread.thread_id != path.stem:
            # thread_id comes from file *content*, not the filename — but
            # write_run() uses it verbatim in output paths (labels_*/{id}.json,
            # raw/{id}.json) and every downstream artifact (provenance.json,
            # review_queue.json) is keyed by it, while selection above is keyed
            # by filename stem. Requiring equality closes both a path-traversal
            # hole (a content thread_id like "../../corpus/threads/dev-0001"
            # would otherwise be used unvalidated in a write path) and a silent
            # join mismatch against gold labels keyed by filename.
            raise CorpusError(
                f"{path}: thread_id {thread.thread_id!r} does not match filename stem "
                f"{path.stem!r} — every downstream artifact is keyed by thread_id, so a "
                "mismatch here would silently break joins against it, and an unsafe value "
                "would otherwise be used verbatim in an output file path"
            )
        if thread.thread_id in seen_ids:
            raise CorpusError(
                f"duplicate thread_id {thread.thread_id!r}: both {seen_ids[thread.thread_id]} and "
                f"{path} produce this id — every downstream artifact is keyed by thread_id, so one "
                "would silently overwrite the other"
            )
        seen_ids[thread.thread_id] = path
        threads.append(thread)
    return threads


def resolve_out_dir(args: argparse.Namespace, data_dir: Path) -> Path:
    if args.out is not None:
        return args.out
    suffix = "" if args.split == "all" else f"-{args.split}"
    return data_dir / f"predicted-{args.client}-{args.condition}{suffix}"


def validate_annotator_id(annotator_id: str) -> None:
    """--annotator-id is used verbatim as a directory-name fragment
    (labels_{annotator_id}) below --out. Without this check a value like
    "../../../corpus/predicted" turns that into a multi-segment relative
    path that can resolve outside --out entirely — including into the gold
    corpus, where --force would then delete it. Restricting to a plain
    filename-safe charset rules out path separators and any '..' segment."""
    if not annotator_id or not set(annotator_id) <= _SAFE_ANNOTATOR_ID_CHARS:
        raise CorpusError(
            f"--annotator-id {annotator_id!r} is not safe: it is used verbatim in a directory name "
            "(labels_{annotator_id}) and must be non-empty, containing only letters, digits, '_', and '-'."
        )


def _resolves_into(path: Path, ancestor: Path) -> bool:
    resolved_path = path.resolve()
    resolved_ancestor = ancestor.resolve()
    return resolved_path == resolved_ancestor or resolved_ancestor in resolved_path.parents


def assert_out_dir_safe(out_dir: Path, corpus_dir: Path) -> None:
    if _resolves_into(out_dir, corpus_dir):
        raise OutputDirUnsafeError(
            f"refusing to write pipeline output into {out_dir} — it is the corpus directory "
            f"({corpus_dir}) or a path inside it. Pass a different --out."
        )
    if (out_dir / "threads").is_dir():
        raise OutputDirUnsafeError(
            f"refusing to write pipeline output into {out_dir} — it contains a threads/ "
            "subdirectory, which means it is (or shadows) a corpus directory, not a "
            "predictions directory. Pass a different --out."
        )


def assert_cache_dir_safe(cache_dir: Path, corpus_dir: Path) -> None:
    if _resolves_into(cache_dir, corpus_dir):
        raise OutputDirUnsafeError(
            f"refusing to use {cache_dir} as --cache-dir — it is the corpus directory "
            f"({corpus_dir}) or a path inside it. Pass a different --cache-dir."
        )


def _labels_dir(out_dir: Path, annotator_id: str) -> Path:
    """Defense-in-depth backstop behind validate_annotator_id(): even if an
    unsafe annotator_id somehow reached here, refuse to touch anything that
    doesn't resolve to a direct child of out_dir, rather than trusting string
    concatenation before an rmtree() or mkdir()."""
    labels_dir = out_dir / f"labels_{annotator_id}"
    if labels_dir.resolve().parent != out_dir.resolve():
        raise OutputDirUnsafeError(
            f"labels_{annotator_id} would resolve outside {out_dir} — refusing "
            "(--annotator-id looks unsafe)"
        )
    return labels_dir


def _existing_owned_paths(out_dir: Path, annotator_id: str) -> list[Path]:
    existing = []
    for name in OWNED_FILES:
        p = out_dir / name
        if p.exists():
            existing.append(p)
    for name in OWNED_DIRS:
        p = out_dir / name
        if p.is_dir() and any(p.iterdir()):
            existing.append(p)
    labels_dir = _labels_dir(out_dir, annotator_id)
    if labels_dir.is_dir() and any(labels_dir.iterdir()):
        existing.append(labels_dir)
    return existing


def _other_labels_dirs(out_dir: Path, annotator_id: str) -> list[Path]:
    """labels_* dirs belonging to a different --annotator-id. Left alone by
    design (accumulating multiple annotators' labels in one --out is a
    supported workflow — see labels_annotator_b in the annotation docs), but
    a stale one from an earlier, differently-configured run is a footgun for
    annotator_agreement.py's labels_* auto-discovery, so it's worth a
    warning."""
    if not out_dir.is_dir():
        return []
    this_one = f"labels_{annotator_id}"
    return sorted(
        p for p in out_dir.glob("labels_*")
        if p.is_dir() and p.name != this_one and any(p.iterdir())
    )


def prepare_out_dir(out_dir: Path, annotator_id: str, force: bool) -> None:
    existing = _existing_owned_paths(out_dir, annotator_id) if out_dir.exists() else []
    if existing and not force:
        names = ", ".join(p.name for p in existing)
        raise OutputDirUnsafeError(
            f"{out_dir} already contains pipeline output ({names}) from a previous run. "
            "Refusing to overwrite it silently — pass --force to intentionally re-run "
            "(this clears only run_pipeline.py's own files/dirs first), or --out to write "
            "to a different directory."
        )
    # Warn only once the run is actually going ahead — otherwise a run that
    # aborts above (no --force) would print a warning about a directory it
    # was never going to touch either way.
    others = _other_labels_dirs(out_dir, annotator_id)
    if others:
        names = ", ".join(p.name for p in others)
        print(
            f"run_pipeline.py: warning: {out_dir} also contains {names} from a different "
            "--annotator-id. Left untouched, but this run's provenance.json describes only "
            f"labels_{annotator_id} — if {names} came from a different client/condition, "
            "downstream agreement/scoring over this bundle will silently mix them in.",
            file=sys.stderr,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in existing:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


@dataclass
class RunResult:
    outcomes: list[ClassificationOutcome]
    flagged: list[GateReason]
    validation_errors: dict[str, list[str]] = field(default_factory=dict)


def run(
    client: LLMClient,
    threads: list[RFIThread],
    annotator_id: str,
    confidence_threshold: float,
    max_repair_attempts: int,
    fail_fast: bool,
) -> RunResult:
    outcomes: list[ClassificationOutcome] = []
    for thread in threads:
        try:
            outcome = classify_thread(client, thread, max_repair_attempts=max_repair_attempts)
        except Exception as e:
            if fail_fast:
                raise
            outcome = ClassificationOutcome(
                thread_id=thread.thread_id,
                status="error",
                label=None,
                # classify_thread() doesn't expose how many attempts it made
                # before raising, but a call was necessarily made for the
                # exception to occur at all, so 0 would understate it; 1 is a
                # documented lower bound, not an exact count.
                attempts=1,
                repaired=False,
                raw_responses=[],
                error_detail=f"{type(e).__name__}: {e}",
            )
        outcomes.append(outcome)

    validation_errors: dict[str, list[str]] = {}
    ok_labels = []
    for outcome in outcomes:
        if outcome.status == "ok" and outcome.label is not None:
            outcome.label.annotator = annotator_id
            validation_errors[outcome.thread_id] = validate_label(outcome.label)
            ok_labels.append(outcome.label)
        else:
            validation_errors[outcome.thread_id] = []

    _, flagged = gate(ok_labels, confidence_threshold=confidence_threshold)

    return RunResult(outcomes=outcomes, flagged=flagged, validation_errors=validation_errors)


def build_provenance(args: argparse.Namespace, client: LLMClient, threads: list[RFIThread], out_dir: Path) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "client": client.name,
        "model": client.model,
        "condition": args.condition,
        "corpus_dir": str(args.corpus),
        "split": args.split,
        "thread_ids": sorted(t.thread_id for t in threads),
        "thread_count": len(threads),
        "confidence_threshold": args.confidence_threshold,
        "max_repair_attempts": args.max_repair_attempts,
        "annotator_id": args.annotator_id,
        # Read off the client itself (RuleBasedStubClient has no cache_dir
        # attribute at all) rather than args.cache_dir/args.no_cache, so this
        # can't claim a cache was used when the client never had one.
        "cache_dir": str(cd) if (cd := getattr(client, "cache_dir", None)) else None,
        "out_dir": str(out_dir),
        "client_stats": dict(getattr(client, "stats", {})),
        "disclaimer": (
            "Model-generated predictions, not ground truth. Check review_queue.json "
            "(gated / parse_failed / errored / invalid_labels) before treating any "
            "label here as final."
        ),
    }


def write_run(out_dir: Path, provenance: dict, result: RunResult, save_raw: bool) -> None:
    outcomes_sorted = sorted(result.outcomes, key=lambda o: o.thread_id)
    summary = summarize_outcomes(result.outcomes)
    errored = sum(1 for o in result.outcomes if o.status == "error")

    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    predictions = {
        "provenance": provenance,
        "summary": {**asdict(summary), "errored": errored, "gated": len(result.flagged)},
        "outcomes": [
            {
                **outcome_to_dict(o),
                "validation_errors": result.validation_errors.get(o.thread_id, []),
                "n_raw_responses": len(o.raw_responses),
            }
            for o in outcomes_sorted
        ],
    }
    (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    annotator_id = provenance["annotator_id"]
    labels_dir = _labels_dir(out_dir, annotator_id)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for o in outcomes_sorted:
        if o.status == "ok" and o.label is not None:
            write_label(o.label, labels_dir / f"{o.thread_id}.json")

    raw_dir = out_dir / "raw"
    for o in outcomes_sorted:
        if o.status == "parse_failed" or save_raw:
            if o.raw_responses:
                raw_dir.mkdir(parents=True, exist_ok=True)
                payload = [asdict(r) for r in o.raw_responses]
                (raw_dir / f"{o.thread_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    review_queue = {
        "gated": [
            {"thread_id": gr.thread_id, "annotator": gr.annotator, "reasons": gr.reasons}
            for gr in sorted(result.flagged, key=lambda gr: gr.thread_id)
        ],
        "parse_failed": [
            {"thread_id": o.thread_id, "error_detail": o.error_detail}
            for o in outcomes_sorted if o.status == "parse_failed"
        ],
        "errored": [
            {"thread_id": o.thread_id, "error_detail": o.error_detail}
            for o in outcomes_sorted if o.status == "error"
        ],
        "invalid_labels": [
            {"thread_id": tid, "errors": errs}
            for tid, errs in sorted(result.validation_errors.items())
            if errs
        ],
    }
    (out_dir / "review_queue.json").write_text(json.dumps(review_queue, indent=2), encoding="utf-8")

    lines = [
        f"# run_pipeline.py summary — {provenance['client']}/{provenance['condition']} ({provenance['split']})",
        "",
        f"Generated: {provenance['generated_at']}",
        f"Threads: {summary.total} (ok={summary.ok}, repaired={summary.repaired}, "
        f"parse_failed={summary.parse_failed}, errored={errored})",
        f"Gated for review: {len(review_queue['gated'])}",
        f"Invalid labels (parsed but schema-invalid): {len(review_queue['invalid_labels'])}",
        "",
    ]
    if review_queue["gated"]:
        lines.append("## Gated")
        for item in review_queue["gated"]:
            lines.append(f"- {item['thread_id']}: {'; '.join(item['reasons'])}")
        lines.append("")
    if review_queue["parse_failed"]:
        lines.append("## Parse failed")
        for item in review_queue["parse_failed"]:
            lines.append(f"- {item['thread_id']}: {item['error_detail']}")
        lines.append("")
    if review_queue["errored"]:
        lines.append("## Errored")
        for item in review_queue["errored"]:
            lines.append(f"- {item['thread_id']}: {item['error_detail']}")
        lines.append("")
    if review_queue["invalid_labels"]:
        lines.append("## Invalid labels")
        for item in review_queue["invalid_labels"]:
            lines.append(f"- {item['thread_id']}: {'; '.join(item['errors'])}")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "pilot-data"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=data_dir / "corpus")
    parser.add_argument("--client", choices=("stub", "anthropic", "openai"), default="stub")
    parser.add_argument("--condition", choices=("bare", "vocab", "policy"), default="bare")
    parser.add_argument("--split", choices=SPLITS, default="all")
    parser.add_argument("--thread-id", action="append", dest="thread_id", default=None,
                         help="Classify only these thread ids (repeatable). Overrides --split and --limit.")
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true",
                         help="Clear this run's own previous output (provenance.json, predictions.json, "
                              "review_queue.json, summary.md, raw/, labels_{annotator-id}/) before writing.")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--max-repair-attempts", type=int, default=MAX_REPAIR_ATTEMPTS)
    parser.add_argument("--annotator-id", default=DEFAULT_ANNOTATOR_ID)
    parser.add_argument("--cache-dir", type=Path, default=data_dir / "llm-cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--save-raw", action="store_true", help="Write raw/{thread_id}.json for every outcome, not just parse_failed ones.")
    parser.add_argument("--fail-fast", action="store_true", help="Re-raise a client exception immediately instead of recording it as an 'error' outcome.")
    args = parser.parse_args(argv)

    out_dir = resolve_out_dir(args, data_dir)
    cache_dir = None if args.no_cache else args.cache_dir

    try:
        if args.limit < 0:
            raise CorpusError(f"--limit must be >= 0 (0 = no limit), got {args.limit}")
        if args.max_repair_attempts < 0:
            raise CorpusError(
                f"--max-repair-attempts must be >= 0, got {args.max_repair_attempts} — a negative "
                "value makes classify_thread()'s range(1, n+2) empty, so every thread would be "
                "reported parse_failed without a single client call ever being made"
            )
        if not (0.0 <= args.confidence_threshold <= 1.0):
            raise CorpusError(f"--confidence-threshold must be in [0, 1], got {args.confidence_threshold}")
        validate_annotator_id(args.annotator_id)
        assert_out_dir_safe(out_dir, args.corpus)
        if cache_dir is not None:
            assert_cache_dir_safe(cache_dir, args.corpus)
        paths = select_thread_paths(args.corpus, args.split, args.thread_id, args.limit)
        threads = load_threads(paths)
        # Built before prepare_out_dir() below so a client construction
        # failure (missing SDK, unset API key) is caught here, before a
        # previous run's --force'd output has been deleted — not after.
        client = build_client(args.client, condition=args.condition, cache_dir=cache_dir)
        prepare_out_dir(out_dir, args.annotator_id, args.force)
    except (CorpusError, OutputDirUnsafeError, RuntimeError, OSError) as e:
        print(f"run_pipeline.py: {e}", file=sys.stderr)
        return 2

    result = run(
        client,
        threads,
        annotator_id=args.annotator_id,
        confidence_threshold=args.confidence_threshold,
        max_repair_attempts=args.max_repair_attempts,
        fail_fast=args.fail_fast,
    )
    provenance = build_provenance(args, client, threads, out_dir)
    try:
        write_run(out_dir, provenance, result, save_raw=args.save_raw)
    except OSError as e:
        # Classification already happened (and any real-provider budget
        # already spent) by this point, so this can only be a filesystem
        # problem writing the bundle out — report it the same tidy way as
        # every other CLI-level failure rather than a raw traceback.
        print(f"run_pipeline.py: failed writing output to {out_dir}: {e}", file=sys.stderr)
        return 2

    summary = summarize_outcomes(result.outcomes)
    errored = sum(1 for o in result.outcomes if o.status == "error")
    print(f"{summary.total} threads: ok={summary.ok} repaired={summary.repaired} "
          f"parse_failed={summary.parse_failed} errored={errored} gated={len(result.flagged)}")
    print(f"written to {out_dir}")

    return 1 if (summary.parse_failed or errored) else 0


if __name__ == "__main__":
    raise SystemExit(main())
