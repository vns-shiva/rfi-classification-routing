"""generate_annotator_b.py: produce the second, independent label set the
paper's inter-annotator-agreement (IAA) analysis needs, per
pilot-data/corpus/labels_annotator_b/README.md.

No second human coder is available for this pilot. Rather than skip IAA or
silently reuse one of the ablation grid's own arms (which would not be an
*independent* second opinion — it would just be that arm scoring against
itself), this script uses a model that never appears anywhere in the
ablation grid (llm_client.CONDITIONS x {anthropic claude-sonnet-5, openai
gpt-5}): Anthropic's claude-haiku-4-5-20251001, under the "vocab" condition
(schema + taxonomy, no routing_policy.py decision table) — the closest match
to what the README actually asks a human annotator to do: choose values from
RFI_TYPES/DISCIPLINES/etc. by name, but make their own judgment call on
escalation rather than citing a lookup table. This is a disclosed LLM proxy
for a second human annotator, not a claim that it IS one; the manuscript's
limitations section says so explicitly.

Scope: the dev split only (40 of 240 threads), not the full corpus. IAA
measures label-scheme reliability, not the eval-split accuracy the ablation
grid is scored on, so double-coding the pre-existing, already-untouched dev
split is a natural, principled, and cost-bounded subset rather than an
arbitrary sample carved out of the eval set the paper reports headline
numbers on.

Writes pilot-data/corpus/labels_annotator_b/<thread_id>.json, matching
schema.write_label()'s shape, with .annotator stamped "annotator_b" (required
by annotator_agreement.load_annotator_labels()) and every label passed
through validate_label() before being written — an invalid label is reported
and skipped, never written, so a corrupt file can't silently break the
agreement computation later.

    python code/generate_annotator_b.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify import classify_thread, resolve_label_deadline
from llm_client import AnthropicLLMClient
from prompts import build_repair_instruction
from schema import validate_label, write_label, read_thread

_CORPUS_DEFAULT = Path(__file__).resolve().parent.parent / "pilot-data" / "corpus"
_MODEL = "claude-haiku-4-5-20251001"
_CONDITION = "vocab"


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=_CORPUS_DEFAULT)
    parser.add_argument("--split", default="dev", choices=("all", "dev", "eval"))
    parser.add_argument("--thread-id", action="append", default=None, help="classify only these thread ids; overrides --split/--limit")
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--force", action="store_true", help="overwrite existing labels_annotator_b/*.json")
    parser.add_argument("--max-repair-attempts", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=_CORPUS_DEFAULT.parent / "llm-cache-annotator-b")
    args = parser.parse_args(argv)

    if args.limit < 0:
        parser.error(f"--limit must be >= 0 (0 = no limit), got {args.limit}")

    threads_dir = args.corpus / "threads"
    out_dir = args.corpus / "labels_annotator_b"
    out_dir.mkdir(parents=True, exist_ok=True)

    thread_paths = sorted(threads_dir.glob("*.json"))
    if args.thread_id:
        wanted = set(args.thread_id)
        thread_paths = [p for p in thread_paths if p.stem in wanted]
    else:
        if args.split != "all":
            thread_paths = [p for p in thread_paths if p.stem.startswith(f"{args.split}-")]
        if args.limit:
            thread_paths = thread_paths[: args.limit]
    if not thread_paths:
        parser.error(f"no threads selected (corpus={args.corpus}, split={args.split!r}, thread_id={args.thread_id!r})")

    if not args.force:
        already = [p for p in thread_paths if (out_dir / f"{p.stem}.json").exists()]
        if already:
            parser.error(
                f"{len(already)} of {len(thread_paths)} selected threads already have a "
                f"labels_annotator_b/*.json (e.g. {already[0].name}). Pass --force to re-label them."
            )

    client = AnthropicLLMClient(model=_MODEL, condition=_CONDITION, cache_dir=args.cache_dir)

    ok = parse_failed = invalid = 0
    for path in thread_paths:
        thread = read_thread(path)
        outcome = classify_thread(client, thread, max_repair_attempts=args.max_repair_attempts)
        if outcome.status != "ok":
            parse_failed += 1
            print(f"parse_failed: {thread.thread_id}: {outcome.error_detail}")
            continue

        label = outcome.label
        label.annotator = "annotator_b"
        errors = validate_label(label)
        if errors:
            # classify_thread()'s own repair loop only covers
            # ClassificationParseError (malformed JSON); a label that parses
            # fine but fails validate_label()'s semantic checks (e.g. a
            # csi_division/primary_discipline mismatch) reaches here instead.
            # Give it one more chance with the same repair-instruction
            # mechanism classify.py uses for parse errors, quoting the
            # actual validate_label() errors instead of a parse-error detail.
            previous_payload = outcome.raw_responses[-1].payload if outcome.raw_responses else ""
            repair_suffix = build_repair_instruction(previous_payload, "; ".join(errors))
            repaired_label, _raw = client.classify_raw(thread, repair_suffix=repair_suffix)
            repaired_label.annotator = "annotator_b"
            resolve_label_deadline(repaired_label, thread)
            repaired_errors = validate_label(repaired_label)
            if repaired_errors:
                invalid += 1
                print(f"invalid, skipped: {thread.thread_id}: {errors} (repair attempt still invalid: {repaired_errors})")
                continue
            label = repaired_label

        write_label(label, out_dir / f"{thread.thread_id}.json")
        ok += 1

    print(
        f"{len(thread_paths)} threads: ok={ok} parse_failed={parse_failed} invalid={invalid} "
        f"(model={_MODEL} condition={_CONDITION}) written to {out_dir}"
    )
    return 0 if ok == len(thread_paths) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
