"""classify.py: per-thread RFI classification with a bounded repair retry.

Orchestration layer sitting on top of llm_client.LLMClient.classify_raw():
calls the client once, and if it raises ClassificationParseError, retries at
most MAX_REPAIR_ATTEMPTS more times with a repair instruction appended to the
user prompt (prompts.build_repair_instruction()) via classify_raw()'s
repair_suffix parameter, so each retry changes the (system, user) cache key
instead of replaying the same cached failure.

Also resolves each label's deadline_text into a concrete
deadline_iso/deadline_resolution_status here, in the orchestration layer —
llm_client.py never touches deadlines.py itself — matching
meeting-action-extraction/code/pipeline.py's _to_action_item() pattern.

classify.py does not call gating.gate(): that is run_pipeline.py's job, once
per full batch of produced labels, not per-thread here.

Run standalone for a quick manual check:

    python code/classify.py --client stub --condition bare pilot-data/corpus/rfi-001.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deadlines import resolve_deadline
from llm_client import ClassificationParseError, LLMClient, RawResponse, build_client
from prompts import build_repair_instruction
from schema import RFILabel, RFIThread, label_to_dict, read_thread

MAX_REPAIR_ATTEMPTS = 1


@dataclass
class ClassificationOutcome:
    """The result of classifying one thread, including the repair-retry
    history — kept even on success so a caller can report how often repairs
    were needed without re-deriving it from raw_responses."""
    thread_id: str
    status: str  # "ok" or "parse_failed" here; run_pipeline.py's run() also produces "error"
    # for a non-parse client exception (network/API failure) — this field is a plain str with
    # no enum validation specifically so callers can extend it without changing this dataclass.
    label: RFILabel | None
    attempts: int
    repaired: bool
    raw_responses: list[RawResponse] = field(default_factory=list)
    error_detail: str | None = None


def resolve_label_deadline(label: RFILabel, thread: RFIThread) -> None:
    """Mutates label.deadline_iso/deadline_resolution_status in place from
    label.deadline_text, anchored on thread.date_submitted. Required because
    ChatLLMClient/RuleBasedStubClient never populate these fields themselves
    (see this module's docstring)."""
    deadline_iso, status = resolve_deadline(label.deadline_text, thread.date_submitted)
    label.deadline_iso = deadline_iso
    label.deadline_resolution_status = status


def classify_thread(
    client: LLMClient,
    thread: RFIThread,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
) -> ClassificationOutcome:
    raw_responses: list[RawResponse] = []
    repair_suffix = ""
    last_error: ClassificationParseError | None = None

    for attempt in range(1, max_repair_attempts + 2):  # 1 initial + N repairs
        try:
            label, raw = client.classify_raw(thread, repair_suffix=repair_suffix)
        except ClassificationParseError as e:
            last_error = e
            raw_responses.append(RawResponse(payload=e.payload, model=client.model or client.name, cache_hit=False))
            repair_suffix = build_repair_instruction(e.payload, e.detail)
            continue

        raw_responses.append(raw)
        resolve_label_deadline(label, thread)
        return ClassificationOutcome(
            thread_id=thread.thread_id,
            status="ok",
            label=label,
            attempts=attempt,
            repaired=attempt > 1,
            raw_responses=raw_responses,
        )

    return ClassificationOutcome(
        thread_id=thread.thread_id,
        status="parse_failed",
        label=None,
        attempts=max_repair_attempts + 1,
        repaired=False,
        raw_responses=raw_responses,
        error_detail=last_error.detail if last_error else None,
    )


def classify_corpus(
    client: LLMClient,
    threads: list[RFIThread],
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
) -> list[ClassificationOutcome]:
    return [classify_thread(client, thread, max_repair_attempts) for thread in threads]


def outcome_to_dict(outcome: ClassificationOutcome) -> dict:
    return {
        "thread_id": outcome.thread_id,
        "status": outcome.status,
        "attempts": outcome.attempts,
        "repaired": outcome.repaired,
        "error_detail": outcome.error_detail,
        "label": label_to_dict(outcome.label) if outcome.label is not None else None,
    }


@dataclass
class OutcomeSummary:
    total: int
    ok: int
    parse_failed: int
    repaired: int


def summarize_outcomes(outcomes: list[ClassificationOutcome]) -> OutcomeSummary:
    return OutcomeSummary(
        total=len(outcomes),
        ok=sum(1 for o in outcomes if o.status == "ok"),
        parse_failed=sum(1 for o in outcomes if o.status == "parse_failed"),
        repaired=sum(1 for o in outcomes if o.repaired),
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify one RFI thread and print the outcome as JSON.")
    parser.add_argument("thread_path", type=Path)
    parser.add_argument("--client", default="stub", choices=("stub", "anthropic", "openai"))
    parser.add_argument("--condition", default="bare", choices=("bare", "vocab", "policy"))
    args = parser.parse_args(argv)

    client = build_client(args.client, condition=args.condition)
    thread = read_thread(args.thread_path)
    outcome = classify_thread(client, thread)
    print(json.dumps(outcome_to_dict(outcome), indent=2))
    return 0 if outcome.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
