"""RFI deadline resolution: turns the classifier's verbatim deadline_text
phrase into an ISO date plus a resolution_status, anchored on the RFI's own
date_submitted (schema.RFIThread.date_submitted) — there is no
"current real-world date" involved anywhere, matching the sibling paper's
meeting_date anchor in ../meeting-action-extraction/code/deadlines.py.

Deliberate deviation from that module: it gates "no deadline stated" on
action_type via a NO_DEADLINE_TYPES set ({"decision", "risk_flag"}), because
those action types structurally have no deadline concept at all.
RFI_TYPES has no analogue to gate on: an RFI is, by definition, a request
that expects a response, so every rfi_type in vocabulary.RFI_TYPES
structurally *has* a deadline concept even when the submitter didn't state
one explicitly. There is therefore no NO_DEADLINE_TYPES set here, and
"none stated" (or an empty/"n/a" phrase) always resolves to
(None, "unresolved") rather than ever branching to an "n/a" status.

resolution_status values:
  - resolved_to_date: an unambiguous ISO date was derived.
  - unresolved: a deadline phrase exists (or is missing) but doesn't pin
    down a single calendar date — including event-anchored phrases like
    "before the concrete pour", whose target date isn't part of this data
    model.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30}

# Compound words ("twenty-one", "twenty one") are added explicitly rather than
# derived at parse time so the lookup stays a flat dict -> value.
_WORD_NUMBERS: dict[str, int] = {**_ONES, **_TENS}
for _tens_word, _tens_val in _TENS.items():
    for _ones_word, _ones_val in _ONES.items():
        if _ones_val >= 10:
            continue
        _WORD_NUMBERS[f"{_tens_word}-{_ones_word}"] = _tens_val + _ones_val
        _WORD_NUMBERS[f"{_tens_word} {_ones_word}"] = _tens_val + _ones_val

# Sorted longest-first so "fourteen" is tried before "four" and "twenty-one"
# before "one" when both start at the same offset in the input text —
# otherwise the shorter word (a substring of the longer one) would win.
_NUMBER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")\b"
    r"|\b\d+\b"
)


def _next_weekday_on_or_after(start: date, weekday_name: str, allow_same_day: bool) -> date:
    target = WEEKDAYS.index(weekday_name.lower())
    delta = (target - start.weekday()) % 7
    if delta == 0 and not allow_same_day:
        delta = 7
    return start + timedelta(days=delta)


def _parse_count(text: str) -> int | None:
    """Returns the first standalone integer or number-word found in `text`,
    or None if there isn't one. Deliberately does NOT concatenate every digit
    in the phrase (a naive `"".join(c for c in text if c.isdigit())` would
    turn "within 2 weeks of the 15th" into 215) and matches number words on
    word boundaries so "fourteen days" isn't misread as "four"."""
    match = _NUMBER_RE.search(text)
    if match is None:
        return None
    token = match.group(0)
    return int(token) if token.isdigit() else _WORD_NUMBERS[token]


def resolve_deadline(deadline_text: str, date_submitted_iso: str) -> tuple[str | None, str]:
    try:
        submitted = date.fromisoformat(date_submitted_iso)
    except ValueError:
        return None, "unresolved"
    text = deadline_text.strip().lower()

    if text in ("none stated", "", "n/a"):
        return None, "unresolved"

    if text in ("today", "asap"):
        return submitted.isoformat(), "resolved_to_date"

    if text == "tomorrow":
        return (submitted + timedelta(days=1)).isoformat(), "resolved_to_date"

    if text in WEEKDAYS:
        # allow_same_day=True: "respond by Thursday" said on a Thursday means
        # today, not a week from now. Previously False, which silently sent
        # every same-weekday-as-submission record's gold date 7 days late.
        return _next_weekday_on_or_after(submitted, text, allow_same_day=True).isoformat(), "resolved_to_date"

    this_week_friday = _next_weekday_on_or_after(submitted, "friday", allow_same_day=True)

    if text == "this week":
        return this_week_friday.isoformat(), "resolved_to_date"

    if text == "next week":
        # Symmetric with "this week": the Friday of the week after this one.
        # (Previously left unresolved with no stated rationale for treating
        # "next week" as less resolvable than "this week" — there isn't one.)
        return (this_week_friday + timedelta(days=7)).isoformat(), "resolved_to_date"

    for prefix in ("by the end of", "by end of"):
        if text.startswith(prefix):
            unit = text[len(prefix):].strip()
            if unit.startswith("the "):
                unit = unit[len("the "):].strip()
            if unit == "day":
                return submitted.isoformat(), "resolved_to_date"
            if unit == "week":
                return this_week_friday.isoformat(), "resolved_to_date"
            break

    if text.startswith("within"):
        n = _parse_count(text)
        if n is None and re.search(r"\bwithin an?\b", text):
            n = 1
        if n is not None:
            if "week" in text:
                return (submitted + timedelta(weeks=n)).isoformat(), "resolved_to_date"
            if "day" in text:
                if "business" in text or "working" in text:
                    result = submitted
                    added = 0
                    while added < n:
                        result += timedelta(days=1)
                        if result.weekday() < 5:  # Mon-Fri
                            added += 1
                    return result.isoformat(), "resolved_to_date"
                return (submitted + timedelta(days=n)).isoformat(), "resolved_to_date"

    # Event-anchored phrases (e.g. "before the concrete pour") have no
    # resolvable calendar date in this data model; fall through with every
    # other unrecognized phrase to the same honest "unresolved" catch-all.
    return None, "unresolved"
