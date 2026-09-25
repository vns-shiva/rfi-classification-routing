# Annotator B instructions

This directory holds a second, independent set of hand labels for the RFI
threads in `pilot-data/corpus/threads/`, used to compute inter-annotator
agreement (Krippendorff's alpha, quadratic-weighted kappa, MASI, set-level
micro-P/R/F1 — see `code/annotator_agreement.py`) against the gold labels in
the sibling `labels/` directory.

**Work independently.** Do not open `labels/` or discuss specific threads
with the other annotator before both label sets are complete — the whole
point of this pass is to measure how much two people, working from the raw
thread text alone, agree with each other.

## What to label

For each thread under `threads/<thread_id>.json`, read the full RFI
exchange (`RFIThread.render()`'s output — question, responses, any
clarification/resubmittal) and produce one label file at
`labels_annotator_b/<thread_id>.json` with the same `thread_id`, matching
`code/schema.py`'s `RFILabel` shape:

- **Classification** — `rfi_type` (one of `code/vocabulary.py`'s
  `RFI_TYPES`), `primary_discipline` and `secondary_disciplines` (from
  `DISCIPLINES`), `csi_division` (from the primary discipline's allowed set
  in `DISCIPLINE_TO_CSI`, or `"—"` if none applies), `urgency` (one of
  `URGENCY_TIERS`, ordered `routine < priority < urgent < critical`).
- **Extraction** — `question_summary` (one or two sentences, your own
  words), `referenced_documents` (drawing/spec numbers actually cited in the
  thread), `proposed_solution` (`"—"` if the submitter proposed none),
  `cost_impact` / `schedule_impact` / `answer_in_documents` (booleans),
  `deadline_text` (the verbatim phrase used in the thread, or `"none
  stated"`).
- **Routing** — `assigned_reviewer` (one of `REVIEWER_ROLES`),
  `routing_rationale` (why that role), `escalation` (true if this RFI
  should jump the normal queue).
- **Metadata** — set `annotator` to exactly `"annotator_b"` (required —
  `load_annotator_labels()` hard-errors on a mismatch between this field
  and the directory it's found in) and `confidence` to your own honest
  confidence in the label, 0.0-1.0. Leave `status` and
  `deadline_resolution_status`/`deadline_iso` at their schema defaults
  unless you are deliberately resolving the deadline text to a calendar
  date.

## Validating before you're done

Run each label through `validate_label()` before committing it — an invalid
label (out-of-vocabulary value, out-of-range confidence, contradictory
deadline fields) will hard-error out of the agreement computation rather
than silently degrade it:

```
python -c "from schema import RFILabel, validate_label; import json, sys; \
d = json.load(open(sys.argv[1])); l = RFILabel(**d); \
print(validate_label(l) or 'OK')" pilot-data/corpus/labels_annotator_b/<thread_id>.json
```

(run from `code/`, or with `code/` on `PYTHONPATH`).
