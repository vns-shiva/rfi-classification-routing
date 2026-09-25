"""Unit tests for generate_rfis.py. Run with:

    python -m unittest discover -s code/tests -t code

Uses the real templates + shared pools (load_templates()/load_shared_pools())
and a fixed-seed build_plan() so every test here is deterministic and
exercises the actual production template content, not synthetic stand-ins.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stratification import build_plan
from templates import UNRESOLVABLE_FAMILIES, load_shared_pools, load_templates

import generate_rfis as gr

_TEMPLATES = load_templates()
_POOLS = load_shared_pools()
_PLAN = build_plan(templates=_TEMPLATES, seed=12345)
_RECORDS = gr.generate_corpus(_PLAN, _TEMPLATES, _POOLS)


class TestExtractDeadlineSpan(unittest.TestCase):
    def test_relative_days_pulls_within_clause_out_of_full_sentence(self):
        self.assertEqual(
            gr._extract_deadline_span(
                "relative_days", "Please respond within five days so we can place the order."
            ),
            "within five days",
        )

    def test_business_days_phrase_is_matched_by_the_shared_within_pattern(self):
        # The within-pattern is shared across relative_days/relative_weeks/
        # business_days, so it must extract correctly regardless of which of
        # those three family keys the phrase happens to be filed under.
        self.assertEqual(
            gr._extract_deadline_span(
                "business_days", "Please respond within five business days so we can place the order."
            ),
            "within five business days",
        )

    def test_within_pattern_is_bounded_and_fails_closed_on_a_distant_unit_word(self):
        # Regression: the pattern used to be an unbounded `.*?`, so it would
        # swallow an entire unrelated clause to reach some later day/week
        # word instead of failing closed when the real span isn't nearby.
        self.assertIsNone(
            gr._extract_deadline_span(
                "relative_days",
                "Please respond within the window we discussed; the pour is in three days.",
            )
        )

    def test_relative_weeks(self):
        self.assertEqual(
            gr._extract_deadline_span("relative_weeks", "We need this within two weeks of today's date."),
            "within two weeks",
        )

    def test_weekday_takes_the_first_match_when_two_are_present(self):
        self.assertEqual(
            gr._extract_deadline_span(
                "weekday", "Please respond by Thursday, ahead of Friday's pour."
            ),
            "Thursday",
        )

    def test_weekday_embedded_in_end_of_day_phrase(self):
        self.assertEqual(
            gr._extract_deadline_span("weekday", "We need direction by end of day Wednesday."),
            "Wednesday",
        )

    def test_this_next_week(self):
        self.assertEqual(
            gr._extract_deadline_span("this_next_week", "A response next week would be appreciated."),
            "next week",
        )

    def test_end_of_stops_right_after_day_or_week_so_no_trailing_text_leaks_in(self):
        self.assertEqual(
            gr._extract_deadline_span(
                "end_of", "We need this by the end of the day since the crew mobilizes tomorrow."
            ),
            "by the end of the day",
        )

    def test_unresolvable_families_have_no_pattern_and_return_none(self):
        for family in UNRESOLVABLE_FAMILIES:
            self.assertIsNone(gr._extract_deadline_span(family, "whatever text is here"))

    def test_unresolvable_families_never_resolve_even_on_their_real_template_sentences(self):
        # test_unresolvable_families_have_no_pattern_and_return_none above only
        # proves dict membership with placeholder text; this exercises the
        # actual event-anchored/ambiguous sentences end to end through
        # resolve_deadline(), same as a resolvable family's real phrases are.
        from deadlines import resolve_deadline

        real_examples = {
            "none": "No specific response date is requested.",
            "event_anchored": "We need direction before the concrete pour.",
            "ambiguous": "A prompt response would be appreciated.",
        }
        for family, phrase in real_examples.items():
            self.assertIsNone(gr._extract_deadline_span(family, phrase))
            self.assertEqual(resolve_deadline(phrase, "2026-03-10"), (None, "unresolved"))

    def test_no_match_in_text_returns_none(self):
        self.assertIsNone(gr._extract_deadline_span("weekday", "There is no day name in this sentence."))

    def test_pattern_coverage_matches_exactly_the_resolvable_families(self):
        # Regression guard for the original bug class: a dropped or typo'd
        # family key here would silently reintroduce 100%-unresolved for
        # that one family, with nothing else failing.
        from templates import DEADLINE_FAMILIES

        resolvable = set(DEADLINE_FAMILIES) - set(UNRESOLVABLE_FAMILIES)
        self.assertEqual(set(gr._DEADLINE_SPAN_PATTERNS), resolvable)


class TestExtractThenResolveGoldenDates(unittest.TestCase):
    """Golden (family, sentence, date_submitted) -> expected ISO date table
    covering all six resolvable families end to end through the same
    extract-then-resolve path generate_one() uses. Earlier tests here only
    checked deadline_resolution_status, which is exactly why the weekday
    same-day-as-submission bug (7 days late) passed the full suite: status
    was "resolved_to_date" either way, only the date value was wrong.
    """

    def _extract_then_resolve(self, family: str, sentence: str, date_submitted_iso: str):
        from deadlines import resolve_deadline

        span = gr._extract_deadline_span(family, sentence)
        self.assertIsNotNone(span, f"expected a resolvable span in {sentence!r}")
        return resolve_deadline(span, date_submitted_iso)

    def test_relative_days(self):
        self.assertEqual(
            self._extract_then_resolve(
                "relative_days", "Please respond within five days so we can proceed.", "2026-03-10",
            ),
            ("2026-03-15", "resolved_to_date"),
        )

    def test_relative_weeks(self):
        self.assertEqual(
            self._extract_then_resolve(
                "relative_weeks", "Please respond within two weeks so we can hold the release date.", "2026-03-10",
            ),
            ("2026-03-24", "resolved_to_date"),
        )

    def test_business_days_skips_weekends(self):
        self.assertEqual(
            self._extract_then_resolve(
                "business_days", "Please respond within five business days so we can place the order.", "2026-03-10",
            ),
            ("2026-03-17", "resolved_to_date"),
        )

    def test_weekday_after_submission_day(self):
        # 2026-03-10 is a Tuesday.
        self.assertEqual(
            self._extract_then_resolve(
                "weekday", "Please respond by Thursday, ahead of Friday's pour.", "2026-03-10",
            ),
            ("2026-03-12", "resolved_to_date"),
        )

    def test_weekday_same_as_submission_day_resolves_to_today_not_next_week(self):
        # Regression for the reviewer-caught bug: 2026-03-12 is a Thursday,
        # so "by Thursday" said that same day must resolve to that same day,
        # not the following Thursday.
        self.assertEqual(
            self._extract_then_resolve(
                "weekday", "Please respond Thursday so we can keep the crew moving.", "2026-03-12",
            ),
            ("2026-03-12", "resolved_to_date"),
        )

    def test_this_next_week(self):
        # 2026-03-10 is a Tuesday; "this week" resolves to that week's Friday.
        self.assertEqual(
            self._extract_then_resolve(
                "this_next_week", "A response this week would be appreciated.", "2026-03-10",
            ),
            ("2026-03-13", "resolved_to_date"),
        )

    def test_end_of_day(self):
        self.assertEqual(
            self._extract_then_resolve(
                "end_of", "We need this by the end of the day since the crew mobilizes tomorrow.", "2026-03-10",
            ),
            ("2026-03-10", "resolved_to_date"),
        )

    def test_end_of_day_with_embedded_weekday_uses_the_end_of_pattern_not_the_bare_weekday(self):
        # Regression for the reviewer-caught finding that the weekday family
        # pattern discards the "end of day" qualifier when both a weekday
        # name and an end-of-day phrase share a sentence -- this asserts the
        # cell's own family (end_of) is what actually gets used to extract,
        # and that it yields the qualified span, not the bare day name.
        sentence = "We need direction by end of day Wednesday."
        self.assertEqual(
            gr._extract_deadline_span("end_of", sentence),
            "by end of day",
        )
        self.assertEqual(
            self._extract_then_resolve("end_of", sentence, "2026-03-11"),  # 2026-03-11 is a Wednesday
            ("2026-03-11", "resolved_to_date"),
        )


class TestDeadlineResolutionEndToEnd(unittest.TestCase):
    """Regression test for the bug where the full deadline_phrasing sentence
    (not just the resolvable span) was passed to resolve_deadline(), which
    made every record resolve as "unresolved" regardless of deadline_family.
    """

    def test_resolvable_families_produce_real_dates_and_unresolvable_ones_dont(self):
        stats = gr.corpus_stats(_RECORDS)
        counts = stats["deadline_resolution_status_counts"]
        self.assertIn("resolved_to_date", counts)
        self.assertGreater(counts["resolved_to_date"], 0)

        by_family_status: dict[str, set[str]] = {}
        for r in _RECORDS:
            family = r.provenance.cell["deadline_family"]
            by_family_status.setdefault(family, set()).add(r.label.deadline_resolution_status)

        for family in UNRESOLVABLE_FAMILIES:
            self.assertEqual(
                by_family_status.get(family, {"unresolved"}), {"unresolved"},
                f"{family!r} should never resolve to a date",
            )
        resolvable_families = set(by_family_status) - set(UNRESOLVABLE_FAMILIES)
        self.assertTrue(resolvable_families, "expected at least one resolvable family in the fixed-seed plan")
        for family in resolvable_families:
            self.assertEqual(
                by_family_status[family], {"resolved_to_date"},
                f"{family!r} should always resolve to a date",
            )

    def test_deadline_text_is_always_a_genuine_substring_of_the_rendered_thread(self):
        for r in _RECORDS:
            full_text = r.thread.render()
            if r.label.deadline_text != "none stated":
                self.assertIn(r.label.deadline_text, full_text)

    def test_deadline_phrase_evidence_rate_is_perfect_for_the_fixed_seed_plan(self):
        stats = gr.corpus_stats(_RECORDS)
        self.assertEqual(stats["deadline_phrase_evidence_rate"], 1.0)


class TestValidateRecord(unittest.TestCase):
    def test_full_fixed_seed_corpus_has_zero_validation_errors(self):
        errors = []
        for r in _RECORDS:
            errors.extend(f"{r.provenance.thread_id}: {e}" for e in gr.validate_record(r))
        self.assertEqual(errors, [])


class TestDeterminism(unittest.TestCase):
    def test_generating_the_same_entry_twice_is_byte_identical(self):
        entry = _PLAN.entries[0]
        a = gr.generate_one(entry, _TEMPLATES, _POOLS)
        b = gr.generate_one(entry, _TEMPLATES, _POOLS)
        self.assertEqual(a.thread, b.thread)
        self.assertEqual(a.label, b.label)
        self.assertEqual(a.provenance, b.provenance)


class TestGoldRoutingConsistency(unittest.TestCase):
    def test_every_record_assigned_reviewer_matches_its_template_gold_routing_table(self):
        from templates import gold_row_for_cell

        for r in _RECORDS:
            template = _TEMPLATES[r.provenance.template_id]
            row = gold_row_for_cell(template, _plan_cell_for(r))
            self.assertEqual(r.label.assigned_reviewer, row.assigned_reviewer)
            self.assertEqual(r.label.routing_rationale, row.routing_rationale)
            self.assertEqual(r.label.escalation, row.escalation)


def _plan_cell_for(record: gr.GeneratedRecord):
    return next(e.cell for e in _PLAN.entries if e.thread_id == record.provenance.thread_id)


class TestRoundTrip(unittest.TestCase):
    def test_write_then_read_preserves_thread_label_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            gr.write_corpus(_RECORDS, out_dir)
            back = {r.provenance.thread_id: r for r in gr.read_corpus(out_dir)}
            self.assertEqual(len(back), len(_RECORDS))
            for original in _RECORDS:
                rt = back[original.provenance.thread_id]
                self.assertEqual(rt.thread, original.thread)
                self.assertEqual(rt.label, original.label)
                self.assertEqual(rt.provenance, original.provenance)

    def test_read_corpus_with_explicit_thread_ids_filters_correctly(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d)
            gr.write_corpus(_RECORDS, out_dir)
            wanted = [_RECORDS[0].provenance.thread_id, _RECORDS[1].provenance.thread_id]
            back = gr.read_corpus(out_dir, thread_ids=wanted)
            self.assertEqual(sorted(r.provenance.thread_id for r in back), sorted(wanted))


if __name__ == "__main__":
    unittest.main()
