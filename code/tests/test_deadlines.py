"""Unit tests for deadlines.py's resolve_deadline(). Run with:

    python -m unittest discover -s code/tests -t code
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deadlines import resolve_deadline


class TestResolveDeadline(unittest.TestCase):
    def test_today(self):
        self.assertEqual(resolve_deadline("today", "2026-03-10"), ("2026-03-10", "resolved_to_date"))

    def test_asap_resolves_same_day_as_today(self):
        self.assertEqual(resolve_deadline("asap", "2026-03-10"), ("2026-03-10", "resolved_to_date"))

    def test_tomorrow(self):
        self.assertEqual(resolve_deadline("tomorrow", "2026-03-10"), ("2026-03-11", "resolved_to_date"))

    def test_weekday_after_submission_day(self):
        # 2026-03-10 is a Tuesday; "wednesday" -> the very next day
        self.assertEqual(resolve_deadline("wednesday", "2026-03-10"), ("2026-03-11", "resolved_to_date"))

    def test_weekday_same_as_submission_day_resolves_to_today(self):
        # "respond by Tuesday" said on a Tuesday means today, not a week out.
        self.assertEqual(resolve_deadline("tuesday", "2026-03-10"), ("2026-03-10", "resolved_to_date"))

    def test_this_week_resolves_to_friday_same_week(self):
        self.assertEqual(resolve_deadline("this week", "2026-03-10"), ("2026-03-13", "resolved_to_date"))

    def test_by_end_of_day(self):
        self.assertEqual(resolve_deadline("by end of day", "2026-03-10"), ("2026-03-10", "resolved_to_date"))

    def test_by_end_of_week(self):
        self.assertEqual(resolve_deadline("by end of week", "2026-03-10"), ("2026-03-13", "resolved_to_date"))

    def test_within_n_weeks(self):
        self.assertEqual(resolve_deadline("within two weeks", "2026-03-10"), ("2026-03-24", "resolved_to_date"))

    def test_within_n_days(self):
        self.assertEqual(resolve_deadline("within three days", "2026-03-10"), ("2026-03-13", "resolved_to_date"))

    def test_within_digit_days(self):
        self.assertEqual(resolve_deadline("within 5 days", "2026-03-10"), ("2026-03-15", "resolved_to_date"))

    def test_none_stated_is_always_unresolved_no_na_status_exists(self):
        # Unlike the sibling meeting-extraction paper's NO_DEADLINE_TYPES
        # gate, every rfi_type here structurally expects a response, so
        # there is no "n/a" branch to hit.
        self.assertEqual(resolve_deadline("none stated", "2026-03-10"), (None, "unresolved"))

    def test_empty_string_is_unresolved(self):
        self.assertEqual(resolve_deadline("", "2026-03-10"), (None, "unresolved"))

    def test_next_week_resolves_to_friday_of_the_following_week(self):
        # Symmetric with "this week" (2026-03-13): the Friday one week later.
        self.assertEqual(resolve_deadline("next week", "2026-03-10"), ("2026-03-20", "resolved_to_date"))

    def test_by_the_end_of_the_week(self):
        self.assertEqual(resolve_deadline("by the end of the week", "2026-03-10"), ("2026-03-13", "resolved_to_date"))

    def test_by_end_of_the_week(self):
        self.assertEqual(resolve_deadline("by end of the week", "2026-03-10"), ("2026-03-13", "resolved_to_date"))

    def test_within_a_week(self):
        self.assertEqual(resolve_deadline("within a week", "2026-03-10"), ("2026-03-17", "resolved_to_date"))

    def test_within_ten_days(self):
        self.assertEqual(resolve_deadline("within ten days", "2026-03-10"), ("2026-03-20", "resolved_to_date"))

    def test_within_fourteen_days_is_not_misread_as_four(self):
        self.assertEqual(resolve_deadline("within fourteen days", "2026-03-10"), ("2026-03-24", "resolved_to_date"))

    def test_within_twenty_one_days_is_not_misread_as_one(self):
        self.assertEqual(resolve_deadline("within twenty-one days", "2026-03-10"), ("2026-03-31", "resolved_to_date"))

    def test_within_0_days_resolves_to_submission_date_not_unresolved(self):
        self.assertEqual(resolve_deadline("within 0 days", "2026-03-10"), ("2026-03-10", "resolved_to_date"))

    def test_within_n_business_days_skips_weekends(self):
        # 2026-03-10 is a Tuesday. Ten *business* days lands on 2026-03-24
        # (two intervening weekends skipped), not the naive calendar-day
        # answer of 2026-03-20 that "within ten days" would give.
        self.assertEqual(
            resolve_deadline("within 10 business days", "2026-03-10"), ("2026-03-24", "resolved_to_date"),
        )

    def test_within_n_working_days_skips_weekends(self):
        self.assertEqual(
            resolve_deadline("within 5 working days", "2026-03-10"), ("2026-03-17", "resolved_to_date"),
        )

    def test_stray_digits_elsewhere_in_the_phrase_do_not_get_concatenated(self):
        # A naive digit-concatenation parser would read "2" and "15" as "215".
        self.assertEqual(
            resolve_deadline("within 2 weeks of the 15th", "2026-03-10"), ("2026-03-24", "resolved_to_date"),
        )

    def test_malformed_date_submitted_is_unresolved_not_raising(self):
        self.assertEqual(resolve_deadline("today", "not-a-date"), (None, "unresolved"))

    def test_event_anchored_phrase_is_unresolved(self):
        self.assertEqual(
            resolve_deadline("before the concrete pour", "2026-03-10"), (None, "unresolved"),
        )

    def test_unrecognized_phrase_is_unresolved_not_raising(self):
        self.assertEqual(resolve_deadline("whenever is convenient", "2026-03-10"), (None, "unresolved"))


if __name__ == "__main__":
    unittest.main()
