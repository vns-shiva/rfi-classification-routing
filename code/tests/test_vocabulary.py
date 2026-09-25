"""Unit tests for vocabulary.py's closed vocabularies and helper functions.
Run with:

    python -m unittest discover -s code/tests -t code
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vocabulary import (
    DISCIPLINE_TO_CSI,
    DISCIPLINES,
    URGENCY_TIERS,
    csi_divisions_for_discipline,
    disciplines_for_csi,
    urgency_rank,
)


class TestCsiDivisionsForDiscipline(unittest.TestCase):
    def test_returns_the_mapped_set(self):
        self.assertEqual(csi_divisions_for_discipline("Mechanical"), {"23 00 00 (HVAC)"})

    def test_general_has_no_divisions(self):
        self.assertEqual(csi_divisions_for_discipline("General"), set())

    def test_unknown_discipline_returns_empty_set_not_raising(self):
        self.assertEqual(csi_divisions_for_discipline("not_a_discipline"), set())

    def test_returns_a_copy_not_the_live_set(self):
        result = csi_divisions_for_discipline("Mechanical")
        result.add("99 00 00 (Bogus)")
        self.assertNotIn("99 00 00 (Bogus)", DISCIPLINE_TO_CSI["Mechanical"])


class TestDisciplinesForCsi(unittest.TestCase):
    def test_shared_division_returns_both_disciplines(self):
        # "21 00 00 (Fire Suppression)" is legitimately shared by Plumbing
        # and Fire_Protection per the module's own docstring.
        self.assertEqual(
            disciplines_for_csi("21 00 00 (Fire Suppression)"),
            {"Plumbing", "Fire_Protection"},
        )

    def test_unshared_division_returns_one_discipline(self):
        self.assertEqual(disciplines_for_csi("23 00 00 (HVAC)"), {"Mechanical"})

    def test_unknown_division_returns_empty_set(self):
        self.assertEqual(disciplines_for_csi("99 00 00 (Bogus)"), set())


class TestUrgencyRank(unittest.TestCase):
    def test_ranks_match_ordinal_position(self):
        for i, tier in enumerate(URGENCY_TIERS):
            self.assertEqual(urgency_rank(tier), i)

    def test_unknown_tier_raises(self):
        with self.assertRaises(ValueError):
            urgency_rank("whenever")


class TestDisciplineToCsiInternalConsistency(unittest.TestCase):
    def test_every_discipline_has_a_csi_entry(self):
        self.assertEqual(set(DISCIPLINE_TO_CSI.keys()), DISCIPLINES)


if __name__ == "__main__":
    unittest.main()
