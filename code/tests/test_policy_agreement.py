"""Regression test: every template's gold_routing_table must agree with
routing_policy.route_rfi() exactly where it claims to (policy_divergence is
False) and must actually differ from it everywhere it claims a divergence
(policy_divergence is True). Run with:

    python -m unittest discover -s code/tests -t code

This is a template-authoring correctness check, not the corpus-level
gold-vs-policy audit templates.py's module docstring assigns to the
not-yet-built policy_audit.py (that one runs against generated RFILabel
rows after generate_rfis.py exists). This test instead walks every
expressible (template, Cell) pair now, using the same gold_row_for_cell()
lookup stratification.py itself relies on, so a template that mislabels a
row's policy_divergence flag or drifts out of sync with routing_policy.py
fails loudly instead of silently degrading the "does the model learn the
policy" evaluation the divergence flag exists to support.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from routing_policy import route_rfi
from stratification import expressible_pairs, gold_row_for_cell, load_templates


class TestGoldRoutingTableAgreesWithPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.templates = load_templates()
        cls.pairs = expressible_pairs(cls.templates)

    def test_every_expressible_pair_has_a_resolvable_gold_row(self):
        # expressible_pairs() already filters to cells where gold_row_for_cell()
        # is not None, but assert it explicitly so a future refactor of that
        # filter can't silently let unresolvable cells back in here.
        for template_id, cell in self.pairs:
            row = gold_row_for_cell(self.templates[template_id], cell)
            self.assertIsNotNone(row, f"{template_id} / {cell}")

    def test_policy_divergence_flag_agrees_with_route_rfi(self):
        mismatches = []
        for template_id, cell in self.pairs:
            row = gold_row_for_cell(self.templates[template_id], cell)
            policy = route_rfi(
                rfi_type=cell.rfi_type,
                primary_discipline=cell.primary_discipline,
                cost_impact=cell.cost_impact,
                schedule_impact=cell.schedule_impact,
                urgency=cell.urgency,
            )
            reviewer_matches = row.assigned_reviewer == policy.assigned_reviewer
            escalation_matches = row.escalation == policy.escalation
            fully_agrees = reviewer_matches and escalation_matches

            if row.policy_divergence and fully_agrees:
                mismatches.append(
                    f"{template_id} / {cell}: row is flagged policy_divergence=True "
                    f"but assigned_reviewer={row.assigned_reviewer!r} and "
                    f"escalation={row.escalation!r} both match route_rfi() "
                    f"({policy.assigned_reviewer!r}, {policy.escalation!r}) -- no actual divergence"
                )
            elif not row.policy_divergence and not fully_agrees:
                mismatches.append(
                    f"{template_id} / {cell}: row is flagged policy_divergence=False but "
                    f"assigned_reviewer={row.assigned_reviewer!r}/escalation={row.escalation!r} "
                    f"disagrees with route_rfi()'s {policy.assigned_reviewer!r}/{policy.escalation!r}"
                )

        self.assertEqual([], mismatches, "\n" + "\n".join(mismatches))

    def test_diverging_rows_carry_a_nonempty_divergence_reason(self):
        missing_reasons = []
        for template_id, cell in self.pairs:
            row = gold_row_for_cell(self.templates[template_id], cell)
            if row.policy_divergence and not row.divergence_reason.strip():
                missing_reasons.append(f"{template_id} / {cell}")
        self.assertEqual([], missing_reasons)


if __name__ == "__main__":
    unittest.main()
