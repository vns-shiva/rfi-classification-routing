"""Unit tests for policy_audit.py. Run with:

    python -m unittest discover -s code/tests -t code

Every test builds its own corpus under a tempfile.TemporaryDirectory() and
that directory is removed automatically on exit (pass or fail) -- no fixture
is left behind, per the project's "delete test data after every test" rule.
Cell/provenance fixtures reuse real (template_id, Cell) pairs pulled from
stratification.expressible_pairs() rather than hand-rolled cell dicts, since
Cell.__post_init__ validates every axis against AXIS_VOCAB and a template_id
must actually resolve via load_templates() for divergence_census() to work --
mirroring test_policy_agreement.py's own fixture strategy.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import policy_audit as pa
from gating import gate
from routing_policy import NoRuleMatchedError, escalation_for, route_rfi
from schema import RFILabel, write_label
from stratification import Cell, expressible_pairs, gold_row_for_cell, load_templates
from vocabulary import csi_divisions_for_discipline


def _label(**overrides) -> RFILabel:
    base = dict(
        thread_id="rfi-001",
        rfi_type="design_clarification",
        primary_discipline="Structural",
        secondary_disciplines=[],
        csi_division="05 00 00 (Metals)",
        urgency="priority",
        question_summary="Confirm beam-to-column connection detail at gridline C4.",
        referenced_documents=["S-401", "S-501"],
        proposed_solution="—",
        cost_impact=False,
        schedule_impact=False,
        answer_in_documents=False,
        deadline_text="within two weeks",
        assigned_reviewer="Structural Engineer",
        routing_rationale="Structural connection detail; no cost/code trigger.",
        escalation=False,
        confidence=0.8,
        status="open",
        annotator="gold",
        deadline_iso=None,
        deadline_resolution_status="unresolved",
    )
    base.update(overrides)
    return RFILabel(**base)


def _write_corpus_thread(corpus_dir: Path, label: RFILabel, template_id: str, cell: Cell) -> None:
    labels_dir = corpus_dir / "labels"
    prov_dir = corpus_dir / "provenance"
    labels_dir.mkdir(parents=True, exist_ok=True)
    prov_dir.mkdir(parents=True, exist_ok=True)
    write_label(label, labels_dir / f"{label.thread_id}.json")
    prov = {"thread_id": label.thread_id, "template_id": template_id, "cell": cell.to_dict()}
    (prov_dir / f"{label.thread_id}.json").write_text(json.dumps(prov), encoding="utf-8")


class PolicyAuditTestCase(unittest.TestCase):
    """Shares one real load_templates()/expressible_pairs() pass across every
    test in this module -- loading templates from disk is unnecessary to
    repeat per test and every test here only reads from the returned pairs,
    never mutates them."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.templates = load_templates()
        cls.pairs = expressible_pairs(cls.templates)

    @classmethod
    def _pick_pair(cls, want_divergence: bool) -> tuple[str, Cell]:
        for template_id, cell in cls.pairs:
            row = gold_row_for_cell(cls.templates[template_id], cell)
            if row is not None and row.policy_divergence is want_divergence:
                return template_id, cell
        raise AssertionError(f"no expressible pair with policy_divergence={want_divergence!r} found")


class TestLoadCorpus(PolicyAuditTestCase):
    def test_reads_label_and_cross_referenced_provenance(self):
        template_id, cell = self._pick_pair(want_divergence=False)
        row = gold_row_for_cell(self.templates[template_id], cell)
        label = _label(
            thread_id="rfi-100",
            rfi_type=cell.rfi_type,
            primary_discipline=cell.primary_discipline,
            csi_division=sorted(csi_divisions_for_discipline(cell.primary_discipline))[0],
            urgency=cell.urgency,
            cost_impact=cell.cost_impact,
            schedule_impact=cell.schedule_impact,
            assigned_reviewer=row.assigned_reviewer,
            escalation=row.escalation,
        )
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            _write_corpus_thread(corpus_dir, label, template_id, cell)
            threads = pa.load_corpus(corpus_dir)

        self.assertEqual(1, len(threads))
        self.assertEqual("rfi-100", threads[0].thread_id)
        self.assertEqual(template_id, threads[0].template_id)
        self.assertEqual(cell, threads[0].cell)
        self.assertEqual(row.assigned_reviewer, threads[0].label.assigned_reviewer)

    def test_reads_every_thread_sorted_by_thread_id(self):
        template_id, cell = self._pick_pair(want_divergence=False)
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            for tid in ("rfi-003", "rfi-001", "rfi-002"):
                _write_corpus_thread(corpus_dir, _label(thread_id=tid), template_id, cell)
            threads = pa.load_corpus(corpus_dir)

        self.assertEqual(["rfi-001", "rfi-002", "rfi-003"], [t.thread_id for t in threads])

    def test_raises_on_invalid_gold_label(self):
        template_id, cell = self._pick_pair(want_divergence=False)
        with tempfile.TemporaryDirectory() as tmp:
            corpus_dir = Path(tmp)
            # empty question_summary fails validate_label's REQUIRED_NONEMPTY_FIELDS-adjacent check.
            _write_corpus_thread(corpus_dir, _label(question_summary=""), template_id, cell)
            with self.assertRaises(ValueError):
                pa.load_corpus(corpus_dir)


class TestRuleInventoryAndReachability(unittest.TestCase):
    def test_covers_the_full_768_cell_grid(self):
        section = pa.rule_inventory_and_reachability()
        self.assertEqual(768, section.summary["n_cells"])
        self.assertEqual(768, sum(section.summary["rule_fire_counts"].values()))

    def test_every_rule_id_fires_and_every_reviewer_is_reachable(self):
        from routing_policy import RULE_IDS, reachable_reviewers

        section = pa.rule_inventory_and_reachability()
        self.assertEqual([], section.summary["unreachable_rules"])
        self.assertEqual(set(RULE_IDS), set(section.summary["rule_fire_counts"].keys()))
        self.assertEqual(reachable_reviewers(), set(section.summary["reachable_reviewers"]))


class TestRuleHitCounts(unittest.TestCase):
    def test_counts_sum_to_thread_count_and_split_by_rfi_type(self):
        template_id, cell = ("t-fixture", None)  # unused; rule_hit_counts only reads label fields
        threads = [
            pa.CorpusThread(
                thread_id="a",
                template_id="t1",
                cell=None,
                label=_label(thread_id="a", rfi_type="code_compliance_question"),
            ),
            pa.CorpusThread(
                thread_id="b",
                template_id="t1",
                cell=None,
                label=_label(thread_id="b", rfi_type="design_clarification"),
            ),
        ]
        section = pa.rule_hit_counts(threads)
        self.assertEqual(2, section.summary["n_threads"])
        self.assertEqual(2, sum(section.summary["rule_fire_counts"].values()))
        self.assertEqual(1, section.details["rule_fire_counts_by_rfi_type"]["code_compliance_question"]["R1_code_compliance"])
        self.assertEqual(1, section.details["rule_fire_counts_by_rfi_type"]["design_clarification"]["R4_discipline_default"])


class TestEscalationConsistency(unittest.TestCase):
    def test_matches_when_gold_follows_the_shared_formula(self):
        threads = [
            pa.CorpusThread(
                thread_id="a",
                template_id="t1",
                cell=None,
                label=_label(thread_id="a", cost_impact=True, schedule_impact=True, urgency="routine", escalation=True),
            )
        ]
        section = pa.escalation_consistency(threads)
        self.assertEqual(1, section.summary["n_match"])
        self.assertEqual(0, section.summary["n_mismatch"])

    def test_mismatch_detected_when_gold_contradicts_the_formula(self):
        threads = [
            pa.CorpusThread(
                thread_id="a",
                template_id="t1",
                cell=None,
                label=_label(thread_id="a", cost_impact=False, schedule_impact=False, urgency="routine", escalation=True),
            )
        ]
        section = pa.escalation_consistency(threads)
        self.assertEqual(0, section.summary["n_match"])
        self.assertEqual(1, section.summary["n_mismatch"])
        self.assertEqual("a", section.details["mismatches"][0]["thread_id"])


class TestRuleContribution(unittest.TestCase):
    def test_ablating_a_specific_rule_flips_only_its_supporting_threads(self):
        threads = [
            pa.CorpusThread(thread_id="a", template_id="t1", cell=None, label=_label(thread_id="a", rfi_type="code_compliance_question")),
            pa.CorpusThread(thread_id="b", template_id="t1", cell=None, label=_label(thread_id="b", rfi_type="design_clarification")),
        ]
        section = pa.rule_contribution(threads)
        r1 = section.details["per_rule"]["R1_code_compliance"]
        self.assertEqual(1, r1["n_support"])
        self.assertEqual(1, r1["n_changed_if_removed"])
        self.assertEqual(0, r1["n_unroutable_if_removed"])

    def test_ablating_the_catch_all_rule_makes_its_support_cases_unroutable(self):
        threads = [
            pa.CorpusThread(thread_id="a", template_id="t1", cell=None, label=_label(thread_id="a", rfi_type="design_clarification"))
        ]
        section = pa.rule_contribution(threads)
        r4 = section.details["per_rule"]["R4_discipline_default"]
        self.assertEqual(1, r4["n_support"])
        self.assertEqual(1, r4["n_unroutable_if_removed"])
        self.assertEqual(1, r4["n_changed_if_removed"])

    def test_min_support_gate_flags_low_support_rules(self):
        threads = [
            pa.CorpusThread(thread_id="a", template_id="t1", cell=None, label=_label(thread_id="a", rfi_type="code_compliance_question"))
        ]
        section = pa.rule_contribution(threads)
        self.assertFalse(section.details["per_rule"]["R1_code_compliance"]["meets_min_support"])
        self.assertEqual(pa.MIN_RULE_SUPPORT, section.summary["min_rule_support"])


class TestDivergenceCensus(PolicyAuditTestCase):
    def test_consistent_nondivergent_thread_produces_no_case(self):
        template_id, cell = self._pick_pair(want_divergence=False)
        row = gold_row_for_cell(self.templates[template_id], cell)
        threads = [
            pa.CorpusThread(
                thread_id="a",
                template_id=template_id,
                cell=cell,
                label=_label(thread_id="a", assigned_reviewer=row.assigned_reviewer, escalation=row.escalation),
            )
        ]
        section, cases = pa.divergence_census(threads, self.templates)
        self.assertEqual([], cases)
        self.assertEqual(1, section.summary["n_agrees_with_policy"])
        self.assertEqual(0, section.summary["n_false_positive"])
        self.assertEqual(0, section.summary["n_missed"])

    def test_false_positive_flag_detected_when_gold_actually_agrees_with_policy(self):
        template_id, cell = self._pick_pair(want_divergence=True)
        policy = route_rfi(cell.rfi_type, cell.primary_discipline, cell.cost_impact, cell.schedule_impact, cell.urgency)
        threads = [
            pa.CorpusThread(
                thread_id="a",
                template_id=template_id,
                cell=cell,
                label=_label(thread_id="a", assigned_reviewer=policy.assigned_reviewer, escalation=policy.escalation),
            )
        ]
        section, cases = pa.divergence_census(threads, self.templates)
        self.assertEqual(1, len(cases))
        self.assertEqual("false_positive_divergence_flag", cases[0].kind)
        self.assertEqual(1, section.summary["n_false_positive"])

    def test_missed_divergence_flag_detected_when_gold_actually_diverges(self):
        template_id, cell = self._pick_pair(want_divergence=False)
        threads = [
            pa.CorpusThread(
                thread_id="a",
                template_id=template_id,
                cell=cell,
                label=_label(thread_id="a", assigned_reviewer="Code Official/AHJ Liaison", escalation=True),
            )
        ]
        section, cases = pa.divergence_census(threads, self.templates)
        self.assertEqual(1, len(cases))
        self.assertEqual("missed_divergence_flag", cases[0].kind)
        self.assertEqual(1, section.summary["n_missed"])

    def test_unresolvable_template_id_is_counted_and_skipped(self):
        template_id, cell = self._pick_pair(want_divergence=False)
        threads = [
            pa.CorpusThread(thread_id="a", template_id="not-a-real-template", cell=cell, label=_label(thread_id="a"))
        ]
        section, cases = pa.divergence_census(threads, self.templates)
        self.assertEqual([], cases)
        self.assertEqual(1, section.summary["n_unresolvable"])


class TestGateInteraction(unittest.TestCase):
    def test_every_escalated_thread_is_flagged(self):
        threads = [
            pa.CorpusThread(thread_id="a", template_id="t1", cell=None, label=_label(thread_id="a", escalation=True)),
            pa.CorpusThread(thread_id="b", template_id="t1", cell=None, label=_label(thread_id="b", escalation=False)),
        ]
        section = pa.gate_interaction(threads)
        self.assertEqual(1, section.summary["n_escalated"])
        self.assertEqual(1, section.summary["n_escalated_and_flagged"])
        self.assertEqual(0, section.summary["escalated_but_not_flagged"])

    def test_flagged_rule_counts_tracks_the_rule_that_fired(self):
        threads = [
            pa.CorpusThread(
                thread_id="a",
                template_id="t1",
                cell=None,
                label=_label(thread_id="a", rfi_type="code_compliance_question", escalation=True),
            )
        ]
        section = pa.gate_interaction(threads)
        self.assertEqual({"R1_code_compliance": 1}, section.details["flagged_rule_counts"])


class TestReportWriters(unittest.TestCase):
    def _tiny_report(self) -> pa.AuditReport:
        section = pa.AuditSection(name="s", summary={"n": 1, "rate": float("nan")}, details={"k": {"a": 1}})
        case = pa.DivergenceCase(
            thread_id="a",
            template_id="t1",
            cell={"rfi_type": "design_clarification"},
            gold_assigned_reviewer="Structural Engineer",
            gold_escalation=False,
            policy_assigned_reviewer="Architect of Record",
            policy_escalation=False,
            row_policy_divergence=True,
            actual_divergence=True,
            kind="missed_divergence_flag",
        )
        return pa.AuditReport(corpus_dir="c", n_threads=1, sections={"s": section}, divergence_cases=[case], strict_failures=["oops"])

    def test_json_report_round_trips_through_json_and_serializes_nan_as_null(self):
        report = self._tiny_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            pa.write_json_report(report, path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(1, data["n_threads"])
        self.assertIsNone(data["sections"]["s"]["summary"]["rate"])
        self.assertEqual(1, len(data["divergence_cases"]))
        self.assertEqual(["oops"], data["strict_failures"])

    def test_markdown_report_lists_sections_and_divergence_cases(self):
        report = self._tiny_report()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            pa.write_markdown_report(report, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("## s", text)
        self.assertIn("missed_divergence_flag", text)
        self.assertIn("Strict-mode failures", text)


class TestRunAuditAgainstRealCorpus(unittest.TestCase):
    """The one integration-style test in this module: runs the full audit
    against the actual committed corpus, matching test_policy_agreement.py's
    real-templates approach. A future corpus regeneration that breaks the
    non-circularity guarantee (S5) or the shared escalation formula (S3)
    should fail loudly here."""

    def test_real_corpus_has_no_strict_failures(self):
        corpus_dir = Path(__file__).resolve().parent.parent.parent / "pilot-data" / "corpus"
        report = pa.run_audit(corpus_dir)
        self.assertEqual([], report.strict_failures)
        self.assertEqual(240, report.n_threads)
        self.assertGreater(report.n_threads, report.sections["divergence_census"].summary["n_diverges_from_policy"])


if __name__ == "__main__":
    unittest.main()
