"""Unit tests for classify.py: the bounded repair retry and per-thread
deadline resolution built on top of llm_client.LLMClient.classify_raw(). Run
with:

    python -m unittest discover -s code/tests -t code
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classify import (
    classify_corpus,
    classify_thread,
    outcome_to_dict,
    resolve_label_deadline,
    summarize_outcomes,
)
from llm_client import ChatLLMClient, RuleBasedStubClient
from schema import RFILabel, RFIMessage, RFIThread, validate_label

_VALID_JSON = """{
  "rfi_type": "design_clarification",
  "primary_discipline": "Structural",
  "secondary_disciplines": [],
  "csi_division": "05 00 00 (Metals)",
  "urgency": "priority",
  "question_summary": "Confirm connection detail.",
  "referenced_documents": ["S-401"],
  "proposed_solution": "—",
  "cost_impact": false,
  "schedule_impact": true,
  "answer_in_documents": false,
  "deadline_text": "tomorrow",
  "assigned_reviewer": "Structural Engineer",
  "routing_rationale": "Structural detail question.",
  "escalation": false,
  "confidence": 0.9
}"""

_MALFORMED_JSON = "not json at all"


def _thread(text: str = "Please confirm the connection detail.", thread_id: str = "rfi-001") -> RFIThread:
    return RFIThread(
        thread_id=thread_id,
        project="Riverside Medical Office Building",
        rfi_number="RFI-014",
        date_submitted="2025-03-14",
        submitted_by_role="GC Superintendent",
        subject="Test subject",
        messages=[
            RFIMessage(1, "GC Superintendent", "Marcus Webb", "2025-03-14", "initial_question", text),
        ],
    )


class _ScriptedChatClient(ChatLLMClient):
    """Canned-response double that returns a different response on each
    successive _complete() call (unlike test_llm_client._FakeChatClient,
    which always returns the same fixed response) — needed to exercise
    classify_thread()'s repair retry, which requires the second call's
    response to differ from the first. Also records every user prompt it was
    called with, so a test can assert the retry call's prompt actually
    carries the repair instruction."""

    name = "scripted"

    def __init__(self, responses: list[str], **kwargs):
        super().__init__(**kwargs)
        self._responses = list(responses)
        self.seen_users: list[str] = []

    def _complete(self, system: str, user: str) -> str:
        self.seen_users.append(user)
        return self._responses.pop(0)


class TestResolveLabelDeadline(unittest.TestCase):
    def test_mutates_deadline_iso_and_status(self):
        thread = _thread()
        label = RFILabel(
            thread_id="rfi-001", rfi_type="design_clarification", primary_discipline="Structural",
            secondary_disciplines=[], csi_division="05 00 00 (Metals)", urgency="priority",
            question_summary="Confirm connection detail.", referenced_documents=["S-401"],
            proposed_solution="—", cost_impact=False, schedule_impact=True, answer_in_documents=False,
            deadline_text="tomorrow", assigned_reviewer="Structural Engineer",
            routing_rationale="Structural detail question.", escalation=False,
        )
        self.assertIsNone(label.deadline_iso)
        self.assertEqual(label.deadline_resolution_status, "unresolved")

        resolve_label_deadline(label, thread)

        self.assertEqual(label.deadline_iso, "2025-03-15")
        self.assertEqual(label.deadline_resolution_status, "resolved_to_date")
        self.assertEqual(validate_label(label), [])

    def test_none_stated_stays_unresolved(self):
        thread = _thread()
        label = RFILabel(
            thread_id="rfi-001", rfi_type="design_clarification", primary_discipline="Structural",
            secondary_disciplines=[], csi_division="05 00 00 (Metals)", urgency="routine",
            question_summary="Confirm connection detail.", referenced_documents=[],
            proposed_solution="—", cost_impact=False, schedule_impact=False, answer_in_documents=False,
            deadline_text="none stated", assigned_reviewer="Structural Engineer",
            routing_rationale="Structural detail question.", escalation=False,
        )
        resolve_label_deadline(label, thread)
        self.assertIsNone(label.deadline_iso)
        self.assertEqual(label.deadline_resolution_status, "unresolved")
        self.assertEqual(validate_label(label), [])


class TestClassifyThreadWithRuleBasedStub(unittest.TestCase):
    def test_succeeds_on_first_attempt_with_resolved_deadline(self):
        client = RuleBasedStubClient()
        thread = _thread("This is urgent, please respond within two days.")
        outcome = classify_thread(client, thread)
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.attempts, 1)
        self.assertFalse(outcome.repaired)
        self.assertIsNone(outcome.error_detail)
        self.assertEqual(len(outcome.raw_responses), 1)
        self.assertIsNotNone(outcome.label)
        self.assertEqual(validate_label(outcome.label), [])


class TestClassifyThreadRepairRetry(unittest.TestCase):
    def test_repairs_after_a_parse_error(self):
        client = _ScriptedChatClient([_MALFORMED_JSON, _VALID_JSON])
        thread = _thread()

        outcome = classify_thread(client, thread)

        self.assertEqual(outcome.status, "ok")
        self.assertEqual(outcome.attempts, 2)
        self.assertTrue(outcome.repaired)
        self.assertEqual(len(outcome.raw_responses), 2)
        self.assertEqual(outcome.raw_responses[0].payload, _MALFORMED_JSON)
        self.assertEqual(validate_label(outcome.label), [])
        # deadline_text "tomorrow" from _VALID_JSON must have been resolved
        self.assertEqual(outcome.label.deadline_resolution_status, "resolved_to_date")

        # the retry's user prompt must carry the repair instruction, and
        # differ from the first call's prompt (else it would hit the same
        # cache key and just replay the same failure against a real cache).
        self.assertEqual(len(client.seen_users), 2)
        self.assertNotEqual(client.seen_users[0], client.seen_users[1])
        self.assertIn("previous response could not be parsed", client.seen_users[1])
        self.assertIn(_MALFORMED_JSON, client.seen_users[1])

    def test_exhausts_repair_attempts_and_reports_parse_failed(self):
        client = _ScriptedChatClient([_MALFORMED_JSON, _MALFORMED_JSON])
        thread = _thread()

        outcome = classify_thread(client, thread, max_repair_attempts=1)

        self.assertEqual(outcome.status, "parse_failed")
        self.assertIsNone(outcome.label)
        self.assertEqual(outcome.attempts, 2)
        self.assertFalse(outcome.repaired)
        self.assertEqual(len(outcome.raw_responses), 2)
        self.assertIsNotNone(outcome.error_detail)


class TestClassifyCorpus(unittest.TestCase):
    def test_returns_one_outcome_per_thread(self):
        client = RuleBasedStubClient()
        threads = [_thread(thread_id="rfi-001"), _thread(thread_id="rfi-002")]
        outcomes = classify_corpus(client, threads)
        self.assertEqual([o.thread_id for o in outcomes], ["rfi-001", "rfi-002"])
        self.assertTrue(all(o.status == "ok" for o in outcomes))


class TestOutcomeToDict(unittest.TestCase):
    def test_ok_outcome_includes_the_label_dict(self):
        client = RuleBasedStubClient()
        outcome = classify_thread(client, _thread())
        d = outcome_to_dict(outcome)
        self.assertEqual(d["thread_id"], "rfi-001")
        self.assertEqual(d["status"], "ok")
        self.assertIsNotNone(d["label"])
        self.assertEqual(d["label"]["thread_id"], "rfi-001")

    def test_parse_failed_outcome_has_null_label(self):
        client = _ScriptedChatClient([_MALFORMED_JSON, _MALFORMED_JSON])
        outcome = classify_thread(client, _thread(), max_repair_attempts=1)
        d = outcome_to_dict(outcome)
        self.assertEqual(d["status"], "parse_failed")
        self.assertIsNone(d["label"])
        self.assertIsNotNone(d["error_detail"])


class TestSummarizeOutcomes(unittest.TestCase):
    def test_counts_ok_parse_failed_and_repaired(self):
        client_ok = RuleBasedStubClient()
        ok_outcome = classify_thread(client_ok, _thread(thread_id="rfi-001"))

        repaired_client = _ScriptedChatClient([_MALFORMED_JSON, _VALID_JSON])
        repaired_outcome = classify_thread(repaired_client, _thread(thread_id="rfi-002"))

        failed_client = _ScriptedChatClient([_MALFORMED_JSON, _MALFORMED_JSON])
        failed_outcome = classify_thread(failed_client, _thread(thread_id="rfi-003"), max_repair_attempts=1)

        summary = summarize_outcomes([ok_outcome, repaired_outcome, failed_outcome])
        self.assertEqual(summary.total, 3)
        self.assertEqual(summary.ok, 2)
        self.assertEqual(summary.parse_failed, 1)
        self.assertEqual(summary.repaired, 1)


if __name__ == "__main__":
    unittest.main()
