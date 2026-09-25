"""Unit tests for llm_client.py: the rule-based stub end to end, and
ChatLLMClient's JSON-parsing/caching orchestration via a canned-response
double (no real API calls). Run with:

    python -m unittest discover -s code/tests -t code
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import (
    ChatLLMClient,
    ClassificationParseError,
    RawResponse,
    RuleBasedStubClient,
    _parse_dotenv_line,
    build_client,
)
from schema import RFILabel, RFIMessage, RFIThread, validate_label


def _thread(text: str, thread_id: str = "rfi-001") -> RFIThread:
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


class TestRuleBasedStubClient(unittest.TestCase):
    def test_rejects_unknown_condition(self):
        with self.assertRaises(ValueError):
            RuleBasedStubClient(condition="not_a_condition")

    def test_classify_produces_a_valid_label(self):
        client = RuleBasedStubClient()
        thread = _thread(
            "Structural drawing S-401 shows a moment connection at gridline C4 that "
            "conflicts with S-501's shear tab detail. This is urgent and will delay "
            "the schedule and affect cost if not resolved within two days."
        )
        label = client.classify(thread)
        self.assertEqual(label.thread_id, "rfi-001")
        self.assertEqual(label.primary_discipline, "Structural")
        self.assertEqual(label.rfi_type, "document_discrepancy")
        self.assertEqual(label.urgency, "urgent")
        self.assertTrue(label.cost_impact)
        self.assertTrue(label.schedule_impact)
        self.assertIn("S-401", label.referenced_documents)
        self.assertIn("S-501", label.referenced_documents)
        self.assertEqual(validate_label(label), [])

    def test_classify_raw_returns_label_and_raw_response(self):
        client = RuleBasedStubClient()
        thread = _thread("Please confirm the finish schedule for the lobby.")
        label, raw = client.classify_raw(thread)
        self.assertEqual(validate_label(label), [])
        self.assertIsInstance(raw, RawResponse)
        self.assertEqual(raw.model, "rule_based_stub")
        self.assertEqual(raw.payload, "")
        self.assertFalse(raw.cache_hit)
        # classify() is now the inherited LLMClient.classify() wrapper —
        # confirm it still returns the same label classify_raw() does.
        self.assertEqual(client.classify(thread), label)

    def test_finish_keyword_matches_architectural_not_general(self):
        # "finish" is in _DISCIPLINE_KEYWORDS["Architectural"], so this text
        # does NOT fall through to the General default (a prior version of
        # this test asserted General under an untested assumption, guarded
        # behind an `if` that made the assertion vacuous either way).
        client = RuleBasedStubClient()
        thread = _thread("Please confirm the finish schedule for the lobby.")
        label = client.classify(thread)
        self.assertEqual(label.primary_discipline, "Architectural")
        self.assertNotEqual(label.csi_division, "—")
        self.assertEqual(validate_label(label), [])

    def test_general_discipline_falls_back_to_dash_csi(self):
        # No discipline keyword table entry matches this text at all, so the
        # General default and its "—" CSI division both apply.
        client = RuleBasedStubClient()
        thread = _thread("Can you clarify what is expected here before Friday?")
        label = client.classify(thread)
        self.assertEqual(label.primary_discipline, "General")
        self.assertEqual(label.csi_division, "—")
        self.assertEqual(validate_label(label), [])

    def test_code_compliance_detection(self):
        client = RuleBasedStubClient()
        thread = _thread("Does this handrail detail meet IBC code requirements?")
        label = client.classify(thread)
        self.assertEqual(label.rfi_type, "code_compliance_question")

    def test_field_condition_conflict_is_not_swallowed_by_generic_conflict(self):
        # This text contains the literal word "conflicts", which would hit
        # the generic document_discrepancy branch first if the cascade were
        # not ordered most-specific-first.
        client = RuleBasedStubClient()
        thread = _thread(
            "The existing field condition at column C4 conflicts with what "
            "the structural drawings show."
        )
        label = client.classify(thread)
        self.assertEqual(label.rfi_type, "field_condition_conflict")

    def test_multi_word_within_deadline_is_extracted(self):
        # _DEADLINE_PATTERNS' "within" rule used to allow only a single word
        # between "within" and the unit (e.g. "within 10 days"), missing a
        # multi-word count like "business days".
        client = RuleBasedStubClient()
        thread = _thread("Please respond within 10 business days of receipt.")
        label = client.classify(thread)
        self.assertEqual(label.deadline_text, "within 10 business days")

    def test_within_the_past_is_not_treated_as_a_deadline(self):
        # A retrospective reference ("this came up within the past two
        # weeks") is not a forward-looking deadline; the bare "within ...
        # days/weeks" pattern used to misread it as one.
        client = RuleBasedStubClient()
        thread = _thread("This issue was raised within the past two weeks and needs resolution.")
        label = client.classify(thread)
        self.assertEqual(label.deadline_text, "none stated")

    def test_coordination_conflict_is_not_swallowed_by_generic_conflict(self):
        # Same trap as above, for the coordination_conflict branch.
        client = RuleBasedStubClient()
        thread = _thread(
            "There is a coordination conflict between the ductwork and the "
            "structural beam at gridline B3."
        )
        label = client.classify(thread)
        self.assertEqual(label.rfi_type, "coordination_conflict")


class TestParseDotenvLine(unittest.TestCase):
    def test_plain_key_value(self):
        self.assertEqual(_parse_dotenv_line("FOO=bar"), ("FOO", "bar"))

    def test_double_quoted_value_is_unwrapped(self):
        self.assertEqual(_parse_dotenv_line('FOO="bar baz"'), ("FOO", "bar baz"))

    def test_single_quoted_value_is_unwrapped(self):
        self.assertEqual(_parse_dotenv_line("FOO='bar baz'"), ("FOO", "bar baz"))

    def test_mismatched_quotes_are_left_alone(self):
        # A stray leading quote with no matching trailing one is not a
        # quoting convention — stripping just the first character would
        # silently corrupt the value instead of leaving it as-is.
        self.assertEqual(_parse_dotenv_line('FOO="bar'), ("FOO", '"bar'))

    def test_blank_and_comment_lines_return_none(self):
        self.assertIsNone(_parse_dotenv_line(""))
        self.assertIsNone(_parse_dotenv_line("   "))
        self.assertIsNone(_parse_dotenv_line("# a comment"))

    def test_line_without_equals_returns_none(self):
        self.assertIsNone(_parse_dotenv_line("not a valid line"))

    def test_surrounding_whitespace_on_key_and_value_is_stripped(self):
        self.assertEqual(_parse_dotenv_line("  FOO  =  bar  "), ("FOO", "bar"))

    def test_line_with_no_key_returns_none(self):
        # "=orphan" has no key at all; os.environ.setdefault("", ...) raises
        # ValueError, which would take down module import over one bad line.
        self.assertIsNone(_parse_dotenv_line("=orphan"))
        self.assertIsNone(_parse_dotenv_line("   =orphan"))


class _FakeChatClient(ChatLLMClient):
    """Canned-response double for exercising ChatLLMClient's shared
    orchestration (JSON parsing, field normalization, caching) without a
    real provider."""

    name = "fake"

    def __init__(self, response: str, **kwargs):
        super().__init__(**kwargs)
        self._response = response
        self.call_count = 0

    def _complete(self, system: str, user: str) -> str:
        self.call_count += 1
        return self._response


_VALID_RESPONSE = json.dumps({
    "rfi_type": "design_clarification",
    "primary_discipline": "Structural",
    "secondary_disciplines": [],
    "csi_division": "05 00 00 (Metals)",
    "urgency": "priority",
    "question_summary": "Confirm connection detail.",
    "referenced_documents": ["S-401"],
    "proposed_solution": "—",
    "cost_impact": False,
    "schedule_impact": True,
    "answer_in_documents": False,
    "deadline_text": "within two weeks",
    "assigned_reviewer": "Structural Engineer",
    "routing_rationale": "Structural detail question.",
    "escalation": False,
    "confidence": 0.9,
})


class TestChatLLMClient(unittest.TestCase):
    def test_classify_parses_a_valid_response(self):
        client = _FakeChatClient(_VALID_RESPONSE)
        label = client.classify(_thread("irrelevant"))
        self.assertEqual(label.primary_discipline, "Structural")
        self.assertEqual(validate_label(label), [])

    def test_strips_markdown_fences(self):
        fenced = f"```json\n{_VALID_RESPONSE}\n```"
        client = _FakeChatClient(fenced)
        label = client.classify(_thread("irrelevant"))
        self.assertEqual(label.rfi_type, "design_clarification")

    def test_strips_markdown_fences_on_a_single_line(self):
        # A fenced payload collapsed onto one line ("```json {...}```", no
        # embedded newline) used to leave the "```json" prefix in place
        # because the old implementation only stripped the leading fence
        # when it could split on "\n".
        fenced = f"```json {_VALID_RESPONSE}```"
        client = _FakeChatClient(fenced)
        label = client.classify(_thread("irrelevant"))
        self.assertEqual(label.rfi_type, "design_clarification")

    def test_strips_markdown_fences_on_a_single_line_without_language_tag(self):
        fenced = f"```{_VALID_RESPONSE}```"
        client = _FakeChatClient(fenced)
        label = client.classify(_thread("irrelevant"))
        self.assertEqual(label.rfi_type, "design_clarification")

    def test_strips_markdown_fences_with_a_space_before_the_language_tag(self):
        # "``` json\n...\n```" (a space between the fence and the tag) used
        # to defeat the old regex-based tag strip, which anchored the tag
        # directly onto the fence with no space allowed in between.
        fenced = f"``` json\n{_VALID_RESPONSE}\n```"
        client = _FakeChatClient(fenced)
        label = client.classify(_thread("irrelevant"))
        self.assertEqual(label.rfi_type, "design_clarification")

    def test_strips_markdown_fences_with_trailing_text_on_the_opening_line(self):
        # "```json output\n...\n```" — trailing text after the language tag
        # on the same line as the opening fence used to survive the old
        # regex-based strip, which only consumed a single tag token.
        fenced = f"```json output\n{_VALID_RESPONSE}\n```"
        client = _FakeChatClient(fenced)
        label = client.classify(_thread("irrelevant"))
        self.assertEqual(label.rfi_type, "design_clarification")

    def test_scalar_list_fields_are_normalized(self):
        obj = json.loads(_VALID_RESPONSE)
        obj["secondary_disciplines"] = "Mechanical"  # scalar instead of list
        client = _FakeChatClient(json.dumps(obj))
        label = client.classify(_thread("irrelevant"))
        self.assertEqual(label.secondary_disciplines, ["Mechanical"])

    def test_malformed_json_raises_runtime_error(self):
        client = _FakeChatClient("not json at all")
        with self.assertRaises(RuntimeError):
            client.classify(_thread("irrelevant"))

    def test_non_object_json_raises_runtime_error(self):
        client = _FakeChatClient("[1, 2, 3]")
        with self.assertRaises(RuntimeError):
            client.classify(_thread("irrelevant"))

    def test_caching_avoids_a_second_complete_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            client = _FakeChatClient(_VALID_RESPONSE, cache_dir=cache_dir)
            thread = _thread("same text every time", thread_id="rfi-001")
            client.classify(thread)
            client.classify(thread)
            self.assertEqual(client.call_count, 1)
            self.assertEqual(client.stats["calls"], 1)
            self.assertEqual(client.stats["cache_hits"], 1)

    def test_different_models_sharing_a_cache_dir_do_not_collide(self):
        # Same provider name, different .model — the cache key must
        # distinguish them or one model's cached response would silently
        # stand in for the other's.
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            thread = _thread("same text every time", thread_id="rfi-001")

            client_a = _FakeChatClient(_VALID_RESPONSE, cache_dir=cache_dir)
            client_a.model = "model-a"
            client_a.classify(thread)

            other_response = json.loads(_VALID_RESPONSE)
            other_response["primary_discipline"] = "Mechanical"
            client_b = _FakeChatClient(json.dumps(other_response), cache_dir=cache_dir)
            client_b.model = "model-b"
            label_b = client_b.classify(thread)

            self.assertEqual(client_b.call_count, 1)
            self.assertEqual(label_b.primary_discipline, "Mechanical")

    def test_empty_response_is_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            client = _FakeChatClient("", cache_dir=cache_dir)
            thread = _thread("same text every time", thread_id="rfi-001")
            with self.assertRaises(RuntimeError):
                client.classify(thread)
            with self.assertRaises(RuntimeError):
                client.classify(thread)
            # Both calls actually hit _complete; an empty payload must never
            # be served back out of the cache as a fake "hit".
            self.assertEqual(client.call_count, 2)
            self.assertEqual(client.stats["cache_hits"], 0)

    def test_explicit_json_null_falls_back_to_default_not_none(self):
        obj = json.loads(_VALID_RESPONSE)
        obj["urgency"] = None
        obj["confidence"] = None
        obj["cost_impact"] = None
        client = _FakeChatClient(json.dumps(obj))
        label = client.classify(_thread("irrelevant"))
        self.assertEqual(label.urgency, "")
        self.assertEqual(label.confidence, 0.5)
        self.assertEqual(label.cost_impact, False)

    def test_string_false_is_not_coerced_to_true(self):
        # bool("false") is True in Python; a model emitting the JSON string
        # "false" for a boolean field must not have it silently inverted.
        obj = json.loads(_VALID_RESPONSE)
        obj["cost_impact"] = "false"
        obj["schedule_impact"] = "true"
        client = _FakeChatClient(json.dumps(obj))
        label = client.classify(_thread("irrelevant"))
        self.assertFalse(label.cost_impact)
        self.assertTrue(label.schedule_impact)

    def test_non_numeric_confidence_falls_back_to_default(self):
        obj = json.loads(_VALID_RESPONSE)
        obj["confidence"] = "high"
        client = _FakeChatClient(json.dumps(obj))
        label = client.classify(_thread("irrelevant"))
        self.assertEqual(label.confidence, 0.5)


class TestChatLLMClientClassifyRaw(unittest.TestCase):
    def test_classify_raw_returns_label_and_raw_response_on_a_miss(self):
        client = _FakeChatClient(_VALID_RESPONSE)
        client.model = "fake-model-1"
        label, raw = client.classify_raw(_thread("irrelevant"))
        self.assertEqual(validate_label(label), [])
        self.assertIsInstance(raw, RawResponse)
        self.assertEqual(raw.payload, _VALID_RESPONSE)
        self.assertEqual(raw.model, "fake-model-1")
        self.assertFalse(raw.cache_hit)

    def test_classify_raw_reports_cache_hit_true_on_second_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            client = _FakeChatClient(_VALID_RESPONSE, cache_dir=cache_dir)
            thread = _thread("same text every time", thread_id="rfi-001")

            _, raw_miss = client.classify_raw(thread)
            _, raw_hit = client.classify_raw(thread)

            self.assertFalse(raw_miss.cache_hit)
            self.assertTrue(raw_hit.cache_hit)
            self.assertEqual(client.call_count, 1)

    def test_classify_still_works_via_inherited_wrapper(self):
        client = _FakeChatClient(_VALID_RESPONSE)
        label = client.classify(_thread("irrelevant"))
        self.assertEqual(label.primary_discipline, "Structural")
        self.assertEqual(validate_label(label), [])

    def test_malformed_json_raises_classification_parse_error_with_payload_and_detail(self):
        client = _FakeChatClient("not json at all")
        with self.assertRaises(ClassificationParseError) as ctx:
            client.classify_raw(_thread("irrelevant"))
        self.assertEqual(ctx.exception.payload, "not json at all")
        self.assertTrue(ctx.exception.detail)
        # ClassificationParseError must still satisfy the existing
        # assertRaises(RuntimeError) contract used elsewhere in this file.
        self.assertIsInstance(ctx.exception, RuntimeError)

    def test_non_object_json_raises_classification_parse_error(self):
        client = _FakeChatClient("[1, 2, 3]")
        with self.assertRaises(ClassificationParseError) as ctx:
            client.classify_raw(_thread("irrelevant"))
        self.assertEqual(ctx.exception.payload, "[1, 2, 3]")

    def test_repair_suffix_changes_the_cache_key(self):
        # classify.py's bounded repair retry appends a repair instruction to
        # the user prompt specifically so a retry after a parse failure
        # doesn't replay the same cached (system, user) response — confirm
        # that a non-empty repair_suffix actually produces a cache miss
        # against a call with no suffix, even for the same thread/cache_dir.
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            client = _FakeChatClient(_VALID_RESPONSE, cache_dir=cache_dir)
            thread = _thread("same text every time", thread_id="rfi-001")

            client.classify_raw(thread)
            client.classify_raw(thread, repair_suffix="\n\nPlease fix the JSON.")

            self.assertEqual(client.call_count, 2)

    def test_repair_suffix_is_appended_to_the_user_prompt(self):
        seen_users = []

        class _RecordingClient(ChatLLMClient):
            name = "recording"

            def _complete(self, system: str, user: str) -> str:
                seen_users.append(user)
                return _VALID_RESPONSE

        client = _RecordingClient()
        client.classify_raw(_thread("irrelevant"), repair_suffix="\n\nREPAIR TEXT")
        self.assertTrue(seen_users[0].endswith("\n\nREPAIR TEXT"))

    def test_rule_based_stub_accepts_and_ignores_repair_suffix(self):
        # RuleBasedStubClient never raises ClassificationParseError, so it
        # never actually gets retried, but it must still accept the shared
        # LLMClient.classify_raw() keyword without erroring.
        client = RuleBasedStubClient()
        label, raw = client.classify_raw(_thread("irrelevant"), repair_suffix="ignored")
        self.assertEqual(validate_label(label), [])
        self.assertEqual(raw.payload, "")


class TestBuildClient(unittest.TestCase):
    def test_stub(self):
        client = build_client("stub")
        self.assertIsInstance(client, RuleBasedStubClient)

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            build_client("not_a_provider")


if __name__ == "__main__":
    unittest.main()
