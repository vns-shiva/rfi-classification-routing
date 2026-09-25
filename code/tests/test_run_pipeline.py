"""Unit tests for run_pipeline.py: the end-to-end classify-a-split-and-write-
a-reviewable-bundle CLI built on classify.py + gating.py. Every test here
uses RuleBasedStubClient or an in-process scripted ChatLLMClient double —
never a real client/API key. Run with:

    python -m unittest discover -s code/tests -t code
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_pipeline
from llm_client import ChatLLMClient, RuleBasedStubClient
from schema import RFIMessage, RFIThread, read_label, write_thread

_CODE_DIR = Path(__file__).resolve().parent.parent


def _thread(thread_id: str, text: str = "Please confirm the connection detail.") -> RFIThread:
    return RFIThread(
        thread_id=thread_id,
        project="Riverside Medical Office Building",
        rfi_number="RFI-014",
        date_submitted="2025-03-14",
        submitted_by_role="GC Superintendent",
        subject="Test subject",
        messages=[RFIMessage(1, "GC Superintendent", "Marcus Webb", "2025-03-14", "initial_question", text)],
    )


def _make_corpus(root: Path, thread_specs: list[tuple[str, str]]) -> Path:
    """thread_specs: list of (thread_id, text). Writes only threads/, no
    labels/ — run_pipeline.py never reads gold labels, only threads."""
    corpus_dir = root / "corpus"
    (corpus_dir / "threads").mkdir(parents=True)
    for thread_id, text in thread_specs:
        write_thread(_thread(thread_id, text), corpus_dir / "threads" / f"{thread_id}.json")
    return corpus_dir


def _sha256_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


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

_LOW_CONFIDENCE_JSON = _VALID_JSON.replace('"confidence": 0.9', '"confidence": 0.1')
_MALFORMED_JSON = "not json at all"


class _ScriptedChatClient(ChatLLMClient):
    """Canned-response double keyed by thread_id, so a test can control the
    outcome (ok / parse_failed / raises) per thread in one batch run — needed
    because run_pipeline.run() classifies a whole list of threads at once."""

    name = "scripted"
    model = "scripted-model"

    def __init__(self, responses_by_thread: dict[str, list[str] | Exception], **kwargs):
        super().__init__(**kwargs)
        self._responses_by_thread = {k: (list(v) if isinstance(v, list) else v) for k, v in responses_by_thread.items()}
        self._current_thread_id: str | None = None

    def classify_raw(self, thread, repair_suffix: str = ""):
        self._current_thread_id = thread.thread_id
        return super().classify_raw(thread, repair_suffix=repair_suffix)

    def _complete(self, system: str, user: str) -> str:
        entry = self._responses_by_thread[self._current_thread_id]
        if isinstance(entry, Exception):
            raise entry
        return entry.pop(0)


class TestSelectThreadPaths(unittest.TestCase):
    def test_split_filters_by_prefix(self, tmp=None):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a"), ("dev-0002", "a"), ("eval-0001", "a")])
            dev_paths = run_pipeline.select_thread_paths(corpus, "dev", None, 0)
            self.assertEqual([p.stem for p in dev_paths], ["dev-0001", "dev-0002"])
            all_paths = run_pipeline.select_thread_paths(corpus, "all", None, 0)
            self.assertEqual(len(all_paths), 3)

    def test_limit_truncates(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a"), ("dev-0002", "a"), ("dev-0003", "a")])
            paths = run_pipeline.select_thread_paths(corpus, "dev", None, 2)
            self.assertEqual(len(paths), 2)

    def test_thread_id_overrides_split_and_limit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a"), ("eval-0001", "a")])
            paths = run_pipeline.select_thread_paths(corpus, "dev", ["eval-0001"], 1)
            self.assertEqual([p.stem for p in paths], ["eval-0001"])

    def test_missing_thread_id_raises_corpus_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a")])
            with self.assertRaises(run_pipeline.CorpusError):
                run_pipeline.select_thread_paths(corpus, "all", ["nope-0001"], 0)

    def test_empty_selection_raises_corpus_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a")])
            with self.assertRaises(run_pipeline.CorpusError):
                run_pipeline.select_thread_paths(corpus, "eval", None, 0)

    def test_missing_threads_dir_raises_corpus_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td) / "corpus"
            corpus.mkdir()
            with self.assertRaises(run_pipeline.CorpusError):
                run_pipeline.select_thread_paths(corpus, "all", None, 0)

    def test_repeated_thread_id_is_deduped_not_a_corpus_error(self):
        # A literal repeat of the same --thread-id value is the same
        # file/id, not a genuine collision — load_threads()'s duplicate-id
        # check would otherwise reject this even though nothing is wrong.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a"), ("dev-0002", "a")])
            paths = run_pipeline.select_thread_paths(
                corpus, "all", ["dev-0001", "dev-0002", "dev-0001"], 0
            )
            self.assertEqual([p.stem for p in paths], ["dev-0001", "dev-0002"])


class TestLoadThreads(unittest.TestCase):
    def test_duplicate_thread_id_across_files_raises_corpus_error(self):
        # Two distinct filenames both claiming the same thread_id is no
        # longer reachable through a normal path list: the thread_id/stem
        # equality check above would catch it first as a mismatch, since at
        # most one filename can equal a given thread_id. The duplicate-id
        # check is still live defense-in-depth against any path list with a
        # repeated entry (e.g. a future select_thread_paths() bug, or a
        # hand-built list) — exercised directly here rather than through two
        # distinct filenames on disk.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a")])
            paths = sorted((corpus / "threads").glob("*.json"))
            with self.assertRaises(run_pipeline.CorpusError):
                run_pipeline.load_threads(paths + paths)

    def test_no_duplicates_loads_cleanly(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a"), ("dev-0002", "b")])
            paths = sorted((corpus / "threads").glob("*.json"))
            threads = run_pipeline.load_threads(paths)
            self.assertEqual([t.thread_id for t in threads], ["dev-0001", "dev-0002"])

    def test_thread_id_mismatched_with_filename_raises_corpus_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a")])
            # Content thread_id disagrees with the filename it was loaded
            # from, even without any path-traversal characters involved.
            write_thread(_thread("dev-0002"), corpus / "threads" / "dev-0001.json")
            paths = sorted((corpus / "threads").glob("*.json"))
            with self.assertRaises(run_pipeline.CorpusError):
                run_pipeline.load_threads(paths)

    def test_cli_rejects_traversal_via_thread_id_content_and_leaves_corpus_untouched(self):
        # A corpus JSON file's thread_id is attacker/author-controlled content,
        # not derived from its filename. write_run() used to write labels_*/
        # and raw/ files keyed by this value verbatim, so a value like
        # "../../corpus/threads/dev-0001" could overwrite a gold thread file.
        # load_threads() must now refuse this before any output is written.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
            traversal_id = "../../corpus/threads/dev-0001"
            write_thread(_thread(traversal_id), corpus / "threads" / "dev-0002.json")
            before = _sha256_tree(corpus)

            rc = run_pipeline.main([
                "--corpus", str(corpus), "--client", "stub", "--out", str(tmp_root / "out"),
                "--no-cache",
            ])

            self.assertEqual(rc, 2)
            self.assertEqual(before, _sha256_tree(corpus))
            self.assertFalse((tmp_root / "out").exists())


class TestValidateAnnotatorId(unittest.TestCase):
    def test_accepts_safe_ids(self):
        for value in ("pipeline", "annotator_b", "abc-123", "A1"):
            run_pipeline.validate_annotator_id(value)  # must not raise

    def test_rejects_empty(self):
        with self.assertRaises(run_pipeline.CorpusError):
            run_pipeline.validate_annotator_id("")

    def test_rejects_path_separators_and_dots(self):
        for value in ("../../../corpus/predicted", "a/b", "a\\b", "..", "."):
            with self.assertRaises(run_pipeline.CorpusError):
                run_pipeline.validate_annotator_id(value)

    def test_cli_rejects_unsafe_annotator_id_exit_code_two_and_nothing_escapes(self):
        # out_dir / f"labels_{annotator_id}" with annotator_id
        # "../../../corpus/predicted" resolves against tmp_root/out as
        # tmp_root/corpus/predicted (it walks up out of "labels_..", out of
        # "out", then back down into "corpus/predicted") — NOT
        # tmp_root/predicted. A prior version of this test asserted against
        # the wrong path, so it would have passed even with the escape live.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
            before = _sha256_tree(corpus)

            rc = run_pipeline.main([
                "--corpus", str(corpus), "--client", "stub", "--out", str(tmp_root / "out"),
                "--no-cache", "--annotator-id", "../../../corpus/predicted",
            ])

            self.assertEqual(rc, 2)
            self.assertEqual(before, _sha256_tree(corpus))
            self.assertFalse((corpus / "predicted").exists())
            self.assertFalse((tmp_root / "out").exists())
            self.assertFalse((tmp_root / "predicted").exists())


class TestLabelsDir(unittest.TestCase):
    def test_returns_expected_path_for_safe_id(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            self.assertEqual(run_pipeline._labels_dir(out, "pipeline"), out / "labels_pipeline")

    def test_raises_if_id_would_escape_out_dir(self):
        # Backstop behind validate_annotator_id(): even an annotator_id that
        # somehow reached here containing a path separator must not produce
        # a path outside out_dir.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            with self.assertRaises(run_pipeline.OutputDirUnsafeError):
                run_pipeline._labels_dir(out, "../escaped")


class TestAssertOutDirSafe(unittest.TestCase):
    def test_refuses_corpus_dir_itself(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a")])
            with self.assertRaises(run_pipeline.OutputDirUnsafeError):
                run_pipeline.assert_out_dir_safe(corpus, corpus)

    def test_refuses_path_inside_corpus_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a")])
            with self.assertRaises(run_pipeline.OutputDirUnsafeError):
                run_pipeline.assert_out_dir_safe(corpus / "threads", corpus)

    def test_refuses_a_dir_that_itself_contains_a_threads_subdir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a")])
            shadow = Path(td) / "shadow"
            (shadow / "threads").mkdir(parents=True)
            with self.assertRaises(run_pipeline.OutputDirUnsafeError):
                run_pipeline.assert_out_dir_safe(shadow, corpus)

    def test_accepts_a_plain_sibling_directory(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a")])
            out = Path(td) / "predicted-stub-bare"
            run_pipeline.assert_out_dir_safe(out, corpus)  # must not raise


class TestAssertCacheDirSafe(unittest.TestCase):
    def test_refuses_cache_dir_inside_corpus(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a")])
            with self.assertRaises(run_pipeline.OutputDirUnsafeError):
                run_pipeline.assert_cache_dir_safe(corpus / "cache", corpus)

    def test_refuses_corpus_dir_itself(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a")])
            with self.assertRaises(run_pipeline.OutputDirUnsafeError):
                run_pipeline.assert_cache_dir_safe(corpus, corpus)

    def test_accepts_a_sibling_cache_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            corpus = _make_corpus(Path(td), [("dev-0001", "a")])
            cache = Path(td) / "llm-cache"
            run_pipeline.assert_cache_dir_safe(cache, corpus)  # must not raise

    def test_cli_rejects_cache_dir_inside_corpus_exit_code_two(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
            rc = run_pipeline.main([
                "--corpus", str(corpus), "--client", "stub", "--out", str(tmp_root / "out"),
                "--cache-dir", str(corpus / "cache"),
            ])
            self.assertEqual(rc, 2)


class TestPrepareOutDir(unittest.TestCase):
    def test_refuses_nonempty_output_without_force(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            (out / "provenance.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(run_pipeline.OutputDirUnsafeError):
                run_pipeline.prepare_out_dir(out, "pipeline", force=False)

    def test_force_clears_only_owned_paths_and_keeps_unrelated_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            (out / "provenance.json").write_text("{}", encoding="utf-8")
            (out / "notes.txt").write_text("keep me", encoding="utf-8")
            labels_dir = out / "labels_pipeline"
            labels_dir.mkdir()
            (labels_dir / "dev-0001.json").write_text("{}", encoding="utf-8")

            run_pipeline.prepare_out_dir(out, "pipeline", force=True)

            self.assertFalse((out / "provenance.json").exists())
            self.assertFalse(labels_dir.exists())
            self.assertTrue((out / "notes.txt").exists())

    def test_fresh_directory_needs_no_force(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "brand_new"
            run_pipeline.prepare_out_dir(out, "pipeline", force=False)  # must not raise
            self.assertTrue(out.is_dir())


class TestPrepareOutDirWarnsOnOtherAnnotators(unittest.TestCase):
    def test_warns_on_stderr_and_leaves_other_annotator_dir_untouched(self):
        import io
        import tempfile
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            other_dir = out / "labels_annotator_b"
            other_dir.mkdir()
            (other_dir / "dev-0001.json").write_text("{}", encoding="utf-8")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                run_pipeline.prepare_out_dir(out, "pipeline", force=False)

            self.assertIn("labels_annotator_b", stderr.getvalue())
            self.assertTrue((other_dir / "dev-0001.json").exists())

    def test_force_does_not_touch_other_annotators_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            (out / "provenance.json").write_text("{}", encoding="utf-8")
            other_dir = out / "labels_annotator_b"
            other_dir.mkdir()
            (other_dir / "dev-0001.json").write_text("{}", encoding="utf-8")

            run_pipeline.prepare_out_dir(out, "pipeline", force=True)

            self.assertTrue((other_dir / "dev-0001.json").exists())

    def test_no_warning_when_no_other_annotator_dirs_present(self):
        import io
        import tempfile
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                run_pipeline.prepare_out_dir(out, "pipeline", force=False)
            self.assertEqual(stderr.getvalue(), "")


class TestRunHappyPath(unittest.TestCase):
    def test_stub_client_produces_one_outcome_per_thread_all_gated(self):
        client = RuleBasedStubClient()
        threads = [_thread("dev-0001"), _thread("dev-0002")]
        result = run_pipeline.run(
            client, threads, annotator_id="pipeline", confidence_threshold=0.6,
            max_repair_attempts=1, fail_fast=False,
        )
        self.assertEqual(len(result.outcomes), 2)
        self.assertTrue(all(o.status == "ok" for o in result.outcomes))
        # RuleBasedStubClient hardcodes confidence=0.4 < the 0.6 default
        # threshold, so every stub-produced label is expected to be gated.
        self.assertEqual({gr.thread_id for gr in result.flagged}, {"dev-0001", "dev-0002"})
        for outcome in result.outcomes:
            self.assertEqual(outcome.label.annotator, "pipeline")

    def test_stub_client_labels_are_schema_valid(self):
        client = RuleBasedStubClient()
        threads = [_thread("dev-0001")]
        result = run_pipeline.run(
            client, threads, annotator_id="pipeline", confidence_threshold=0.6,
            max_repair_attempts=1, fail_fast=False,
        )
        self.assertEqual(result.validation_errors["dev-0001"], [])


class TestRunClientExceptionHandling(unittest.TestCase):
    def test_non_parse_exception_becomes_error_outcome(self):
        client = _ScriptedChatClient({"dev-0001": RuntimeError("simulated API timeout")})
        result = run_pipeline.run(
            client, [_thread("dev-0001")], annotator_id="pipeline", confidence_threshold=0.6,
            max_repair_attempts=1, fail_fast=False,
        )
        outcome = result.outcomes[0]
        self.assertEqual(outcome.status, "error")
        self.assertIn("simulated API timeout", outcome.error_detail)
        self.assertIsNone(outcome.label)

    def test_fail_fast_reraises_instead_of_recording_error(self):
        client = _ScriptedChatClient({"dev-0001": RuntimeError("simulated API timeout")})
        with self.assertRaises(RuntimeError):
            run_pipeline.run(
                client, [_thread("dev-0001")], annotator_id="pipeline", confidence_threshold=0.6,
                max_repair_attempts=1, fail_fast=True,
            )

    def test_parse_failed_threads_are_not_gated_and_carry_no_label(self):
        client = _ScriptedChatClient({"dev-0001": [_MALFORMED_JSON, _MALFORMED_JSON]})
        result = run_pipeline.run(
            client, [_thread("dev-0001")], annotator_id="pipeline", confidence_threshold=0.6,
            max_repair_attempts=1, fail_fast=False,
        )
        outcome = result.outcomes[0]
        self.assertEqual(outcome.status, "parse_failed")
        self.assertEqual(result.flagged, [])

    def test_confidence_threshold_plumbed_through_to_gate(self):
        client = _ScriptedChatClient({"dev-0001": [_LOW_CONFIDENCE_JSON]})
        result_default = run_pipeline.run(
            client, [_thread("dev-0001")], annotator_id="pipeline", confidence_threshold=0.6,
            max_repair_attempts=1, fail_fast=False,
        )
        self.assertEqual(len(result_default.flagged), 1)

        client2 = _ScriptedChatClient({"dev-0001": [_LOW_CONFIDENCE_JSON]})
        result_loose = run_pipeline.run(
            client2, [_thread("dev-0001")], annotator_id="pipeline", confidence_threshold=0.05,
            max_repair_attempts=1, fail_fast=False,
        )
        self.assertEqual(len(result_loose.flagged), 0)


class TestWriteRunArtifacts(unittest.TestCase):
    def _run_and_write(self, tmp_root: Path, client, threads, **run_kwargs):
        defaults = dict(annotator_id="pipeline", confidence_threshold=0.6, max_repair_attempts=1, fail_fast=False)
        defaults.update(run_kwargs)
        result = run_pipeline.run(client, threads, **defaults)

        args = argparse.Namespace(
            client=client.name,
            condition="bare",
            corpus=tmp_root / "corpus",
            split="all",
            confidence_threshold=defaults["confidence_threshold"],
            max_repair_attempts=defaults["max_repair_attempts"],
            annotator_id=defaults["annotator_id"],
            cache_dir=None,
            no_cache=True,
        )
        out_dir = tmp_root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        provenance = run_pipeline.build_provenance(args, client, threads, out_dir)
        run_pipeline.write_run(out_dir, provenance, result, save_raw=False)
        return out_dir, result, provenance

    def test_all_six_artifacts_are_written(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            _make_corpus(tmp_root, [("dev-0001", "a")])
            out_dir, _, _ = self._run_and_write(tmp_root, RuleBasedStubClient(), [_thread("dev-0001")])
            self.assertTrue((out_dir / "provenance.json").exists())
            self.assertTrue((out_dir / "predictions.json").exists())
            self.assertTrue((out_dir / "review_queue.json").exists())
            self.assertTrue((out_dir / "summary.md").exists())
            self.assertTrue((out_dir / "labels_pipeline" / "dev-0001.json").exists())
            # raw/ is only populated for parse_failed threads or --save-raw;
            # the ok-status stub outcome above doesn't trigger it.

    def test_provenance_banner_fields(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            _make_corpus(tmp_root, [("dev-0001", "a")])
            client = RuleBasedStubClient()
            out_dir, _, provenance = self._run_and_write(tmp_root, client, [_thread("dev-0001")])
            on_disk = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["client"], "rule_based_stub")
            self.assertEqual(on_disk["thread_ids"], ["dev-0001"])
            self.assertEqual(on_disk["thread_count"], 1)
            self.assertEqual(on_disk["annotator_id"], "pipeline")
            self.assertIn("disclaimer", on_disk)

    def test_written_label_round_trips_and_is_stamped_with_annotator_id(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            _make_corpus(tmp_root, [("dev-0001", "a")])
            out_dir, _, _ = self._run_and_write(
                tmp_root, RuleBasedStubClient(), [_thread("dev-0001")], annotator_id="pipeline",
            )
            label = read_label(out_dir / "labels_pipeline" / "dev-0001.json")
            self.assertEqual(label.thread_id, "dev-0001")
            self.assertEqual(label.annotator, "pipeline")

    def test_custom_annotator_id_used_for_label_dir_and_stamp(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            _make_corpus(tmp_root, [("dev-0001", "a")])
            out_dir, _, _ = self._run_and_write(
                tmp_root, RuleBasedStubClient(), [_thread("dev-0001")], annotator_id="predictions",
            )
            self.assertTrue((out_dir / "labels_predictions" / "dev-0001.json").exists())
            label = read_label(out_dir / "labels_predictions" / "dev-0001.json")
            self.assertEqual(label.annotator, "predictions")

    def test_review_queue_gated_matches_gate_output_exactly(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            _make_corpus(tmp_root, [("dev-0001", "a"), ("dev-0002", "a")])
            client = RuleBasedStubClient()
            out_dir, result, _ = self._run_and_write(tmp_root, client, [_thread("dev-0001"), _thread("dev-0002")])
            review_queue = json.loads((out_dir / "review_queue.json").read_text(encoding="utf-8"))
            gated_ids = {item["thread_id"] for item in review_queue["gated"]}
            self.assertEqual(gated_ids, {gr.thread_id for gr in result.flagged})

    def test_review_queue_buckets_parse_failed_and_errored_separately(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            _make_corpus(tmp_root, [("dev-0001", "a"), ("dev-0002", "a")])
            client = _ScriptedChatClient({
                "dev-0001": [_MALFORMED_JSON, _MALFORMED_JSON],
                "dev-0002": RuntimeError("boom"),
            })
            out_dir, _, _ = self._run_and_write(tmp_root, client, [_thread("dev-0001"), _thread("dev-0002")])
            review_queue = json.loads((out_dir / "review_queue.json").read_text(encoding="utf-8"))
            self.assertEqual([item["thread_id"] for item in review_queue["parse_failed"]], ["dev-0001"])
            self.assertEqual([item["thread_id"] for item in review_queue["errored"]], ["dev-0002"])

    def test_invalid_label_surfaced_in_review_queue(self):
        import tempfile
        # A parseable-but-schema-invalid label: urgency is not one of
        # URGENCY_TIERS, which validate_label() rejects but classify_raw()
        # itself never checks (it only requires valid JSON shape).
        invalid_json = _VALID_JSON.replace('"urgency": "priority"', '"urgency": "whenever"')
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            _make_corpus(tmp_root, [("dev-0001", "a")])
            client = _ScriptedChatClient({"dev-0001": [invalid_json]})
            out_dir, _, _ = self._run_and_write(tmp_root, client, [_thread("dev-0001")])
            review_queue = json.loads((out_dir / "review_queue.json").read_text(encoding="utf-8"))
            self.assertEqual(len(review_queue["invalid_labels"]), 1)
            self.assertEqual(review_queue["invalid_labels"][0]["thread_id"], "dev-0001")

    def test_save_raw_writes_raw_response_for_ok_outcomes_too(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            _make_corpus(tmp_root, [("dev-0001", "a")])
            client = _ScriptedChatClient({"dev-0001": [_VALID_JSON]})
            result = run_pipeline.run(
                client, [_thread("dev-0001")], annotator_id="pipeline", confidence_threshold=0.6,
                max_repair_attempts=1, fail_fast=False,
            )

            args = argparse.Namespace(
                client=client.name, condition="bare", corpus=tmp_root / "corpus", split="all",
                confidence_threshold=0.6, max_repair_attempts=1, annotator_id="pipeline",
                cache_dir=None, no_cache=True,
            )

            out_dir = tmp_root / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            provenance = run_pipeline.build_provenance(args, client, [_thread("dev-0001")], out_dir)
            run_pipeline.write_run(out_dir, provenance, result, save_raw=True)

            self.assertTrue((out_dir / "raw" / "dev-0001.json").exists())

    def test_summary_matches_summarize_outcomes_and_errored_count(self):
        import tempfile
        from classify import summarize_outcomes
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            _make_corpus(tmp_root, [("dev-0001", "a"), ("dev-0002", "a")])
            client = _ScriptedChatClient({
                "dev-0001": [_VALID_JSON],
                "dev-0002": RuntimeError("boom"),
            })
            out_dir, result, _ = self._run_and_write(tmp_root, client, [_thread("dev-0001"), _thread("dev-0002")])
            expected = summarize_outcomes(result.outcomes)
            predictions = json.loads((out_dir / "predictions.json").read_text(encoding="utf-8"))
            self.assertEqual(predictions["summary"]["total"], expected.total)
            self.assertEqual(predictions["summary"]["ok"], expected.ok)
            self.assertEqual(predictions["summary"]["errored"], 1)

    def test_deterministic_ordering_of_outcomes_by_thread_id(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            _make_corpus(tmp_root, [("dev-0003", "a"), ("dev-0001", "a"), ("dev-0002", "a")])
            client = RuleBasedStubClient()
            threads = [_thread("dev-0003"), _thread("dev-0001"), _thread("dev-0002")]
            out_dir, _, _ = self._run_and_write(tmp_root, client, threads)
            predictions = json.loads((out_dir / "predictions.json").read_text(encoding="utf-8"))
            ids = [o["thread_id"] for o in predictions["outcomes"]]
            self.assertEqual(ids, sorted(ids))


class TestBuildProvenanceCacheDir(unittest.TestCase):
    # build_provenance() deliberately reads cache_dir off the client object
    # (getattr(client, "cache_dir", None)), not off args.cache_dir/no_cache,
    # so it can't claim a cache was used when the client never had one —
    # neither branch of that was covered by any existing test.
    def test_none_for_client_with_no_cache_dir_attribute(self):
        client = RuleBasedStubClient()
        self.assertFalse(hasattr(client, "cache_dir"))
        args = argparse.Namespace(
            client=client.name, condition="bare", corpus=Path("corpus"), split="all",
            confidence_threshold=0.6, max_repair_attempts=1, annotator_id="pipeline",
        )
        provenance = run_pipeline.build_provenance(args, client, [_thread("dev-0001")], Path("out"))
        self.assertIsNone(provenance["cache_dir"])

    def test_reports_real_path_for_client_with_cache_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "cache"
            client = _ScriptedChatClient({"dev-0001": [_VALID_JSON]}, cache_dir=cache_dir)
            args = argparse.Namespace(
                client=client.name, condition="bare", corpus=Path("corpus"), split="all",
                confidence_threshold=0.6, max_repair_attempts=1, annotator_id="pipeline",
            )
            provenance = run_pipeline.build_provenance(args, client, [_thread("dev-0001")], Path("out"))
            self.assertEqual(provenance["cache_dir"], str(cache_dir))


class TestCorpusUntouched(unittest.TestCase):
    def test_corpus_tree_bytes_unchanged_after_a_full_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a"), ("dev-0002", "b")])
            before = _sha256_tree(corpus)

            argv = [
                "--corpus", str(corpus), "--client", "stub", "--out", str(tmp_root / "out"),
                "--no-cache",
            ]
            rc = run_pipeline.main(argv)

            after = _sha256_tree(corpus)
            self.assertEqual(rc, 0)
            self.assertEqual(before, after)


class TestMainCLI(unittest.TestCase):
    def test_happy_path_exit_code_zero(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
            rc = run_pipeline.main(["--corpus", str(corpus), "--client", "stub", "--out", str(tmp_root / "out"), "--no-cache"])
            self.assertEqual(rc, 0)

    def test_unsafe_out_dir_exit_code_two(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
            rc = run_pipeline.main(["--corpus", str(corpus), "--client", "stub", "--out", str(corpus), "--no-cache"])
            self.assertEqual(rc, 2)

    def test_empty_selection_exit_code_two(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
            rc = run_pipeline.main([
                "--corpus", str(corpus), "--client", "stub", "--split", "eval",
                "--out", str(tmp_root / "out"), "--no-cache",
            ])
            self.assertEqual(rc, 2)

    def test_nonempty_out_dir_without_force_exit_code_two_then_force_succeeds(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
            out = tmp_root / "out"
            argv = ["--corpus", str(corpus), "--client", "stub", "--out", str(out), "--no-cache"]
            self.assertEqual(run_pipeline.main(argv), 0)
            self.assertEqual(run_pipeline.main(argv), 2)
            self.assertEqual(run_pipeline.main(argv + ["--force"]), 0)

    def test_negative_limit_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
            rc = run_pipeline.main([
                "--corpus", str(corpus), "--client", "stub", "--out", str(tmp_root / "out"),
                "--no-cache", "--limit", "-1",
            ])
            self.assertEqual(rc, 2)
            self.assertFalse((tmp_root / "out").exists())

    def test_negative_max_repair_attempts_rejected(self):
        # A negative value makes classify_thread()'s range(1, n+2) empty, so
        # every thread would silently come back parse_failed without a
        # single client call ever being made — must be rejected up front.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
            rc = run_pipeline.main([
                "--corpus", str(corpus), "--client", "stub", "--out", str(tmp_root / "out"),
                "--no-cache", "--max-repair-attempts", "-1",
            ])
            self.assertEqual(rc, 2)
            self.assertFalse((tmp_root / "out").exists())

    def test_confidence_threshold_out_of_range_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
            rc = run_pipeline.main([
                "--corpus", str(corpus), "--client", "stub", "--out", str(tmp_root / "out"),
                "--no-cache", "--confidence-threshold", "1.5",
            ])
            self.assertEqual(rc, 2)
            self.assertFalse((tmp_root / "out").exists())

    def test_out_dir_setup_oserror_exit_code_two(self):
        # --out pointing at an existing plain file (not a directory) makes
        # prepare_out_dir()'s out_dir.mkdir(exist_ok=True) raise
        # FileExistsError — a real, portable OSError reaching main()'s
        # setup try block, not a mocked one.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
            out = tmp_root / "out"
            out.write_text("not a directory", encoding="utf-8")
            rc = run_pipeline.main([
                "--corpus", str(corpus), "--client", "stub", "--out", str(out), "--no-cache",
            ])
            self.assertEqual(rc, 2)

    def test_write_run_oserror_after_classification_exit_code_two(self):
        # write_run() is called after main()'s setup try block ends, so it
        # needs its own OSError handling — classification (and any real
        # provider spend) has already happened by that point either way.
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
            with mock.patch.object(run_pipeline, "write_run", side_effect=OSError("disk full")):
                rc = run_pipeline.main([
                    "--corpus", str(corpus), "--client", "stub", "--out", str(tmp_root / "out"),
                    "--no-cache",
                ])
            self.assertEqual(rc, 2)

    def test_client_never_touches_real_credentials_for_stub(self):
        # build_client("stub", ...) must return RuleBasedStubClient, never a
        # ChatLLMClient subclass that would read ANTHROPIC_API_KEY/OPENAI_API_KEY
        # or make a network call.
        client = run_pipeline.build_client("stub", condition="bare", cache_dir=None)
        self.assertIsInstance(client, RuleBasedStubClient)
        self.assertNotIsInstance(client, ChatLLMClient)

        # Belt-and-suspenders: patch ChatLLMClient._complete to blow up if a
        # stub run ever reaches it, then run the real CLI end to end and
        # confirm it completes normally without tripping that patch.
        import tempfile
        from unittest import mock
        with mock.patch.object(
            ChatLLMClient, "_complete",
            side_effect=AssertionError("a stub run must never call a chat client"),
        ):
            with tempfile.TemporaryDirectory() as td:
                tmp_root = Path(td)
                corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
                rc = run_pipeline.main([
                    "--corpus", str(corpus), "--client", "stub",
                    "--out", str(tmp_root / "out"), "--no-cache",
                ])
                self.assertEqual(rc, 0)


class TestClientBuiltBeforeForceDeletes(unittest.TestCase):
    def test_force_does_not_delete_existing_output_if_client_construction_fails(self):
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            corpus = _make_corpus(tmp_root, [("dev-0001", "a")])
            out = tmp_root / "out"
            argv = ["--corpus", str(corpus), "--client", "stub", "--out", str(out), "--no-cache"]
            self.assertEqual(run_pipeline.main(argv), 0)
            before = _sha256_tree(out)

            # AnthropicLLMClient.__init__ raises RuntimeError when no API key
            # is set — this must be caught before prepare_out_dir() deletes
            # the good bundle written above, even with --force.
            failing_argv = [
                "--corpus", str(corpus), "--client", "anthropic", "--out", str(out),
                "--no-cache", "--force",
            ]
            with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
                rc = run_pipeline.main(failing_argv)

            self.assertEqual(rc, 2)
            after = _sha256_tree(out)
            self.assertEqual(before, after)


class TestImportBoundary(unittest.TestCase):
    """run_pipeline.py is the one place gating.gate() should be called for
    this pipeline; classify.py deliberately never imports gating (see
    classify.py's own docstring and test_no_agreement_gating_coupling.py)."""

    def _imported_module_names(self, path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names.add(node.module.split(".")[0])
        return names

    def test_run_pipeline_imports_gating(self):
        self.assertIn("gating", self._imported_module_names(_CODE_DIR / "run_pipeline.py"))

    def test_classify_does_not_import_gating(self):
        self.assertNotIn("gating", self._imported_module_names(_CODE_DIR / "classify.py"))


if __name__ == "__main__":
    unittest.main()
