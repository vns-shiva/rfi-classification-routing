"""Stage 2 (classify): LLM backends for the RFI classification/routing task.

Deliberate structural deviation from
../meeting-action-extraction/code/llm_client.py: that paper extracts a
*list* of action items from a chunk of meeting turns, so its ChatLLMClient
needs a two_pass span-identification -> grounded-extraction path and a
chunk/turn grounding guard. An RFI thread classifies to exactly *one*
RFILabel — there is no multi-item extraction and no sub-thread grounding to
check — so this module's ablation dimension is which context block goes
into the prompt, not how many model calls it takes:

  - "bare":   schema + thread only, no injected taxonomy or policy text.
  - "vocab":  + the discipline/CSI/reviewer-role/urgency vocabularies
              (vocabulary.py).
  - "policy": + the routing_policy.py decision table, so routing_rationale
              can cite it.

Three conditions x two providers is this paper's ablation grid (6 runs),
matching the b1/b2/two_pass x 2-provider shape of the prior paper without
inventing an extraction-specific two-pass mechanism this task doesn't have.

As of 2026-09-23 only RuleBasedStubClient has been exercised (see
tests/test_llm_client.py); AnthropicLLMClient and OpenAILLMClient are real,
credentialed implementations, not yet run against real data — that happens
in run_pipeline.py, after generate_rfis.py and the annotation pass exist to
give them something to be scored against.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from routing_policy import route_rfi
from schema import RFILabel, RFIThread
from vocabulary import csi_divisions_for_discipline, DISCIPLINE_TO_CSI


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    """Parse one .env line into (key, value), or None if it's blank/a
    comment/malformed. Strips matching surrounding quotes from the value —
    a `KEY="value"` or `KEY='value'` line (a common .env convention, and how
    a key copy-pasted from a provider console often gets saved) would
    otherwise store the literal quote characters as part of the credential,
    which then fails provider auth with no hint that the key itself is the
    problem."""
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    if not key:
        # A line like "=orphan" has no key at all — os.environ.setdefault("",
        # ...) raises ValueError at module-import time, which would take
        # down every caller of this module over one malformed .env line.
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def _load_dotenv() -> None:
    """Populate os.environ from code/.env (ANTHROPIC_API_KEY,
    ANTHROPIC_WORKSPACE_ID, OPENAI_API_KEY) without overriding any value the
    calling shell already set. Vendored from the prior paper's llm_client.py."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        parsed = _parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


_load_dotenv()

CONDITIONS = ("bare", "vocab", "policy")

_DISCIPLINE_KEYWORDS: dict[str, list[str]] = {
    "Structural": ["beam", "column", "footing", "rebar", "moment connection", "shear tab", "structural"],
    "Mechanical": ["hvac", "ductwork", "air handler", "vav", "chiller", "mechanical"],
    "Electrical": ["panelboard", "conduit", "switchgear", "circuit", "electrical"],
    "Plumbing": ["domestic water", "sanitary", "plumbing fixture", "waste line"],
    "Fire_Protection": ["sprinkler", "fire suppression", "standpipe", "fire pump"],
    "Civil": ["grading", "storm drain", "site utility", "curb", "civil"],
    "Architectural": ["ceiling", "finish", "door hardware", "curtain wall", "millwork", "architectural"],
}

_DEADLINE_PATTERNS = [
    # up to 3 words between "within" and the unit, so "within 10 business
    # days" (not just single-token counts like "within 5 days") still
    # extracts a verbatim deadline_text snippet instead of falling through
    # to no-deadline-found.
    # Negative lookahead excludes "within the past/last N days" — a
    # retrospective reference to something that already happened, not a
    # forward-looking deadline — which the bare "within ... days" pattern
    # would otherwise misread as a stated deadline.
    r"\bwithin (?!the past\b|the last\b)(?:\w+\s+){0,3}(days?|weeks?)\b", r"\bby end of (day|week)\b", r"\btoday\b",
    r"\btomorrow\b", r"\bthis week\b", r"\bnext week\b", r"\basap\b",
    r"\bbefore .*? (pour|install|submittal)\b",
]

_DOC_REF_RE = re.compile(r"\b[A-Z]{1,3}-\d{3,4}(?:\.\d+)?\b")


@dataclass(frozen=True)
class RawResponse:
    """The raw payload behind a classify_raw() call, kept alongside the
    parsed RFILabel so a caller (classify.py's bounded repair retry) can
    inspect exactly what the model returned without re-deriving it from the
    label — a validation failure downstream needs the original text, not the
    already-coerced field values."""
    payload: str
    model: str
    cache_hit: bool


class ClassificationParseError(RuntimeError):
    """Raised by classify_raw() when the payload can't be parsed into an
    RFILabel. Carries the raw payload and a short detail string so a caller
    can build a repair prompt (see prompts.build_repair_instruction) without
    re-parsing this exception's message text."""

    def __init__(self, message: str, payload: str):
        super().__init__(message)
        self.payload = payload
        self.detail = message


class LLMClient(ABC):
    name: str = "abstract"
    model: str = ""

    @abstractmethod
    def classify_raw(self, thread: RFIThread, repair_suffix: str = "") -> tuple[RFILabel, RawResponse]:
        """repair_suffix: appended to the user prompt, if non-empty — used by
        classify.py's bounded repair retry (see prompts.build_repair_instruction)
        to change the (system, user) cache key on a retry after
        ClassificationParseError. Ignored by any client whose classify_raw()
        never raises that error (e.g. RuleBasedStubClient), since such a
        client never gets a repair retry."""
        ...

    def classify(self, thread: RFIThread) -> RFILabel:
        return self.classify_raw(thread)[0]


class RuleBasedStubClient(LLMClient):
    """Deterministic, non-LLM test double — a keyword matcher over the
    thread text. Its accuracy against gold labels is expected to be
    mediocre; that is the honest floor an LLM backend should be measured
    against, not a claim of classification quality. It exists so
    classify.py/run_pipeline.py has a zero-cost, zero-dependency
    end-to-end path to smoke-test before any real API budget is spent."""

    name = "rule_based_stub"
    model = "rule_based_stub"

    def __init__(self, condition: str = "bare", **kwargs):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition {condition!r} (expected one of {CONDITIONS})")
        self.condition = condition

    def classify_raw(self, thread: RFIThread, repair_suffix: str = "") -> tuple[RFILabel, RawResponse]:
        # repair_suffix is part of LLMClient's shared interface but unused
        # here: this stub never raises ClassificationParseError, so
        # classify.py's repair retry never targets it.
        full_text = " ".join(m.text for m in thread.messages)
        text_l = full_text.lower()

        # Checked most-specific-first: the generic "conflict" keyword inside
        # document_discrepancy's pattern would otherwise swallow the field-
        # condition and coordination cases before they ever get a chance to
        # match (both routinely say "conflict" too, e.g. "coordination
        # conflict between ductwork and beam"). Each alternation is
        # parenthesized so \b binds to the whole group, not just the first
        # and last alternatives.
        rfi_type = "design_clarification"
        if re.search(r"\bsubstitut", text_l):
            rfi_type = "substitution_request"
        elif re.search(r"\b(?:field condition|as-built|existing condition)\b", text_l):
            rfi_type = "field_condition_conflict"
        elif re.search(r"\bcoordinat", text_l):
            rfi_type = "coordination_conflict"
        elif re.search(r"\b(?:conflict|discrepanc|doesn't match|does not match)", text_l):
            rfi_type = "document_discrepancy"
        elif re.search(r"\b(?:code|ibc|nfpa|compliance)\b", text_l):
            rfi_type = "code_compliance_question"

        primary_discipline = "General"
        for disc, keywords in _DISCIPLINE_KEYWORDS.items():
            if any(k in text_l for k in keywords):
                primary_discipline = disc
                break

        urgency = "routine"
        if re.search(r"\bcritical\b|\bimmediat|\bstop work\b|\bsafety\b", text_l):
            urgency = "critical"
        elif re.search(r"\burgent\b", text_l):
            urgency = "urgent"
        elif re.search(r"\bpriority\b|\bexpedite\b", text_l):
            urgency = "priority"

        csi_division = "—"
        if primary_discipline != "General":
            candidates = sorted(csi_divisions_for_discipline(primary_discipline))
            csi_division = candidates[0] if candidates else "—"

        deadline_text = "none stated"
        for pat in _DEADLINE_PATTERNS:
            m = re.search(pat, text_l)
            if m:
                deadline_text = m.group(0)
                break

        cost_impact = bool(re.search(r"\bcost\b|\bchange order\b|\badditional cost\b", text_l))
        schedule_impact = bool(re.search(r"\bschedule\b|\bdelay\b|\bcritical path\b", text_l))

        # Routing is delegated to routing_policy.route_rfi() rather than a
        # second, independently-maintained reviewer table here: two tables
        # inevitably drift (the previous table sent "General" to "GC
        # Superintendent" while routing_policy.DISCIPLINE_REVIEWER sends it
        # to "Architect of Record"), and this stub's escalation flag
        # previously ignored the cost_impact-and-schedule_impact clause that
        # routing_policy.py and prompts.build_policy_block() both specify.
        routing = route_rfi(
            rfi_type=rfi_type,
            primary_discipline=primary_discipline,
            cost_impact=cost_impact,
            schedule_impact=schedule_impact,
            urgency=urgency,
        )

        label = RFILabel(
            thread_id=thread.thread_id,
            rfi_type=rfi_type,
            primary_discipline=primary_discipline,
            secondary_disciplines=[],
            csi_division=csi_division,
            urgency=urgency,
            question_summary=(thread.messages[0].text.strip()[:240] if thread.messages else ""),
            referenced_documents=sorted(set(_DOC_REF_RE.findall(full_text))),
            proposed_solution="—",
            cost_impact=cost_impact,
            schedule_impact=schedule_impact,
            answer_in_documents=False,
            deadline_text=deadline_text,
            assigned_reviewer=routing.assigned_reviewer,
            routing_rationale=routing.routing_rationale,
            escalation=routing.escalation,
            confidence=0.4,
            annotator="",
        )
        return label, RawResponse(payload="", model=self.model, cache_hit=False)


class ChatLLMClient(LLMClient):
    """Shared orchestration for any chat-completion-style provider. A
    subclass implements only `_complete(system, user) -> str` for one model
    call (and `.name`); this class handles prompt assembly, JSON parsing,
    and building the RFILabel from the parsed object.
    """

    def __init__(self, condition: str = "bare", cache_dir: Path | None = None):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition {condition!r} (expected one of {CONDITIONS})")
        self.condition = condition
        self.cache_dir = cache_dir
        self.stats = {"calls": 0, "cache_hits": 0}

    @abstractmethod
    def _complete(self, system: str, user: str) -> str:
        """One provider call. Returns the raw text payload, or raises
        RuntimeError if the response was truncated before any text was
        emitted."""
        ...

    def _cache_key_prefix(self) -> str:
        """`self.name` alone (e.g. "anthropic") collides across every model
        that provider offers — two clients differing only in `.model` would
        silently read each other's cached responses."""
        return f"{self.name}\x1f{self.model}"

    def _cached_complete(self, system: str, user: str) -> tuple[str, bool]:
        """Returns (payload, cache_hit). The hit flag lets classify_raw()
        report provenance (RawResponse.cache_hit) without a second lookup."""
        if self.cache_dir is None:
            self.stats["calls"] += 1
            return self._complete(system, user), False

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(f"{self._cache_key_prefix()}\x1f{system}\x1f{user}".encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{key}.txt"
        if cache_path.exists():
            cached = cache_path.read_text(encoding="utf-8")
            if cached:
                self.stats["cache_hits"] += 1
                return cached, True
            # An empty payload means a prior call failed after writing but
            # before raising downstream (or a truncated response slipped
            # through) — never trust an empty cache entry, always retry.

        self.stats["calls"] += 1
        result = self._complete(system, user)
        if result:
            cache_path.write_text(result, encoding="utf-8")
        return result, False

    @staticmethod
    def _strip_fences(payload: str) -> str:
        payload = payload.strip()
        if not payload.startswith("```"):
            return payload
        if "\n" in payload:
            # A multi-line fenced block: drop the whole opening-fence line
            # (whatever it contains — a language tag, a tag plus trailing
            # text like "```json output", or just "```") rather than trying
            # to regex away the tag in place, then truncate at the last
            # remaining "```" so trailing prose after the closing fence
            # doesn't leak into the JSON payload either.
            payload = payload.split("\n", 1)[1]
            last_fence = payload.rfind("```")
            if last_fence != -1:
                payload = payload[:last_fence]
            return payload.strip()
        # Collapsed onto a single line ("```json {...}```") — no newline
        # separates the language tag from the JSON body, so the line-based
        # approach above doesn't apply; strip the tag/fences in place.
        payload = re.sub(r"^```[A-Za-z0-9_+-]*\s*", "", payload)
        payload = re.sub(r"```\s*$", "", payload).strip()
        return payload

    def _parse_object(self, payload: str) -> dict:
        payload = self._strip_fences(payload)
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Could not parse {self.name} JSON output: {payload[:500]!r}") from e
        if not isinstance(obj, dict):
            raise RuntimeError(f"Expected a JSON object from {self.name}, got: {payload[:200]!r}")
        return obj

    @staticmethod
    def _as_list(value) -> list[str]:
        """Models sometimes return a single string instead of a one-element
        list for a list-typed field (secondary_disciplines,
        referenced_documents) — normalize rather than erroring downstream in
        validate_label with a confusing type mismatch."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        return [str(value)]

    @staticmethod
    def _get(obj: dict, key: str, default):
        """obj.get(key, default) does not apply `default` when the JSON key
        is present but explicitly `null` — a model returning
        {"urgency": null} would otherwise store None where a str/list/float
        was promised, crashing later (.strip(), unhashable list, float(None))
        with no error at parse time. Treat null the same as absent."""
        value = obj.get(key, default)
        return default if value is None else value

    @staticmethod
    def _as_str(value, default: str = "") -> str:
        """A scalar field can come back as a list (or other non-str JSON
        value) when the model over-generates; coerce rather than storing a
        wrong-shaped value that validate_label can't catch."""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return ", ".join(str(v) for v in value) if value else default
        return str(value)

    @staticmethod
    def _as_bool(value, default: bool = False) -> bool:
        """bool("false") is True in Python — a model returning the JSON
        string "false" for cost_impact/schedule_impact/escalation would
        silently invert to True under a bare bool() cast, and validate_label
        has no way to catch a wrong-but-well-typed boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "no", "0", "")
        return bool(value)

    @staticmethod
    def _as_float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_csi_division(value: str, primary_discipline: str) -> str:
        """The vocabulary block (prompts.build_vocabulary_block) hands the
        model a csi_division vocabulary where each option's canonical string
        is "<code> (<title>)", e.g. "23 00 00 (HVAC)" — the parenthesized
        title is part of the enum value, not decoration. Models routinely
        echo back only the numeric code ("23 00 00"), which is otherwise a
        spurious validate_label() failure despite being an unambiguous
        answer: within one primary_discipline's candidate set, no two CSI
        divisions share a code, so a bare code that is a prefix of exactly
        one candidate snaps to that candidate's full canonical string.
        Anything else (a fabricated code, or an already-correct string) is
        returned unchanged and left for validate_label() to accept or
        reject."""
        candidates = DISCIPLINE_TO_CSI.get(primary_discipline, set())
        if value in candidates:
            return value
        matches = [c for c in candidates if c.startswith(value)]
        if len(matches) == 1:
            return matches[0]
        return value

    def _to_label(self, thread_id: str, obj: dict) -> RFILabel:
        get = self._get
        primary_discipline = self._as_str(get(obj, "primary_discipline", ""))
        csi_division = self._normalize_csi_division(
            self._as_str(get(obj, "csi_division", "—"), "—").strip(), primary_discipline
        )
        return RFILabel(
            thread_id=thread_id,
            rfi_type=self._as_str(get(obj, "rfi_type", "")),
            primary_discipline=primary_discipline,
            secondary_disciplines=self._as_list(get(obj, "secondary_disciplines", None)),
            csi_division=csi_division,
            urgency=self._as_str(get(obj, "urgency", "")),
            question_summary=self._as_str(get(obj, "question_summary", "")),
            referenced_documents=self._as_list(get(obj, "referenced_documents", None)),
            proposed_solution=self._as_str(get(obj, "proposed_solution", "—"), "—"),
            cost_impact=self._as_bool(get(obj, "cost_impact", False)),
            schedule_impact=self._as_bool(get(obj, "schedule_impact", False)),
            answer_in_documents=self._as_bool(get(obj, "answer_in_documents", False)),
            deadline_text=self._as_str(get(obj, "deadline_text", "none stated"), "none stated"),
            assigned_reviewer=self._as_str(get(obj, "assigned_reviewer", "")),
            routing_rationale=self._as_str(get(obj, "routing_rationale", "")),
            escalation=self._as_bool(get(obj, "escalation", False)),
            confidence=self._as_float(get(obj, "confidence", 0.5), 0.5),
            annotator="",
        )

    def classify_raw(self, thread: RFIThread, repair_suffix: str = "") -> tuple[RFILabel, RawResponse]:
        from prompts import build_classification_prompt, build_classification_system_prompt

        system = build_classification_system_prompt(self.condition)
        user = build_classification_prompt(thread) + repair_suffix
        payload, cache_hit = self._cached_complete(system, user)
        try:
            obj = self._parse_object(payload)
            label = self._to_label(thread.thread_id, obj)
        except RuntimeError as e:
            raise ClassificationParseError(str(e), payload) from e
        return label, RawResponse(payload=payload, model=self.model, cache_hit=cache_hit)


class AnthropicLLMClient(ChatLLMClient):
    """Real API-backed classifier: imports the real `anthropic` package
    lazily and issues a real request when actually called."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None,
                 workspace_id: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.workspace_id = workspace_id or os.environ.get("ANTHROPIC_WORKSPACE_ID")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. AnthropicLLMClient is real but "
                "cannot run without credentials — use RuleBasedStubClient for "
                "a dependency-free pipeline run, or set the key to exercise this."
            )

    def _client(self):
        import anthropic
        headers = {"anthropic-workspace-id": self.workspace_id} if self.workspace_id else None
        return anthropic.Anthropic(api_key=self.api_key, default_headers=headers)

    def _complete(self, system: str, user: str) -> str:
        client = self._client()
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        payload = "".join(text_blocks).strip()
        if not payload and response.stop_reason == "max_tokens":
            raise RuntimeError("Anthropic response truncated at max_tokens before any text was emitted.")
        return payload


class OpenAILLMClient(ChatLLMClient):
    """Second real API-backed classifier (>=2-provider requirement). Same
    prompts/schema as AnthropicLLMClient so the two are a genuine
    apples-to-apples comparison of provider, not of prompt."""

    name = "openai"

    def __init__(self, model: str = "gpt-5", api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. OpenAILLMClient is real but "
                "cannot run without credentials — use RuleBasedStubClient for "
                "a dependency-free pipeline run, or set the key to exercise this."
            )

    def _client(self):
        import openai
        return openai.OpenAI(api_key=self.api_key)

    def _complete(self, system: str, user: str) -> str:
        client = self._client()
        response = client.chat.completions.create(
            model=self.model,
            max_completion_tokens=4096,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = response.choices[0]
        payload = (choice.message.content or "").strip()
        if not payload and choice.finish_reason == "length":
            raise RuntimeError("OpenAI response truncated at max_completion_tokens before any text was emitted.")
        return payload


def build_client(name: str, condition: str = "bare", cache_dir: Path | None = None) -> LLMClient:
    if name == "stub":
        return RuleBasedStubClient(condition=condition)
    if name == "anthropic":
        return AnthropicLLMClient(condition=condition, cache_dir=cache_dir)
    if name == "openai":
        return OpenAILLMClient(condition=condition, cache_dir=cache_dir)
    raise ValueError(f"Unknown client {name!r} (expected 'stub', 'anthropic', or 'openai')")
