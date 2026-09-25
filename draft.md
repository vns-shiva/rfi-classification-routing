# LLM-Based Automated RFI Classification and Routing for Construction Projects

## Status (internal — not part of the submission)

Draft in progress. All experimental numbers below are real, taken directly
from `pilot-data/routing_eval-*/routing_eval_report.json` and
`pilot-data/annotator_agreement/agreement_report.json` (see
`code/make_figures.py`, which renders the three figures under `figures/`
straight from those same JSON files — no hand-typed numbers in a figure).
Related Work citations are real, verified `elsarticle-num` references
([1]-[30], ordered by first citation — see the References section and
"Open items" at the bottom of this file).

Target venue: *Automation in Construction* (Elsevier), backup *Journal of
Computing in Civil Engineering*. Format: elsarticle-num (numbered,
non-superscript, ordered by first citation), double-spaced body with
continuous line numbers, separate Highlights file (3–5 bullets, ≤85
characters each). See `submission-format-requirements.docx`.

---

## Highlights (draft — final copy goes in a separate Highlights file, not this manuscript)

Each bullet is character-counted against the venue's ≤85-character limit
(`len()` checked directly, not eyeballed):

- Closed vocabularies are required: without them, all LLM outputs are schema-invalid. (83)
- LLM fields plus a routing policy reach 73.5% vs. an 85.5% oracle-ceiling accuracy. (82)
- Adding routing-policy text does not beat closed vocabularies alone for routing. (79)
- Confidence/escalation gating splits a small hard subset from a reliable majority. (81)
- A disclosed LLM-proxy annotator agrees most on discipline, least on escalation. (79)

---

## Abstract (draft)

Requests for Information (RFIs) are a costly bottleneck in construction
project delivery: each RFI must be classified by discipline and urgency,
checked against a "ball-in-court" contractual routing rule, and assigned to
the correct reviewer before it can be answered. We study whether a large
language model (LLM) can perform this classification-and-routing step
reliably and, more specifically, *what part of the prompt actually earns
that reliability*. We construct a stratified, synthetic corpus of 240 RFI
threads spanning six RFI types, eight disciplines, four urgency tiers, and
seven adversarial stress conditions, generated under two anti-fabrication
contracts guaranteeing every extracted deadline phrase and referenced
document is a verbatim substring of its source thread. Against this corpus
we run a three-condition ablation — schema-only ("bare"), schema plus closed
vocabularies ("vocab"), and vocab plus a plain-language routing policy
("policy") — with a credentialed LLM (Anthropic Claude Sonnet 5). The bare
condition fails outright: 100% of 240 outputs violate a closed-vocabulary or
cross-field validation rule and are excluded from scoring. Under vocab,
composing the LLM's classification fields with a deterministic routing
policy reaches 73.5% reviewer-assignment accuracy against gold labels
(Cohen's κ=0.700) versus an 85.5% oracle ceiling (κ=0.834) and a 7.5%
majority baseline; adding the policy text itself does not improve on vocab's
accuracy. Confidence/escalation gating separates a small (18%), genuinely
hard subpopulation (≈39% accuracy) from a large majority routed at over 81%
accuracy. A disclosed LLM proxy for a second annotator, used because no
human coder was available and reported as a limitation, not a claim of
human-human reliability, agrees with gold at Krippendorff's α=0.914
(discipline), 0.818 (urgency), and 0.484 (escalation). A full replication
with a second provider (OpenAI GPT-5) across all three conditions confirms
both headline findings: the bare condition again fails outright (239/240
outputs schema-invalid, the 240th failing before a parseable response), and
composed reviewer-assignment accuracy is materially unchanged by the
policy-text addition (74.0% vocab vs. 74.0% policy, vs. Anthropic's 73.5%
vocab vs. 74.0% policy), with both providers' vocab/policy accuracy landing
within 1.5 points of each other on every scoring arm.

**Keywords:** requests for information; construction document classification; large language models; prompt engineering; structured output validation; inter-annotator agreement; ticket routing

---

## 1. INTRODUCTION

Requests for Information (RFIs) are the formal mechanism by which a
contractor, subcontractor, or trade asks the design team to clarify an
ambiguity, resolve a conflict between contract documents, or approve a
substitution before proceeding with the work. Two operational facts make
RFIs a good target for automation. First, contracts already specify *who*
must answer an RFI as a function of its subject matter, a "ball-in-court"
rule (e.g., AIA A201-2017 §3.2.4; ConsensusDocs 200 §12.2) that assigns
interpretive authority to the architect or the engineer of record for the
relevant discipline, and pulls in the owner's representative once a cost or
schedule impact is implicated. That means correct routing is, in principle,
a deterministic function of a small number of classification fields
(discipline, urgency, cost/schedule impact, escalation), *if* those fields
can be extracted reliably. Second, RFI volume and turnaround time are large
enough on a typical project that misrouting is expensive in aggregate: a
single RFI sent to the wrong reviewer, or held for a level of urgency it
does not warrant, adds days of avoidable float to schedule-critical paths.

Large language models are an obvious candidate for the classification step:
an RFI thread is unstructured natural-language text, and LLMs are
increasingly used across construction and AEC workflows for exactly this
kind of document understanding, including a direct precedent for
LLM-assisted RFI handling itself [1] and broader reviews of LLM applications
across AEC tasks [2,3]. But an LLM's raw output is not, by default, a value
from a closed taxonomy. Asked to name a discipline or a CSI division
without being told what the valid set is, a model will produce a
plausible-sounding but potentially out-of-vocabulary string, and a routing
system built on top of that output inherits whatever downstream failures an
invalid or inconsistent field produces (a reviewer role that does not exist
in the firm's directory; a CSI division that contradicts the stated
discipline). This is not a hypothetical risk unique to this task: prior work
on constrained decoding, schema-guided generation, and function calling has
found that unconstrained LLM output frequently fails structural or semantic
validation even when the underlying content is substantively correct
[4–6].

This paper asks a narrower, more actionable question than "can an LLM
classify RFIs": *which part of the prompt is responsible for whatever
reliability is achieved?* We isolate three nested conditions: a bare schema
description; the schema plus closed vocabularies for every controlled field;
and the vocabularies plus a plain-language rendering of the deterministic
routing policy itself. We hold the model, the corpus, and every other
prompt element fixed across them. This ablation design lets us attribute
accuracy gains, or the lack of them, to a specific, reusable piece of prompt
engineering, not to an unspecified combination of "a good prompt."

Because no public, labeled RFI corpus exists at the scale and stratification
this ablation needs, and because real project RFIs are commercially
sensitive, we construct a synthetic corpus under two anti-fabrication
contracts enforced at generation time instead of by post-hoc filtering:
every deadline phrase and every referenced-document citation a gold label
claims is a verbatim substring of the rendered thread it comes from. This
lets us report extraction accuracy without the confound of the corpus itself
containing invented "ground truth" that never appeared in the text a
classifier actually reads.

Our contributions are: (1) a stratified, adversarial, anti-fabrication RFI
corpus generator and the resulting 240-thread pilot corpus; (2) a
three-condition prompt ablation isolating the marginal contribution of
closed vocabularies and of policy-table injection to schema-valid,
accurate RFI classification; (3) a five-arm routing evaluation
(oracle/composed/direct/policy-fidelity/majority) that separates
classification-field noise from policy/gold divergence instead of
reporting a single end-to-end accuracy number that conflates the two; (4) a
disclosed, LLM-proxy inter-annotator-agreement analysis, reported as a
limitation-bounded proxy for human-human reliability, not a claim of
one; and (5) a full cross-provider replication with a second commercial LLM
(OpenAI GPT-5) across all three conditions, confirming both headline
findings within 1.5 accuracy points on every scoring arm instead of a
replication claim resting on a single provider.

---

## 2. RELATED WORK

### 2.1 RFI management and construction communication analysis

RFIs have been studied directly as a source of schedule and cost risk.
Hanna et al. [7] establish benchmark RFI volumes and turnaround times for
major highway projects, and Mohamed et al. [8] quantify the direct time and
cost the RFI process itself adds to a project. Aibinu et al. [9] extend this
to a data-analytic study of RFI frequency and turnaround time across a
larger project sample, and Joy et al. [10] motivate digitizing and
automating the RFI process specifically, framing it, as this paper does,
as a workflow bottleneck instead of a purely contractual formality. The one
existing attempt to apply a large language model to the RFI process itself,
Panahi et al.'s [1] ChatGPT-based RFI recommender for pre-construction
design review, is the closest direct precedent to this paper's task; it
targets recommendation, not the closed-vocabulary schema validity and
policy-grounded routing this paper isolates, which is the gap this paper's
ablation is designed to fill.

### 2.2 NLP and text mining applied to construction documents

A separate line of work applies text classification and information
extraction to other construction document types — specifications,
submittals, and contracts — without targeting RFIs specifically. Xu et
al. [11] survey text-mining applications across the construction industry
and identify a persistent gap between narrow proof-of-concept classifiers
and deployable pipelines with explicit validity guarantees, the gap this
paper's closed-vocabulary ablation addresses for one task. Moon et al. [12]
use a BERT-based classifier to detect contractual risk clauses in
specifications, Pham and Han [13] apply multitask classification to predict
risk-handling actions in construction contracts, and Dikmen et al. [14]
combine NLP and machine learning for automated contract risk and
responsibility assessment. None of these apply a closed-vocabulary or
policy-conditioning ablation to isolate which part of a prompt is
responsible for a model's output validity, which is this paper's specific
contribution relative to this cluster.

### 2.3 Large language models applied to AEC tasks

Beyond text classification, LLMs are increasingly applied to broader AEC
reasoning tasks. Zheng et al. [15] use a knowledge-graph-enhanced LLM to
automate construction contract review, Erfani and Khanjar [16] compare
several LLMs on construction risk classification, and Emara [17] applies an
ontology-driven GPT-4 pipeline to question-answering over construction
standards. Two recent reviews, Kampelopoulos et al. [2] and Gao et al. [3],
survey LLM applications and implementation strategies across the AEC
industry more broadly, and both identify structured-output reliability as an
open challenge for deployment; this paper treats that challenge as its
central empirical question, not merely a stated limitation.

### 2.4 Structured output, schema-constrained generation, and function calling

This paper's central bare/vocab/policy finding, that supplying a closed
vocabulary (not elaborating the prompt with policy prose) is what makes an
LLM's output schema-valid, sits inside a broader empirical literature on
LLM output structure. Tam et al. [4] show that format restrictions imposed
on an LLM's output (JSON, XML, or a custom schema) can measurably degrade
reasoning performance relative to free-form generation, a tension this
paper's ablation design controls for by holding the output schema fixed
across all three conditions and varying only the vocabulary/policy content
supplied alongside it. Geng et al. [5] introduce JSONSchemaBench, a
benchmark quantifying how reliably current LLMs produce schema-conforming
JSON across schema complexity levels, and Patil et al. [6] introduce Gorilla,
demonstrating that LLMs connected to a large API surface without an explicit
enumerated interface reliably hallucinate arguments and function names
outside the valid set. This is the same class of out-of-vocabulary failure
this paper documents for `rfi_type`, `csi_division`, and `assigned_reviewer`
under the bare condition (§5.1).

### 2.5 LLM-as-classifier and LLM-as-router for ticket and task assignment

Outside construction, LLM-based routing of tickets or tasks to a specialist
handler is an active area with a similar structure to this paper's routing
step. Ong et al.'s RouteLLM [18] learns to route a query to a cost-effective
vs. a high-capability LLM based on predicted response quality — a routing
decision over *models*, whereas this paper routes a classified RFI to a
*human reviewer*, but the underlying question (can a fixed, auditable
routing rule be driven reliably by LLM-produced classification fields) is
the same. Zhou and Li [19] apply few-shot LLM classification to triage
patient inquiries into actionable categories, and Madeyski [20] routes
software engineering tasks to cost-tiered LLMs using code-quality signals;
both are ticket/task routing analogues this paper's five-arm routing
evaluation (§4.1) generalizes by separating classification-field noise from
routing-rule divergence, which neither prior study isolates explicitly.

### 2.6 Inter-annotator agreement and LLM-as-annotator

This paper's agreement methodology (§4.3) follows the standard
chance-corrected agreement literature: Cohen's weighted kappa [21] for
ordinal fields and Passonneau's MASI [22] for set-valued fields such as
`secondary_disciplines` and `referenced_documents`. James [23] provides
recent guidance on selecting an appropriate inter-annotator agreement metric
for NLP annotation tasks specifically, which this paper follows in reporting
field-kind-appropriate metrics (Krippendorff's alpha, MASI, cosine
similarity) instead of a single blanket statistic. A separate and more
directly relevant literature uses an LLM itself as an annotator: Gilardi et
al. [24] report that ChatGPT outperforms crowd workers on several
text-annotation tasks; Gu et al. [25], by contrast, find that LLMs are
useful annotation *assistants* but not reliable independent annotators when
compared against trained human coders; and Bojic et al. [26] compare LLMs
against human annotators specifically on latent, subjective content
dimensions (sentiment, political leaning, sarcasm) and find systematic
divergences on the more subjective fields. This paper's disclosed LLM-proxy
second annotator (§3.5) sits deliberately inside this tension: §5.4's
field-level agreement pattern (near-perfect on structured boolean fields
such as `cost_impact`, α=1.000; high on `primary_discipline` and `urgency`;
but markedly lower on the more judgment-dependent `escalation` field,
α=0.484) is consistent with Gu et al.'s finding that LLM annotation
reliability is uneven across field types, not a uniform substitute for a
human coder. This is why this paper reports the LLM proxy as a disclosed
limitation (§6.2) instead of a claim of human-equivalent reliability.

### 2.7 LLM evaluation methodology and prompt sensitivity

This paper's ablation design holds the corpus, model, and every prompt
element fixed except the vocabulary/policy content under test, following
established practice for isolating a specific prompt component's causal
contribution instead of attributing an accuracy change to an unspecified
"better prompt." Zhu et al.'s PromptBench [27] shows that LLM performance
can vary substantially under adversarial or superficial prompt
perturbations that do not change task semantics, motivating this paper's
choice to vary only semantically meaningful prompt content (a taxonomy, a
policy) across conditions and hold surface wording fixed. Hua et al. [28]
further argue that apparent prompt sensitivity in LLM evaluations is often
an artifact of evaluation design, not a genuine model instability,
reinforcing this paper's use of a fixed, schema-validated scoring pipeline
(`validate_label()`, §3.1) so that an accuracy difference between conditions
reflects a difference in the prompt content itself and not measurement
noise.

### 2.8 Construction industry AI and digitization surveys (motivation)

Two broader surveys motivate this paper's premise that AI-based automation
is a live and expanding concern across construction project delivery.
Egwim et al. [29] systematically review AI applications across the entire
construction value-chain lifecycle, and Gill et al. [30] review recent
advances in applying AI approaches to construction industry problems more
broadly. Neither addresses RFI classification or ball-in-court routing
specifically, which is the gap this paper's contribution (§1) targets within
that broader adoption trend.

---

## 3. METHOD

### 3.1 Task and Label Schema

An RFI thread (`schema.RFIThread`) is a sequence of timestamped messages
(`initial_question`, `response`, `clarification`, `resubmittal`), each
carrying a sender role and free text, rendered as a single document for
classification. A label (`schema.RFILabel`) has three blocks:

- **Classification** — `rfi_type` (one of six values: `design_clarification`,
  `document_discrepancy`, `field_condition_conflict`,
  `substitution_request`, `coordination_conflict`,
  `code_compliance_question`), `primary_discipline` and
  `secondary_disciplines` (from an eight-value discipline taxonomy:
  `Architectural`, `Structural`, `Mechanical`, `Electrical`, `Plumbing`,
  `Fire_Protection`, `Civil`, `General`), `csi_division` (a CSI MasterFormat
  2016 group-level division consistent with the primary discipline, or
  `"—"` only when the discipline is `General`), and `urgency` (an *ordinal*
  four-tier scale, `routine < priority < urgent < critical`).
- **Extraction** — `question_summary`, `referenced_documents` (drawing/spec
  numbers actually cited in the thread), `proposed_solution` (`"—"` if none
  was proposed), `cost_impact`/`schedule_impact`/`answer_in_documents`
  (booleans), and `deadline_text` (a verbatim phrase from the thread, or
  `"none stated"`).
- **Routing** — `assigned_reviewer` (one of a fixed ten-role taxonomy:
  Architect of Record, Structural/Mechanical/Electrical/Civil Engineer,
  Plumbing/Fire Protection Engineer, GC Superintendent, Owner's
  Representative, Cost/Change-Order Manager, Code Official/AHJ Liaison),
  `routing_rationale`, and a boolean `escalation` flag.

`validate_label()` enforces every closed-vocabulary membership and two
cross-field consistency rules: `csi_division` must belong to the CSI
divisions mapped to `primary_discipline` (a many-to-many mapping — several
divisions, such as fire suppression, are legitimately shared by two
disciplines), and `csi_division == "—"` is valid only when
`primary_discipline == "General"`.

### 3.2 Corpus Construction

The corpus generator (`stratification.py` → `generate_rfis.py`) builds a
240-thread plan (40 DEV / 200 EVAL, split disjointly and used disjointly
throughout: DEV never contributes to the ablation's headline accuracy
numbers, only to the majority baseline's fit and to inter-annotator
agreement) with quota-guaranteed coverage of every axis value, not just a
weighted-random draw, since weights alone cannot guarantee a
rare-but-important combination (e.g. `urgency=critical`) appears at all.
Seven stress conditions are included as an explicit adversarial axis:
`answer_in_documents` (the cited document already contains the answer),
`boundary_urgency`, `distractor_cost_language`, `multi_discipline`,
`type_ambiguity`, `unresolvable_deadline`, and a `none` (non-adversarial)
category.

Two anti-fabrication contracts are enforced at generation time, not by
post-hoc filtering: (1) `deadline_text` is only populated when the chosen
template body variant literally contains the deadline phrase as a
substring of the rendered thread; otherwise it is honestly `"none
stated"`, never an invented span; (2) `referenced_documents` only
lists a document whose rendered name actually appears in the thread text.
Both properties are verified programmatically for every generated thread,
not asserted from the templates alone.

Gold labels come from each template's own authored `gold_routing_table`
(`templates.gold_row_for_cell()`), never from `routing_policy.py` itself;
`policy_audit.py` separately measures how often the deterministic policy
agrees with gold (the routing-evaluation "oracle" arm's ceiling below is
this same measurement, restricted to threads with a matched prediction),
so that agreement is a *result*, not a definition.

### 3.3 Routing Policy

`routing_policy.py` operationalizes the "ball-in-court" contractual
practice standard (AIA A201-2017 §3.2.4; ConsensusDocs 200 §12.2) as a
deterministic function from classification fields to `(reviewer,
rationale, escalation)`. It is not used to produce gold labels (§3.2); it
supplies the **composed** and **oracle** routing-evaluation arms (§4.1) and
is optionally rendered into the "policy" prompt condition (§3.4) as
plain-language rules, never as source code, so the model can cite the same
rule the scoring arms use without the system prompt teaching it to game the
scoring.

### 3.4 Prompt Ablation and LLM Clients

Three nested prompt conditions, one system prompt each (`prompts.py`):

- **bare** — the label schema only; no injected taxonomy or policy text.
- **vocab** — bare + the closed vocabularies (§3.1's `RFI_TYPES`,
  `DISCIPLINES`, the discipline→CSI map, `URGENCY_TIERS`,
  `REVIEWER_ROLES`).
- **policy** — vocab + a plain-language rendering of `routing_policy.py`'s
  decision table.

Two credentialed commercial LLM providers are run behind an identical
client interface (`llm_client.build_client`): Anthropic (`claude-sonnet-5`)
and OpenAI (`gpt-5`), plus a deterministic `RuleBasedStubClient` used only
for unit testing, never for reported results. `classify.classify_thread()`
retries once, with a repair instruction appended to the user prompt (never
the system prompt, so the cache key changes on retry), only on a malformed
(unparseable) response; a response that parses but fails
`validate_label()`'s semantic checks is *not* retried by this layer; it is
recorded as an invalid prediction and excluded from scoring (§4.2), which
is itself the outcome the bare condition is designed to surface (§5.1).
`run_pipeline.py` classifies every thread in a split, then runs confidence
gating (`gating.gate()`, threshold 0.6) once over the whole batch, routing
low-confidence, missing-field, or escalation-flagged labels to a
`review_queue` instead of ever discarding them.

### 3.5 Second-Annotator Protocol (Disclosed LLM Proxy)

No second human coder was available for this pilot. Instead of omitting
inter-annotator agreement or reusing one of the ablation grid's own arms
(which would not be an independent second opinion), we generate a second
label set for the 40-thread DEV split only using a model that appears
nowhere in the ablation grid (Anthropic `claude-haiku-4-5-20251001` under
the vocab condition), the closest match to what a human annotator would
actually be asked to do per the annotation instructions
(`pilot-data/corpus/labels_annotator_b/README.md`): choose values from the
closed vocabularies by name, and use independent judgment for escalation
instead of citing the routing-policy lookup table. **This is a disclosed
LLM proxy for a second human annotator, not a claim that it is one**; §5.4
reports its agreement with gold as a data point about label-scheme
reliability under one specific proxy, and §6.2 states this substitution as
an explicit limitation. A one-shot semantic repair pass (reusing the same
repair-instruction mechanism as §3.4, applied to `validate_label()`'s
errors instead of a parse error) resolves labels that parse but fail
validation; all 40 DEV-split threads have a valid `annotator_b` label.

---

## 4. EVALUATION METHODOLOGY

### 4.1 Five Routing-Evaluation Arms

`routing_eval.py` scores every arm only over the 200 EVAL-split threads
that have both a gold label and a schema-valid matched prediction
(`n_eval_split_matched`); the 40 DEV-split threads are reserved for the
majority baseline's fit and for inter-annotator agreement, never mixed into
the scored population.

- **oracle** — `route_from_fields(gold)` vs. gold. The ceiling: how well
  the deterministic policy reproduces gold routing given *perfect*
  classification fields. Not expected to be 1.0, since some gold rows are
  deliberately policy-divergent by design (`policy_audit.py`'s divergence
  census).
- **composed** — `route_from_fields(predicted)` vs. gold. The full
  pipeline: LLM classifies, the deterministic policy routes.
- **direct** — the LLM's own `assigned_reviewer`/`escalation` fields vs.
  gold, with no policy involved.
- **policy_fidelity** — `route_from_fields(predicted)` vs.
  `route_from_fields(gold)`. Holds the policy law fixed on both sides,
  isolating classification-field noise from gold/policy divergence noise.
- **majority** — the single most common DEV-split gold `assigned_reviewer`/
  `escalation` value, applied uniformly to every EVAL-split matched thread.
  Fit and scoring populations are disjoint by construction.

### 4.2 Metrics

Each arm reports reviewer-assignment accuracy and Cohen's κ, escalation
accuracy and κ (quadratic-weighted where urgency's ordinal structure
applies), and a joint "both correct" accuracy. The composed and direct arms
additionally report a **gate breakdown**: accuracy restricted to labels
`gating.gate()` routed to human review vs. accuracy on the remaining,
auto-routed labels. A prediction that fails `validate_label()` is excluded
from every arm and counted separately as `n_invalid_predictions`; `--strict`
turns any such exclusion, or any EVAL-split gold thread with no matched
valid prediction, into a nonzero exit code.

### 4.3 Inter-Annotator Agreement

`annotator_agreement.py` computes, per field: Krippendorff's α for nominal
and ordinal fields (quadratic-weighted for `urgency`), percent agreement,
MASI and micro-precision/recall/F1 for set-valued fields
(`secondary_disciplines`, `referenced_documents`), cosine similarity of
sentence embeddings for free-text fields, and a policy-decomposition
analysis (do the two annotators' own classification-field inputs, run
through the same deterministic routing policy, agree with each other and
with gold, independent of what each annotator's own `assigned_reviewer`
free choice was).

---

## 5. RESULTS

All numbers in this section are read directly from
`pilot-data/routing_eval-anthropic-{bare,vocab,policy}/routing_eval_report.json`,
`pilot-data/routing_eval-openai-{bare,vocab,policy}/routing_eval_report.json`,
and `pilot-data/annotator_agreement/agreement_report.json`, and the three
figures are rendered from the same files by `code/make_figures.py`
(no hand-transcribed figure data).

### 5.1 Schema Validity Requires Closed Vocabularies

Under the bare condition, **100% of 240 Anthropic outputs** and 239 of 240
OpenAI outputs fail `validate_label()` and are excluded from scoring
(`n_eval_split_matched = 0` for both providers' bare runs); the OpenAI
run's 240th thread is a separate failure mode, never producing a
parseable response at all (§5.5). Excluding that single non-parseable
case, the remaining 479 failures (240 Anthropic, 239 OpenAI) are not parsing
failures (each response parses as well-formed JSON) but a semantic one:
without the taxonomy in the prompt, the model free-texts
plausible-sounding but out-of-vocabulary values. A representative failure
(`eval-0001`, Anthropic bare) is rejected for four independent reasons
simultaneously: `rfi_type = "document coordination/conflict"` (not in
`RFI_TYPES`), `urgency = "medium"` (not in `URGENCY_TIERS`, which uses
`routine/priority/urgent/critical`), `csi_division = "Division 23 -
Heating, Ventilating, and Air Conditioning (HVAC)"` (a verbose paraphrase,
not the canonical `"23 00 00 (HVAC)"` string), and `assigned_reviewer =
"Mechanical Engineer of Record"` (not in the fixed ten-role taxonomy).
Under the vocab and policy conditions, by contrast, **0 of 240** outputs
fail validation for either condition, for either provider. Closed-vocabulary
injection is not an optional refinement to this task; it is the difference
between a zero-percent and a fully schema-valid pipeline.

### 5.2 Ablation Accuracy: Vocab vs. Policy

*(Figure 1: `figures/ablation-accuracy.png`)*

*Table 1: Reviewer assignment accuracy (Cohen's κ in parentheses) by
scoring arm, provider, and prompt condition, n=200 EVAL-split threads per
scored column; bare columns are n/a because those outputs are excluded
from scoring (§5.1).*

| Arm | Anthropic bare | Anthropic vocab | Anthropic policy | OpenAI bare | OpenAI vocab | OpenAI policy |
|---|---|---|---|---|---|---|
| oracle | n/a (0 scored) | 0.8550 (κ=0.834) | 0.8550 (κ=0.834) | n/a (0 scored) | 0.8550 (κ=0.834) | 0.8550 (κ=0.834) |
| composed | n/a | 0.7350 (κ=0.700) | 0.7400 (κ=0.705) | n/a | 0.7400 (κ=0.704) | 0.7400 (κ=0.705) |
| direct | n/a | 0.7000 (κ=0.658) | 0.7400 (κ=0.705) | n/a | 0.7150 (κ=0.674) | 0.7400 (κ=0.705) |
| policy_fidelity | n/a | 0.8600 (κ=0.837) | 0.8650 (κ=0.843) | n/a | 0.8700 (κ=0.848) | 0.8700 (κ=0.848) |
| majority | n/a | 0.0750 (κ=0.000) | 0.0750 (κ=0.000) | n/a | 0.0750 (κ=0.000) | 0.0750 (κ=0.000) |

(oracle and majority are computed from the label pool independent of the
LLM condition beyond which threads have a valid matched prediction; their
near-identical values across every column reflect that all four
provider/condition pairs produce a fully valid 200/200 EVAL-split match.)
This is now a completed 2-provider × 3-condition grid (bare excluded from
scoring by construction, per §5.1); see §5.5 for the cross-provider
comparison.

Composing the LLM's classification fields with the deterministic routing
policy (composed) reaches 73.5% (vocab) / 74.0% (policy) reviewer accuracy
against an 85.5% oracle ceiling: most of the residual gap to the
ceiling is attributable to classification-field noise, not to
policy/gold divergence, since `policy_fidelity` (which holds the policy law
fixed on both sides) is *higher* than composed for both conditions
(86.0% / 86.5%), close to the oracle's own 85.5%. Giving the LLM the
routing-policy text directly (`policy` condition) does **not** improve
composed accuracy over `vocab` alone in this pilot (73.5% → 74.0%, a
0.5-point difference on 200 threads); the closed vocabulary, not the
policy-table injection, is doing essentially all of the useful work,
matching §5.1's stronger finding that the vocabulary is what turns the
pipeline schema-valid at all. The `direct` arm (the LLM's own routing
guess, no policy) trails `composed` under `vocab` (70.0% vs. 73.5%) but
ties it under `policy` (74.0% both), plausibly because seeing the policy
text lets the model's own routing guess mimic the deterministic rule it was
shown, collapsing the composed/direct gap that vocab-only leaves open. The
`majority` baseline's 7.5% accuracy against a ten-role taxonomy confirms
the task is not trivially dominated by one reviewer role.

### 5.3 Confidence/Escalation Gating

*(Figure 2: `figures/gate-tradeoff.png`)*

Under vocab, the composed arm's 36 gated EVAL-split threads (3 by low
confidence, 34 by an escalation flag) score 38.9% reviewer accuracy versus
81.1% for the 164 ungated threads, an over-40-point split. Under policy,
56 threads are gated (55 by escalation, 1 by confidence), scoring 39.3% vs.
87.5% ungated. The number gated increases from 36 to 56 under policy,
plausibly because seeing the routing-policy text makes the model more
willing to flag `escalation=true`, without moving the gated subpopulation's
own accuracy (38.9% → 39.3%, essentially unchanged). Confidence/escalation
gating is doing its intended job: separating a small, genuinely
hard-to-route subset from a large majority the pipeline routes reliably,
not serving as a global accuracy dial.

### 5.4 Inter-Annotator Agreement (Disclosed LLM Proxy)

*(Figure 3: `figures/iaa-by-field.png`)*

Over the 40 DEV-split threads, gold vs. the disclosed `annotator_b` proxy
(§3.5) reach Krippendorff's α = 0.914 (primary_discipline), 0.818
(urgency), 0.716 (csi_division and, separately, assigned_reviewer, both
0.716), 0.507 (rfi_type), and 0.484 (escalation, the lowest of the reported
nominal/ordinal/boolean fields). cost_impact (α=1.000) and schedule_impact
(α=0.940) show near-perfect agreement; answer_in_documents is lower
(α=0.368) despite 92.5% raw percent agreement, consistent with a
skewed-prevalence field where chance-corrected agreement diverges sharply
from percent agreement. The policy-decomposition check finds only 40.0%
raw agreement between the two annotators' own five policy-input fields
taken jointly, yet a substantially higher reviewer-alpha of 0.943 when each
annotator's own inputs are separately run through the same deterministic
policy, i.e., the two annotators often reach the *same effective routing
decision* via inputs that differ field-by-field, which the policy
function's many-to-one structure absorbs.

### 5.5 Cross-Provider Replication

A parallel run with OpenAI `gpt-5` completes the full 2-provider ×
3-condition ablation grid. Under bare, OpenAI reproduces §5.1's finding
independently: 239 of 240 threads produced a response, and every one of
those 239 fails `validate_label()` for the same class of reasons as the
Anthropic bare run (out-of-vocabulary values absent a supplied taxonomy);
the 240th thread failed before producing a parseable response at all. Under
vocab and policy, OpenAI produces 240/240 schema-valid outputs for each
condition (the same 100% validity Anthropic achieves once the closed
vocabulary is present, §5.1), and all 200 EVAL-split threads match a gold
label with zero strict-mode failures for both conditions.

The resulting OpenAI accuracy numbers (Table 1, §5.2) closely track
Anthropic's: composed reviewer accuracy is 74.0% under both vocab and
policy for OpenAI (vs. Anthropic's 73.5% → 74.0%), and the `direct` arm
again trails `composed` under vocab (71.5% vs. 74.0%) before converging
under policy (74.0% both), the same qualitative pattern §5.2 reports for
Anthropic. `policy_fidelity` is 87.0% for OpenAI under both conditions,
against an identical 85.5% oracle ceiling (oracle and majority are
provider-independent given a fully valid, fully matched prediction set;
§5.2). Escalation gating also replicates directionally: OpenAI's composed
arm shows a gated/ungated accuracy split of 50.0% vs. 75.0% under vocab (8
gated EVAL-split threads) and 50.0% vs. 83.8% under policy (58 gated
threads); gating count grows sharply from vocab to policy for OpenAI too,
mirroring Anthropic's 36→56 growth in §5.3, though the two providers gate
different absolute counts (36/56 Anthropic vs. 8/58 OpenAI), which we
report without further interpretation given the single-run-per-condition
scope of this pilot (§6.2).

Two independent providers therefore agree, within 1.5 accuracy points on
every scoring arm, on both of this paper's central findings: closed-vocabulary
injection is what separates a 0%- from a fully schema-valid pipeline (§5.1),
and adding the routing-policy text on top of the vocabulary does not further
improve composed routing accuracy (§5.2). Within this experiment, this is
evidence the findings are a property of the vocab/policy prompt design and
task, not an artifact of one model's particular behavior; extending that
claim to other prompt phrasings, model families, or real (non-synthetic)
RFI text would require the replication §6.3 lays out, not this pilot alone.

---

## 6. DISCUSSION AND LIMITATIONS

### 6.1 Threats to Validity

The corpus is synthetic, not sampled from real project RFI logs — real RFIs
may exhibit discourse patterns, ambiguity, or discipline distributions this
generator's templates do not cover, and the 85.5% oracle ceiling is a
property of this corpus's deliberate policy-divergence rate, not a
universal constant. The fully-scored ablation (vocab vs. policy) now spans
two LLM providers (Anthropic and OpenAI, §5.5), each run once per condition;
it does not yet span multiple model families beyond these two commercial
providers, nor repeated sampling within a provider to estimate run-to-run
variance. Gold labels and the routing policy were authored by the same
research team that designed the ablation, which is why `policy_audit.py`'s
independent divergence census and this paper's `oracle` arm both exist as a
check against circularity, but a fully external validation would strengthen
this further.

### 6.2 Limitations (Explicit, In-Scope)

- **Synthetic corpus.** Real construction RFI text was not used or
  available for this pilot; external validity to real project logs is
  untested.
- **Disclosed LLM-proxy second annotator.** §3.5/§5.4's inter-annotator
  agreement numbers measure gold vs. one specific LLM's judgment, not
  human-human reliability. They should be read as a label-scheme
  reliability signal, not a substitute for a human IAA study.
- **Single sample per condition/provider.** Each of the six
  provider/condition cells (§5.5) was run once; point estimates have no
  resampling-based confidence interval, and no repeated-sampling or
  temperature sweep was performed to estimate run-to-run variance.
- **Single confidence-gate threshold.** Gating uses one fixed threshold
  (0.6); §5.3's gate-tradeoff numbers are specific to that choice.
- **240-thread pilot scale.** Per-stratum cell counts (e.g. individual
  adversarial categories, some with n≤5 DEV-split threads; see §5.4's
  stratified breakdown in `pilot-data/annotator_agreement/agreement_report.md`)
  are too small for some stratified agreement estimates to be
  statistically stable on their own.

### 6.3 Future Work

Repeat each provider/condition cell with multiple samples to estimate
run-to-run variance, instead of the single run per cell reported here;
extend the ablation to additional model families beyond the two commercial
providers studied; extend the corpus with a sample of real (de-identified)
project RFI text if a data-sharing agreement becomes available; recruit a
genuine second human annotator to replace or supplement §3.5's disclosed LLM
proxy; sweep the confidence-gate threshold instead of reporting a single
value.

---

## 7. CONCLUSION

This paper isolated *which part of an LLM prompt* is responsible for
reliable RFI classification and routing, instead of only measuring
whether an LLM can do the task at all. Across a stratified, adversarial,
anti-fabrication 240-thread synthetic corpus, the answer is unambiguous:
closed-vocabulary injection is the difference between a 0%- and a
100%-schema-valid pipeline (§5.1), while adding the deterministic
routing-policy text on top of the vocabulary produces no further
improvement in composed routing accuracy (73.5% vs. 74.0% on 200 EVAL-split
threads, §5.2), a result a single end-to-end accuracy number would not have
surfaced, and which the five-arm evaluation design (oracle / composed /
direct / policy_fidelity / majority) was built specifically to separate from
policy/gold divergence. Confidence/escalation gating further shows that most
of the residual error is concentrated in a small, identifiable subpopulation
(§5.3) instead of spread uniformly across the corpus, and a disclosed
LLM-proxy second annotator agrees with gold most on structured fields and
least on the single most judgment-dependent field, escalation (§5.4), a
pattern consistent with the broader LLM-as-annotator literature's finding
that LLM annotation reliability is uneven across field types, not a
uniform stand-in for a human coder [25]. A full replication against a second
commercial provider (§5.5) confirms both central findings within 1.5
accuracy points on every scoring arm: closed-vocabulary injection again
separates a 0%- from a fully schema-valid pipeline, and the policy-text
addition again produces no further improvement in composed routing accuracy
over vocab alone. Within this synthetic, single-run-per-condition pilot,
that agreement is evidence the findings are not one model's idiosyncrasy;
it does not establish that closed-vocabulary injection is necessary or
sufficient for schema validity across other prompt phrasings, model
families, or real (non-synthetic) RFI text, a stronger claim only the
replications named in §6.3 could support.

Two practical implications follow directly. First, teams deploying an LLM
for RFI (or, more generally, construction-document) classification should
treat closed-vocabulary prompt engineering as a required correctness
mechanism, not a stylistic refinement: the bare condition's complete
failure (§5.1) is not a tuning problem an ordinarily "better" prompt would
fix, since all but one of the 480 bare outputs across both providers
parsed as valid JSON and failed only on semantic vocabulary grounds (the
single exception being the one OpenAI response that never parsed at all,
§5.5). Second, a deterministic, auditable routing
policy composed on top of validated LLM classification fields is worth
retaining even when it does not lift accuracy over the classification fields
alone, because it makes the final routing decision traceable to a named
contractual rule (§3.3) instead of an opaque model judgment, a property
this paper's evaluation design can quantify (via `policy_fidelity` vs.
`composed`, §5.2) but that a single blended accuracy metric would obscure.

This is a pilot-scale study on a synthetic corpus with a single run per
provider/condition cell, and §6.2 states its limitations plainly instead of
implying a broader claim than the data supports. The most direct next steps
are exactly the limitations named in §6.2, not new work invented for this
section: repeated sampling to quantify run-to-run variance, validating
against real (de-identified) project RFI text, and replacing the disclosed
LLM proxy with a genuine second human annotator.

---

## CODE AND DATA AVAILABILITY

Corpus generator, prompt/vocabulary/policy modules, classification
pipeline, scoring, and agreement code: `code/`. Synthetic corpus, all raw
LLM predictions, and every evaluation/agreement report referenced in
Section 5: `pilot-data/`. Figures regenerate deterministically from those
JSON reports via `python code/make_figures.py`. Available at
`https://github.com/vns-shiva/rfi-classification-routing`.

---

## APPENDIX A: PROMPTS

All three conditions share one instruction sentence and one output-schema
block; `vocab` appends the vocabulary block, and `policy` appends the
vocabulary block plus the policy block (`prompts.build_classification_system_prompt()`
joins the applicable parts with a single blank line). Text below is
reproduced verbatim from `code/prompts.py` (`PROMPT_VERSION = "v1"`).

### A.1 Shared instruction sentence (all conditions)

```
You are an experienced construction project engineer classifying and routing
a Request for Information (RFI) thread for a commercial construction project.
```

### A.2 Shared output-schema block (all conditions)

```
Return a single JSON object (no prose, no markdown fences) with exactly these fields:

{
  "rfi_type": string,
  "primary_discipline": string,
  "secondary_disciplines": [string, ...],
  "csi_division": string,
  "urgency": string,
  "question_summary": string,
  "referenced_documents": [string, ...],
  "proposed_solution": string,
  "cost_impact": boolean,
  "schedule_impact": boolean,
  "answer_in_documents": boolean,
  "deadline_text": string,
  "assigned_reviewer": string,
  "routing_rationale": string,
  "escalation": boolean,
  "confidence": number
}

Field notes:
- secondary_disciplines: other disciplines materially involved, excluding primary_discipline. Empty list if none.
- csi_division: the CSI MasterFormat 2016 group-level division the RFI's subject matter falls under, or the
  literal string "—" if primary_discipline has no associated CSI division.
- proposed_solution: the submitter's proposed answer if they gave one, else the literal string "—".
- answer_in_documents: true only if the referenced documents already contain the answer to the question asked
  (the submitter should not have needed to ask).
- deadline_text: the verbatim phrase describing when a response is needed, or "none stated" if none is given.
- routing_rationale: one sentence explaining the assigned_reviewer choice.
- confidence: your confidence in this classification as a whole, in [0, 1].
```

The **bare** condition's system prompt is exactly A.1 + A.2, with nothing
else appended: no taxonomy, no policy.

### A.3 Vocabulary block (appended for `vocab` and `policy`)

```
Use only these closed vocabularies:

rfi_type (exactly one): {sorted RFI_TYPES, six values — see §3.1}

primary_discipline / secondary_disciplines (from this set): {sorted DISCIPLINES, eight values — see §3.1}

csi_division must be consistent with primary_discipline, from:
  {DISCIPLINE_TO_CSI, one line per discipline: its group-level CSI division(s), or "(no CSI division; use \"—\")" for General}

urgency (exactly one, low to high severity): {URGENCY_TIERS, in ordinal order: routine, priority, urgent, critical}

assigned_reviewer (exactly one): {REVIEWER_ROLES, ten values — see §3.1}
```

### A.4 Policy block (appended for `policy` only)

```
When choosing assigned_reviewer, apply this routing policy (adapted from AIA A201
§3.2.4 / ConsensusDocs 200 §12.2 ball-in-court practice):

1. If rfi_type is "code_compliance_question": route to "Code Official/AHJ Liaison",
   regardless of discipline.
2. Else if rfi_type is "substitution_request" and cost_impact is true: route to
   "Cost/Change-Order Manager".
3. Else if rfi_type is "field_condition_conflict" and schedule_impact is true and
   urgency is "urgent" or "critical": route to "GC Superintendent".
4. Otherwise: route to the design professional of record for primary_discipline
   (Architect of Record for Architectural or General; the matching discipline
   engineer otherwise; Plumbing/Fire Protection Engineer for both Plumbing and
   Fire_Protection).

Set escalation to true if urgency is "urgent" or "critical", or if both cost_impact
and schedule_impact are true.
```

Note that the model is shown this documented rule in plain language only
(never `routing_policy.py`'s source code), and that gold labels are never
derived from this policy (§3.2); the policy block's presence or absence is
exactly what distinguishes the `policy` condition from `vocab` in §5.2's
ablation.

### A.5 Per-thread user prompt (all conditions)

```
Project: {thread.project}
RFI number: {thread.rfi_number}

{thread.render()}

Classify and route this RFI thread. Respond with the JSON object only.
```

### A.6 Repair instruction (appended to the user prompt on a parse failure only)

Appended to the *user* prompt, never the system prompt: `classify.py`'s
response cache is keyed on `(system, user)`, so changing only the user text
on retry avoids replaying an identical cached failure forever. Used once,
bounded to a single retry, and only on an unparseable (not merely
schema-invalid) response (§3.4). `{truncated}` is the previous response
sliced to 1200 characters (`build_repair_instruction`'s `max_payload_chars`
default) so a wildly long malformed response cannot blow up retry token
cost.

```

Your previous response could not be parsed into the required JSON schema.

Previous response:
{truncated}

Problem: {errors}

Return ONLY a corrected, single JSON object matching the schema above — no prose, no markdown fences.
```

---

## References

[1] Panahi, R., Kivlin, J.-P., Louis, J. Request for Information (RFI)
Recommender System for Pre-Construction Design Review Application Using
Natural Language Processing, Chat-GPT, and Computer Vision. In: Computing in
Civil Engineering 2023. ASCE; 2023. https://doi.org/10.1061/9780784485224.020

[2] Kampelopoulos, D., Tsanousa, A., Vrochidis, S., Kompatsiaris, I. A
review of LLMs and their applications in the architecture, engineering and
construction industry. Artificial Intelligence Review 2025;58:1–46.
https://doi.org/10.1007/s10462-025-11241-7

[3] Gao, Y., Yiu, T.W., Shen, X., Tam, V.W. Large language models in smart
construction: a systematic review of implementation strategies, applications
and future directions. Engineering, Construction and Architectural
Management 2025;33(15):159–181. https://doi.org/10.1108/ECAM-10-2025-1668

[4] Tam, Z.R., Wu, C.-K., Tsai, Y.-L., Lin, C.-Y., Lee, H., Chen, Y.-N. Let
Me Speak Freely? A Study on the Impact of Format Restrictions on Performance
of Large Language Models. arXiv:2408.02442; 2024.

[5] Geng, S., Cooper, H., Moskal, M., Jenkins, S., Berman, J., Ranchin, N.,
West, R., Horvitz, E., Nori, H. JSONSchemaBench: A Rigorous Benchmark of
Structured Outputs for Language Models. arXiv:2501.10868; 2025.

[6] Patil, S.G., Zhang, T., Wang, X., Gonzalez, J.E. Gorilla: Large Language
Model Connected with Massive APIs. arXiv:2305.15334; 2023.

[7] Hanna, A.S., Tadt, E.J., Whited, G.C. Request for Information:
Benchmarks and Metrics for Major Highway Projects. Journal of Construction
Engineering and Management 2012;138(12):1347–1352.

[8] Mohamed, S., Tilley, P.A., Tucker, S.N. Quantifying the Time and Cost
Associated with the Request For Information (RFI) Process in Construction.
Construction Innovation 2007;7(1):35–50.

[9] Aibinu, A.A., Carter, S., Francis, V., Vaz-Serra, P. Request for
information frequency and their turnaround time in construction projects: A
data-analytic study. Built Environment Project and Asset Management
2020;10(1):1–15. https://doi.org/10.1108/BEPAM-10-2018-0130

[10] Joy, L.P., Sahyoun, S., Wang, J., Sourav, M.S.U., Zhao, Y., Yan, J., Ge,
H., Zeng, Y., Nik-Bakht, M. Enhancing Request for Information (RFI) Process
in Construction through Digitalization and Automation. In: Construction
Research Congress (CRC) 2025. CSCE/ASCE; 2025.
https://doi.org/10.22260/CRC-CSCE-2025/0202

[11] Xu, N., Zhou, X., Guo, C., Xiao, B., Wei, F., Hu, Y. Text Mining
Applications in the Construction Industry: Current Status, Research Gaps,
and Prospects. Sustainability 2022;14(24):16846.
https://doi.org/10.3390/su142416846

[12] Moon, S., Chi, S., Im, S.B. Automated detection of contractual risk
clauses from construction specifications using bidirectional encoder
representations from transformers (BERT). Automation in Construction
2022;142:104465. https://doi.org/10.1016/j.autcon.2022.104465

[13] Pham, H.T.T.L., Han, S. Natural Language Processing with Multitask
Classification for Semantic Prediction of Risk-Handling Actions in
Construction Contracts. Journal of Computing in Civil Engineering
2023;37(6):04023027. https://doi.org/10.1061/JCCEE5.CPENG-5218

[14] Dikmen, I., Eken, G., Erol, H., Birgonul, M.T. Automated construction
contract analysis for risk and responsibility assessment using natural
language processing and machine learning. Computers in Industry
2025;166:104251. https://doi.org/10.1016/j.compind.2025.104251

[15] Zheng, C., Wong, S., Su, X., Tang, Y., Nawaz, A., Kassem, M. Automating
construction contract review using knowledge graph-enhanced large language
models. arXiv:2309.12132; 2023.

[16] Erfani, A., Khanjar, H. Large Language Models for Construction Risk
Classification: A Comparative Study. Buildings 2025;15(18):3379.
https://doi.org/10.3390/buildings15183379

[17] Emara, M. Ontology-Driven GPT-4 Question Answering on Construction
Standards. In: Proceedings of the 36th Forum Bauinformatik. RWTH Aachen;
2025. https://doi.org/10.18154/RWTH-CONV-254874

[18] Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J.E.,
Kadous, M.W., Stoica, I. RouteLLM: Learning to Route LLMs with Preference
Data. arXiv:2406.18665; 2024.

[19] Zhou, L., Li, J. Few-Shot Large Language Models for Actionable Triage
Categorization of Online Patient Inquiries. arXiv:2605.15680; 2026.

[20] Madeyski, L. Triage: Routing Software Engineering Tasks to
Cost-Effective LLM Tiers via Code Quality Signals. arXiv:2604.07494; 2026.

[21] Cohen, J. Weighted kappa: Nominal scale agreement with provision for
scaled disagreement or partial credit. Psychological Bulletin
1968;70(4):213–220. https://doi.org/10.1037/h0026256

[22] Passonneau, R.J. Measuring Agreement on Set-valued Items (MASI) for
Semantic and Pragmatic Annotation. In: Proceedings of the Fifth
International Conference on Language Resources and Evaluation (LREC 2006).
ACL Anthology L06-1392; 2006.

[23] James, J. Counting on Consensus: Selecting the Right Inter-annotator
Agreement Metric for NLP Annotation and Evaluation. In: Proceedings of LREC
2026. arXiv:2603.06865; 2026.

[24] Gilardi, F., Alizadeh, M., Kubli, M. ChatGPT outperforms crowd workers
for text-annotation tasks. Proceedings of the National Academy of Sciences
2023;120(30):e2305016120. https://doi.org/10.1073/pnas.2305016120

[25] Gu, F., Li, Z., Colon, C.R., Evans, B., Mondal, I., Boyd-Graber, J.L.
Large Language Models Are Effective Human Annotation Assistants, But Not
Good Independent Annotators. In: Findings of the Association for
Computational Linguistics: ACL 2026. arXiv:2503.06778; 2026.

[26] Bojic, L., Zagovora, O., Zelenkauskaite, A., Vukovic, V., Cabarkapa, M.,
Veseljevic Jerkovic, S., Jovancevic, A. Comparing large language models and
human annotators in latent content analysis of sentiment, political
leaning, emotional intensity and sarcasm. Scientific Reports 2025;15.
https://doi.org/10.1038/s41598-025-96508-3

[27] Zhu, K., Wang, J., Zhou, J., Wang, Z., Chen, H., Wang, Y., Yang, L.,
Ye, W., Zhang, Y., Gong, N.Z., Xie, X. PromptBench: Towards Evaluating the
Robustness of Large Language Models on Adversarial Prompts.
arXiv:2306.04528; 2023.

[28] Hua, A., Tang, K., Gu, C., Gu, J., Wong, E., Qin, Y. Flaw or Artifact?
Rethinking Prompt Sensitivity in Evaluating LLMs. In: Proceedings of EMNLP
2025. arXiv:2509.01790; 2025.

[29] Egwim, C.N., Alaka, H., Demir, E., Balogun, H., Olu-Ajayi, R.,
Sulaimon, I., Wusu, G., Yusuf, W., Muideen, A.A. Artificial Intelligence in
the Construction Industry: A Systematic Review of the Entire Construction
Value Chain Lifecycle. Energies 2024;17(1):182.
https://doi.org/10.3390/en17010182

[30] Gill, E.Z., Cardone, D., Amelio, A. Revolutionizing the construction
industry by cutting edge artificial intelligence approaches: a review.
Frontiers in Artificial Intelligence 2024;7:1474932.
https://doi.org/10.3389/frai.2024.1474932

---

## Internal Tracking (not for submission)

### Open items before this is submission-ready

1. ~~**Blocking:** integrate the verified, real reference list — replace
   every `[C#-n]` placeholder in Section 2 (and any inline citations added
   to Section 1/6) with real numbered `elsarticle-num` citations, in
   first-citation order, and populate the References section (≥25
   entries).~~ Done — 30 verified references, [1]–[30], ordered by first
   citation (Introduction and Related Work).
2. ~~Rewrite Section 2 (Related Work) as full prose once citations
   land.~~ Done — 8 prose subsections (2.1–2.8), each grounded in the
   References list above.
3. ~~Tighten the Abstract to the venue's word limit.~~ Done — no explicit
   limit is stated in `submission-format-requirements.docx`. Originally
   trimmed from 403 to ~230 words; the OpenAI cross-provider replication
   sentence added afterward (item 7) grew it back to 360 words, still with
   no stated cap to violate — preserving every reported numeric result took
   priority over an arbitrary target length.
4. ~~Write Section 7 (Conclusion) once Related Work framing is
   settled.~~ Done — grounded directly in Section 5 (Results) and Section
   6.2 (Limitations); introduces no new unverified claims.
5. ~~Paste verbatim prompt text into Appendix A from `code/prompts.py`.~~
   Done — A.1/A.2/A.4/A.5/A.6 are exact verbatim text; A.3's four
   vocabulary lists are left as `{...}` placeholders pointing at
   `vocabulary.py`'s live constants rather than hand-copied, to inline at
   final build time (a script step, not a manual transcription, to avoid
   drift).
6. ~~Build the submission `.docx`~~ Done — `code/build_docx.py` (this
   paper's own script, not a reuse of the sibling ITcon paper's) renders
   elsarticle-num numbered in-text citations ([1]–[30]) against
   double-spaced body text with continuous line numbers, plus a
   **separate** `highlights.docx` (5 bullets, all ≤85 chars). Tables now
   render black-on-white (`Table Grid` style, explicit black borders/text,
   no blue accent) per the user's 2026-09-25 request.
7. ~~Decide whether/how to re-run OpenAI `vocab`/`policy` once the user adds
   API billing credits and, if so, re-run `code/make_figures.py` and update
   Section 5.5 and Table in 5.2 to include the completed grid.~~ Done —
   billing credits added 2026-09-24; both conditions re-ran at 240/240
   valid outputs, 200/200 EVAL-split matches, 0 strict-mode failures
   (`pilot-data/predicted-openai-{vocab,policy}`,
   `pilot-data/routing_eval-openai-{vocab,policy}/routing_eval_report.json`);
   `code/make_figures.py` re-run (Figure 1 now two-panel, cross-provider);
   Table in §5.2 and §5.5 updated with real completed-grid numbers; §6.1,
   §6.2, and §6.3 updated to drop the now-resolved billing-gap limitation
   and add the single-run-per-cell limitation it was masking.
8. Consider a graphical abstract (encouraged, not mandatory, per
   `submission-format-requirements.docx`).
