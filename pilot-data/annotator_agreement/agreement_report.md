# Annotator Agreement Report

Framing: **inter_annotator** | Annotators: annotator_b, gold | Units: 40

## Field-level agreement

| Field | Kind | Reportable | n_applicable | Summary |
|---|---|---|---|---|
| rfi_type | nominal | True | 40 | alpha=0.507; undefined_reason=n/a; percent_agreement=0.600 |
| primary_discipline | nominal | True | 40 | alpha=0.914; undefined_reason=n/a; percent_agreement=0.925 |
| secondary_disciplines | set | True | 40 | mean_masi=0.275; exact_match_rate=0.725; micro_precision=1.000; micro_recall=0.421; micro_f1=0.593 |
| csi_division | nominal | True | 39 | alpha=0.716; undefined_reason=n/a; percent_agreement=0.744; n_excluded_general_general=1 |
| urgency | ordinal | True | 40 | alpha=0.818; undefined_reason=n/a; percent_agreement=0.825 |
| question_summary | text | True | 40 | mean_cosine_similarity=0.625; median_cosine_similarity=0.614 |
| referenced_documents | set | True | 40 | mean_masi=0.356; exact_match_rate=0.575; micro_precision=0.889; micro_recall=0.727; micro_f1=0.800 |
| proposed_solution | text | True | 0 | sentinel_agreement_rate=0.475; n_both_sentinel=19; n_sentinel_mismatch=21; n_both_real=0; mean_cosine_similarity_when_both_real=n/a |
| cost_impact | boolean | True | 40 | alpha=1.000; undefined_reason=n/a; percent_agreement=1.000 |
| schedule_impact | boolean | True | 40 | alpha=0.940; undefined_reason=n/a; percent_agreement=0.975 |
| answer_in_documents | boolean | True | 40 | alpha=0.368; undefined_reason=n/a; percent_agreement=0.925 |
| deadline_text | text | True | 40 | mean_cosine_similarity=0.493; median_cosine_similarity=0.449 |
| assigned_reviewer | nominal | True | 40 | alpha=0.716; undefined_reason=n/a; percent_agreement=0.750 |
| routing_rationale | text | True | 40 | mean_cosine_similarity=0.153; median_cosine_similarity=0.131 |
| escalation | boolean | True | 40 | alpha=0.484; undefined_reason=n/a; percent_agreement=0.825 |
| confidence | constant | False | 40 | n_mismatched=40 |
| status | constant | False | 40 | n_mismatched=0 |
| deadline_iso | date | True | 8 | mean_abs_day_offset=0.000; median_abs_day_offset=0.000; exact_match_rate=1.000; n_status_mismatch=8 |
| deadline_resolution_status | nominal | True | 40 | alpha=0.530; undefined_reason=n/a; percent_agreement=0.800 |

## Warnings
- confidence is bookkeeping, not scored for agreement, but differed across annotators on 40/40 units

## Policy-decomposition agreement
- Input agreement rate (5 policy fields): 0.400 (40 units)
- Reviewer alpha from each annotator's own inputs: 0.943
- Escalation agreement from each annotator's own inputs: 0.875
- Policy-validity (gold vs. policy(own inputs)) for annotator_b: 0.875
- Policy-validity (gold vs. policy(own inputs)) for gold: 0.925
- Escalation-validity for annotator_b: 0.900
- Escalation-validity for gold: 1.000
- Reviewer alpha given input agreement (16 units): 0.770

## Stratified by `adversarial`

### answer_in_documents (n=3)
- rfi_type: alpha=0.000; undefined_reason=n/a; percent_agreement=0.667
- urgency: alpha=1.000; undefined_reason=n/a; percent_agreement=1.000
- assigned_reviewer: alpha=-0.667; undefined_reason=n/a; percent_agreement=0.000

### boundary_urgency (n=5)
- rfi_type: alpha=n/a; undefined_reason=no_variance; percent_agreement=1.000
- urgency: alpha=-0.125; undefined_reason=n/a; percent_agreement=0.600
- assigned_reviewer: alpha=0.000; undefined_reason=n/a; percent_agreement=0.800

### distractor_cost_language (n=3)
- rfi_type: alpha=n/a; undefined_reason=no_variance; percent_agreement=1.000
- urgency: alpha=n/a; undefined_reason=no_variance; percent_agreement=1.000
- assigned_reviewer: alpha=n/a; undefined_reason=no_variance; percent_agreement=1.000

### multi_discipline (n=4)
- rfi_type: alpha=0.000; undefined_reason=n/a; percent_agreement=0.750
- urgency: alpha=-0.161; undefined_reason=n/a; percent_agreement=0.500
- assigned_reviewer: alpha=0.000; undefined_reason=n/a; percent_agreement=0.750

### none (n=20)
- rfi_type: alpha=-0.208; undefined_reason=n/a; percent_agreement=0.300
- urgency: alpha=0.830; undefined_reason=n/a; percent_agreement=0.900
- assigned_reviewer: alpha=0.802; undefined_reason=n/a; percent_agreement=0.850

### type_ambiguity (n=4)
- rfi_type: alpha=n/a; undefined_reason=no_variance; percent_agreement=1.000
- urgency: alpha=0.000; undefined_reason=n/a; percent_agreement=0.750
- assigned_reviewer: alpha=-0.167; undefined_reason=n/a; percent_agreement=0.500

### unresolvable_deadline (n=1)
- rfi_type: alpha=n/a; undefined_reason=no_variance; percent_agreement=1.000
- urgency: alpha=n/a; undefined_reason=no_variance; percent_agreement=1.000
- assigned_reviewer: alpha=n/a; undefined_reason=no_variance; percent_agreement=1.000

## Stratified by `rfi_type`

### code_compliance_question (n=7)
- rfi_type: alpha=0.000; undefined_reason=n/a; percent_agreement=0.857
- urgency: alpha=0.857; undefined_reason=n/a; percent_agreement=0.857
- assigned_reviewer: alpha=-0.275; undefined_reason=n/a; percent_agreement=0.286

### coordination_conflict (n=4)
- rfi_type: alpha=0.000; undefined_reason=n/a; percent_agreement=0.750
- urgency: alpha=-0.161; undefined_reason=n/a; percent_agreement=0.500
- assigned_reviewer: alpha=0.000; undefined_reason=n/a; percent_agreement=0.750

### design_clarification (n=14)
- rfi_type: alpha=-0.929; undefined_reason=n/a; percent_agreement=0.000
- urgency: alpha=0.765; undefined_reason=n/a; percent_agreement=0.929
- assigned_reviewer: alpha=0.755; undefined_reason=n/a; percent_agreement=0.857

### document_discrepancy (n=5)
- rfi_type: alpha=n/a; undefined_reason=no_variance; percent_agreement=1.000
- urgency: alpha=0.978; undefined_reason=n/a; percent_agreement=0.800
- assigned_reviewer: alpha=n/a; undefined_reason=no_variance; percent_agreement=1.000

### field_condition_conflict (n=5)
- rfi_type: alpha=n/a; undefined_reason=no_variance; percent_agreement=1.000
- urgency: alpha=-0.125; undefined_reason=n/a; percent_agreement=0.600
- assigned_reviewer: alpha=0.000; undefined_reason=n/a; percent_agreement=0.800

### substitution_request (n=5)
- rfi_type: alpha=n/a; undefined_reason=no_variance; percent_agreement=1.000
- urgency: alpha=n/a; undefined_reason=no_variance; percent_agreement=1.000
- assigned_reviewer: alpha=0.000; undefined_reason=n/a; percent_agreement=0.800

## Disagreements: 234 field-level mismatches