# Routing Evaluation Report

Corpus: `pilot-data\corpus` | Predictions: `pilot-data\predicted-openai-vocab` | Annotator: pipeline

- **n_gold**: 240
- **n_predicted**: 240
- **n_matched**: 240
- **n_eval_split_matched**: 200 (the scored population for every arm below)
- **n_dev_split_gold**: 40 (fits the majority baseline)
- **missing_thread_ids**: 0
- **extra_thread_ids**: 0 (predicted, no matching gold label)
- **n_invalid_predictions**: 0 (failed schema.validate_label(), excluded from scoring)
- **gate_threshold**: 0.6
- **majority_reviewer**: 'Code Official/AHJ Liaison'
- **majority_escalation**: False

## oracle

- **n**: 200
- **reviewer_accuracy**: 0.8550 (171/200)
- **reviewer_kappa**: 0.8342 (undefined_reason=None)
- **escalation_accuracy**: 1.0000 (200/200)
- **escalation_kappa**: 1.0000 (undefined_reason=None)
- **both_accuracy**: 0.8550 (171/200)

## composed

- **n**: 200
- **reviewer_accuracy**: 0.7400 (148/200)
- **reviewer_kappa**: 0.7043 (undefined_reason=None)
- **escalation_accuracy**: 0.8750 (175/200)
- **escalation_kappa**: 0.7092 (undefined_reason=None)
- **both_accuracy**: 0.6250 (125/200)
- **gate_breakdown**: gated 0.5000 (4/8) vs. ungated 0.7500 (144/192)
- **gate_breakdown.n_gated_by_category**: {'escalation': 8}

## direct

- **n**: 200
- **reviewer_accuracy**: 0.7150 (143/200)
- **reviewer_kappa**: 0.6737 (undefined_reason=None)
- **escalation_accuracy**: 0.7250 (145/200)
- **escalation_kappa**: 0.1422 (undefined_reason=None)
- **both_accuracy**: 0.6000 (120/200)
- **gate_breakdown**: gated 0.5000 (4/8) vs. ungated 0.7240 (139/192)
- **gate_breakdown.n_gated_by_category**: {'escalation': 8}

## policy_fidelity

- **n**: 200
- **reviewer_accuracy**: 0.8700 (174/200)
- **reviewer_kappa**: 0.8482 (undefined_reason=None)
- **escalation_accuracy**: 0.8750 (175/200)
- **escalation_kappa**: 0.7092 (undefined_reason=None)
- **both_accuracy**: 0.7500 (150/200)

## majority

- **n**: 200
- **reviewer_accuracy**: 0.0750 (15/200)
- **reviewer_kappa**: 0.0000 (undefined_reason=None)
- **escalation_accuracy**: 0.6950 (139/200)
- **escalation_kappa**: 0.0000 (undefined_reason=None)
- **both_accuracy**: 0.0750 (15/200)
