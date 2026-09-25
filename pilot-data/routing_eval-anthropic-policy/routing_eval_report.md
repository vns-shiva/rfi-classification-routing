# Routing Evaluation Report

Corpus: `C:\Users\itsupport\Desktop\Development\papers\rfi-classification-routing\pilot-data\corpus` | Predictions: `..\pilot-data\predicted-anthropic-policy` | Annotator: pipeline

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
- **reviewer_kappa**: 0.7053 (undefined_reason=None)
- **escalation_accuracy**: 0.8900 (178/200)
- **escalation_kappa**: 0.7332 (undefined_reason=None)
- **both_accuracy**: 0.6600 (132/200)
- **gate_breakdown**: gated 0.3929 (22/56) vs. ungated 0.8750 (126/144)
- **gate_breakdown.n_gated_by_category**: {'confidence': 1, 'escalation': 55}

## direct

- **n**: 200
- **reviewer_accuracy**: 0.7400 (148/200)
- **reviewer_kappa**: 0.7053 (undefined_reason=None)
- **escalation_accuracy**: 0.8900 (178/200)
- **escalation_kappa**: 0.7332 (undefined_reason=None)
- **both_accuracy**: 0.6600 (132/200)
- **gate_breakdown**: gated 0.3929 (22/56) vs. ungated 0.8750 (126/144)
- **gate_breakdown.n_gated_by_category**: {'confidence': 1, 'escalation': 55}

## policy_fidelity

- **n**: 200
- **reviewer_accuracy**: 0.8650 (173/200)
- **reviewer_kappa**: 0.8430 (undefined_reason=None)
- **escalation_accuracy**: 0.8900 (178/200)
- **escalation_kappa**: 0.7332 (undefined_reason=None)
- **both_accuracy**: 0.7800 (156/200)

## majority

- **n**: 200
- **reviewer_accuracy**: 0.0750 (15/200)
- **reviewer_kappa**: 0.0000 (undefined_reason=None)
- **escalation_accuracy**: 0.6950 (139/200)
- **escalation_kappa**: 0.0000 (undefined_reason=None)
- **both_accuracy**: 0.0750 (15/200)
