# Routing Evaluation Report

Corpus: `C:\Users\itsupport\Desktop\Development\papers\rfi-classification-routing\pilot-data\corpus` | Predictions: `..\pilot-data\predicted-anthropic-vocab` | Annotator: pipeline

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
- **reviewer_accuracy**: 0.7350 (147/200)
- **reviewer_kappa**: 0.6998 (undefined_reason=None)
- **escalation_accuracy**: 0.8750 (175/200)
- **escalation_kappa**: 0.7038 (undefined_reason=None)
- **both_accuracy**: 0.6500 (130/200)
- **gate_breakdown**: gated 0.3889 (14/36) vs. ungated 0.8110 (133/164)
- **gate_breakdown.n_gated_by_category**: {'confidence': 3, 'escalation': 34}

## direct

- **n**: 200
- **reviewer_accuracy**: 0.7000 (140/200)
- **reviewer_kappa**: 0.6575 (undefined_reason=None)
- **escalation_accuracy**: 0.7650 (153/200)
- **escalation_kappa**: 0.3671 (undefined_reason=None)
- **both_accuracy**: 0.5600 (112/200)
- **gate_breakdown**: gated 0.3611 (13/36) vs. ungated 0.7744 (127/164)
- **gate_breakdown.n_gated_by_category**: {'confidence': 3, 'escalation': 34}

## policy_fidelity

- **n**: 200
- **reviewer_accuracy**: 0.8600 (172/200)
- **reviewer_kappa**: 0.8372 (undefined_reason=None)
- **escalation_accuracy**: 0.8750 (175/200)
- **escalation_kappa**: 0.7038 (undefined_reason=None)
- **both_accuracy**: 0.7700 (154/200)

## majority

- **n**: 200
- **reviewer_accuracy**: 0.0750 (15/200)
- **reviewer_kappa**: 0.0000 (undefined_reason=None)
- **escalation_accuracy**: 0.6950 (139/200)
- **escalation_kappa**: 0.0000 (undefined_reason=None)
- **both_accuracy**: 0.0750 (15/200)
