# PatchCore Output Validation

## Purpose

This document records the reusable validation applied to the saved PatchCore prediction, metric, error-analysis, and confusion-matrix artifacts.

The validation is implemented with the separate `ml-testing-validation-toolkit` package. It checks the consistency of already generated outputs without rerunning the Anomalib model.

## Validation Scope

The workflow validates:

- required prediction-table columns
- missing and infinite values
- duplicate image records
- allowed defect types and labels
- numerical label and anomaly-score ranges
- target-class representation
- prediction lengths and label values
- metric regression thresholds
- confusion-matrix values and schema
- text-to-numeric label mapping
- correctness flags
- error-analysis table consistency
- source JSON report consistency
- project-relative output paths

## Current Validation Summary

| Item | Result |
|---|---:|
| Overall status | PASS |
| Total checks | 21 |
| Passed checks | 21 |
| Failed checks | 0 |
| Prediction rows | 83 |
| Error rows | 2 |

## Recalculated Image-Level Metrics

| Metric | Result |
|---|---:|
| Accuracy | 0.9759 |
| Precision | 0.9841 |
| Recall | 0.9841 |
| F1-score | 0.9841 |

Metric thresholds are development regression checks, not production acceptance criteria.

## Individual Checks

| Check | Status | Message |
|---|---|---|
| `required_columns` | PASS | All 10 required columns are present. |
| `missing_values` | PASS | Missing-value fractions are within the allowed limit of 0.000. |
| `infinite_values` | PASS | No infinite values were found. |
| `duplicate_rows` | PASS | Duplicate count 0 is within the allowed limit of 0. |
| `allowed_values:defect_type` | PASS | Column 'defect_type' contains only allowed values. |
| `allowed_values:true_label` | PASS | Column 'true_label' contains only allowed values. |
| `allowed_values:predicted_label` | PASS | Column 'predicted_label' contains only allowed values. |
| `allowed_values:true_numeric` | PASS | Column 'true_numeric' contains only allowed values. |
| `allowed_values:predicted_numeric` | PASS | Column 'predicted_numeric' contains only allowed values. |
| `numeric_ranges` | PASS | All 3 configured numeric range checks passed. |
| `class_balance:true_label` | PASS | Target 'true_label' satisfies the configured class-balance requirements. |
| `prediction_lengths` | PASS | All model outputs contain 83 sample(s). |
| `prediction_labels` | PASS | All labels are within the configured allowed set. |
| `score_range` | PASS | All scores are within the configured range. |
| `metric_thresholds` | PASS | All 4 configured metric threshold checks passed. |
| `confusion_matrix_consistency` | PASS | The supplied confusion matrix matches the predictions. |
| `label_mapping_consistency` | PASS | Text labels, numeric labels, and correctness flags are internally consistent. |
| `error_table_consistency` | PASS | The error table contains exactly the incorrect predictions. |
| `source_report_consistency` | PASS | The PatchCore JSON report matches the prediction table. |
| `confusion_matrix_schema` | PASS | The confusion-matrix row and column labels are correct. |
| `portable_output_paths` | PASS | Committed PatchCore output paths are project-relative. |

## Validated Source Files

```text
results\patchcore_predictions.csv
results\patchcore_error_analysis.csv
results\patchcore_report.json
results\patchcore_confusion_matrix.csv
```

## Generated Validation Outputs

```text
results/patchcore_validation_report.json
results/patchcore_validation_checks.csv
```

## Interpretation

A passing report means the saved PatchCore artifacts are internally consistent, contain the expected schema, reproduce the reported image-level metrics, and do not expose private absolute paths.

It does not prove production readiness. The underlying model is still evaluated only on the controlled MVTec AD `bottle` benchmark. Threshold calibration, robustness testing, evaluation on additional object categories, and production monitoring remain future work.
