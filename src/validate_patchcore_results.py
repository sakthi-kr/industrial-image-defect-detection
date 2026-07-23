"""Validate saved PatchCore prediction and report artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml_validation_toolkit import (
    ValidationResult,
    print_validation_summary,
    raise_for_validation_failures,
    run_data_checks,
    run_model_checks,
    save_validation_csv,
    save_validation_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = PROJECT_ROOT / "results"

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY / "patchcore_predictions.csv"
)
ERRORS_PATH = (
    RESULTS_DIRECTORY / "patchcore_error_analysis.csv"
)
SOURCE_REPORT_PATH = (
    RESULTS_DIRECTORY / "patchcore_report.json"
)
CONFUSION_MATRIX_PATH = (
    RESULTS_DIRECTORY / "patchcore_confusion_matrix.csv"
)

VALIDATION_JSON_PATH = (
    RESULTS_DIRECTORY / "patchcore_validation_report.json"
)
VALIDATION_CSV_PATH = (
    RESULTS_DIRECTORY / "patchcore_validation_checks.csv"
)

ALLOWED_LABELS = [
    "normal",
    "defective",
]

ALLOWED_NUMERIC_LABELS = [
    0,
    1,
]

ALLOWED_DEFECT_TYPES = [
    "good",
    "broken_large",
    "broken_small",
    "contamination",
]

REQUIRED_COLUMNS = [
    "image_path",
    "file_name",
    "defect_type",
    "true_label",
    "true_numeric",
    "predicted_label",
    "predicted_numeric",
    "pred_score",
    "correct",
    "has_anomaly_map",
]


def make_result(
    name: str,
    passed: bool,
    pass_message: str,
    fail_message: str,
    details: dict[str, Any],
) -> ValidationResult:
    """Create one custom validation result."""
    return ValidationResult(
        name=name,
        passed=passed,
        message=pass_message if passed else fail_message,
        details=details,
    )


def load_csv(
    path: Path,
    *,
    index_col: int | None = None,
) -> pd.DataFrame:
    """Load a required CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run `python src/generate_patchcore_report.py` first."
        )

    return pd.read_csv(
        path,
        index_col=index_col,
    )


def load_json_object(
    path: Path,
) -> dict[str, Any]:
    """Load a required JSON object."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        value = json.load(file)

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected a JSON object in: {path}"
        )

    return value


def calculate_metrics(
    predictions: pd.DataFrame,
) -> dict[str, float]:
    """Recalculate image-level metrics from the prediction table."""
    y_true = predictions[
        "true_numeric"
    ].astype(int)

    y_pred = predictions[
        "predicted_numeric"
    ].astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
    }


def check_label_mapping(
    predictions: pd.DataFrame,
) -> ValidationResult:
    """
    Validate text labels, numeric labels, and correctness flags.
    """
    expected_true_numeric = predictions[
        "true_label"
    ].map(
        {
            "normal": 0,
            "defective": 1,
        }
    )

    expected_predicted_numeric = predictions[
        "predicted_label"
    ].map(
        {
            "normal": 0,
            "defective": 1,
        }
    )

    expected_correct = (
        predictions["true_numeric"]
        == predictions["predicted_numeric"]
    )

    observed_correct = (
        predictions["correct"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    true_mapping_mismatches = predictions.index[
        expected_true_numeric
        != predictions["true_numeric"]
    ].tolist()

    predicted_mapping_mismatches = predictions.index[
        expected_predicted_numeric
        != predictions["predicted_numeric"]
    ].tolist()

    invalid_correct_rows = predictions.index[
        observed_correct.isna()
    ].tolist()

    correct_flag_mismatches = predictions.index[
        observed_correct.fillna(False)
        != expected_correct
    ].tolist()

    passed = not (
        true_mapping_mismatches
        or predicted_mapping_mismatches
        or invalid_correct_rows
        or correct_flag_mismatches
    )

    return make_result(
        name="label_mapping_consistency",
        passed=passed,
        pass_message=(
            "Text labels, numeric labels, and correctness "
            "flags are internally consistent."
        ),
        fail_message=(
            "Saved labels or correctness flags are inconsistent."
        ),
        details={
            "true_label_mismatch_rows": (
                true_mapping_mismatches
            ),
            "predicted_label_mismatch_rows": (
                predicted_mapping_mismatches
            ),
            "invalid_correct_value_rows": (
                invalid_correct_rows
            ),
            "correct_flag_mismatch_rows": (
                correct_flag_mismatches
            ),
        },
    )


def check_error_table(
    predictions: pd.DataFrame,
    errors: pd.DataFrame,
) -> ValidationResult:
    """
    Check that the error table contains exactly the incorrect rows.
    """
    if "image_path" not in errors.columns:
        return make_result(
            name="error_table_consistency",
            passed=False,
            pass_message=(
                "The error table contains exactly the "
                "incorrect predictions."
            ),
            fail_message=(
                "The error table is missing the image_path column."
            ),
            details={
                "observed_columns": list(errors.columns),
            },
        )

    expected_paths = sorted(
        predictions.loc[
            predictions["true_numeric"]
            != predictions["predicted_numeric"],
            "image_path",
        ]
        .astype(str)
        .tolist()
    )

    observed_paths = sorted(
        errors["image_path"]
        .astype(str)
        .tolist()
    )

    missing_paths = sorted(
        set(expected_paths)
        - set(observed_paths)
    )

    unexpected_paths = sorted(
        set(observed_paths)
        - set(expected_paths)
    )

    passed = (
        len(expected_paths)
        == len(observed_paths)
        and not missing_paths
        and not unexpected_paths
    )

    return make_result(
        name="error_table_consistency",
        passed=passed,
        pass_message=(
            "The error table contains exactly the "
            "incorrect predictions."
        ),
        fail_message=(
            "The error table does not match the "
            "incorrect predictions."
        ),
        details={
            "expected_error_count": len(expected_paths),
            "observed_error_count": len(observed_paths),
            "missing_image_paths": missing_paths,
            "unexpected_image_paths": unexpected_paths,
        },
    )


def check_source_report(
    predictions: pd.DataFrame,
    source_report: dict[str, Any],
    metrics: dict[str, float],
) -> ValidationResult:
    """
    Compare the JSON report with values recalculated from the CSV.
    """
    y_true = predictions[
        "true_numeric"
    ].astype(int)

    y_pred = predictions[
        "predicted_numeric"
    ].astype(int)

    expected_values: dict[str, int | float] = {
        "n_test_images": int(
            len(predictions)
        ),
        "n_normal_images": int(
            (y_true == 0).sum()
        ),
        "n_defective_images": int(
            (y_true == 1).sum()
        ),
        "false_positives": int(
            (
                (y_true == 0)
                & (y_pred == 1)
            ).sum()
        ),
        "false_negatives": int(
            (
                (y_true == 1)
                & (y_pred == 0)
            ).sum()
        ),
        "n_errors": int(
            (y_true != y_pred).sum()
        ),
        **metrics,
    }

    integer_fields = {
        "n_test_images",
        "n_normal_images",
        "n_defective_images",
        "false_positives",
        "false_negatives",
        "n_errors",
    }

    missing_fields: list[str] = []
    mismatches: dict[str, dict[str, Any]] = {}

    for field_name, expected_value in expected_values.items():
        if field_name not in source_report:
            missing_fields.append(
                field_name
            )
            continue

        observed_value = source_report[
            field_name
        ]

        try:
            if field_name in integer_fields:
                matches = (
                    int(observed_value)
                    == expected_value
                )
            else:
                matches = math.isclose(
                    float(observed_value),
                    float(expected_value),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
        except (TypeError, ValueError):
            matches = False

        if not matches:
            mismatches[field_name] = {
                "expected": expected_value,
                "observed": observed_value,
            }

    passed = (
        not missing_fields
        and not mismatches
    )

    return make_result(
        name="source_report_consistency",
        passed=passed,
        pass_message=(
            "The PatchCore JSON report matches "
            "the prediction table."
        ),
        fail_message=(
            "The PatchCore JSON report does not "
            "match the prediction table."
        ),
        details={
            "missing_fields": missing_fields,
            "mismatches": mismatches,
            "expected_values": expected_values,
        },
    )


def check_confusion_matrix_schema(
    matrix: pd.DataFrame,
) -> ValidationResult:
    """Validate confusion-matrix row and column names."""
    expected_rows = [
        "true_normal",
        "true_defective",
    ]

    expected_columns = [
        "pred_normal",
        "pred_defective",
    ]

    observed_rows = [
        str(value)
        for value in matrix.index
    ]

    observed_columns = [
        str(value)
        for value in matrix.columns
    ]

    passed = (
        observed_rows == expected_rows
        and observed_columns == expected_columns
    )

    return make_result(
        name="confusion_matrix_schema",
        passed=passed,
        pass_message=(
            "The confusion-matrix row and column "
            "labels are correct."
        ),
        fail_message=(
            "The confusion-matrix row or column "
            "labels are incorrect."
        ),
        details={
            "expected_rows": expected_rows,
            "observed_rows": observed_rows,
            "expected_columns": expected_columns,
            "observed_columns": observed_columns,
        },
    )


def is_absolute_path_text(
    value: Any,
) -> bool:
    """Detect Windows and Unix absolute paths stored as text."""
    text = str(value).replace(
        "\\",
        "/",
    )

    return (
        text.startswith("/")
        or (
            len(text) >= 3
            and text[1:3] == ":/"
        )
    )


def check_portable_paths(
    predictions: pd.DataFrame,
    source_report: dict[str, Any],
) -> ValidationResult:
    """
    Require project-relative paths in committed result files.
    """
    absolute_image_paths = [
        value
        for value in (
            predictions["image_path"]
            .dropna()
            .astype(str)
        )
        if is_absolute_path_text(value)
    ]

    checkpoint_path = source_report.get(
        "checkpoint_path"
    )

    checkpoint_is_absolute = (
        checkpoint_path is not None
        and is_absolute_path_text(
            checkpoint_path
        )
    )

    passed = (
        not absolute_image_paths
        and not checkpoint_is_absolute
    )

    return make_result(
        name="portable_output_paths",
        passed=passed,
        pass_message=(
            "Committed PatchCore output paths "
            "are project-relative."
        ),
        fail_message=(
            "One or more committed PatchCore "
            "output paths are absolute."
        ),
        details={
            "absolute_image_path_count": len(
                absolute_image_paths
            ),
            "absolute_image_paths": (
                absolute_image_paths[:20]
            ),
            "checkpoint_path": checkpoint_path,
            "checkpoint_path_is_absolute": (
                checkpoint_is_absolute
            ),
        },
    )


def main() -> None:
    predictions = load_csv(
        PREDICTIONS_PATH
    )

    errors = load_csv(
        ERRORS_PATH
    )

    source_report = load_json_object(
        SOURCE_REPORT_PATH
    )

    confusion_table = load_csv(
        CONFUSION_MATRIX_PATH,
        index_col=0,
    )

    print("PatchCore result validation")
    print("===========================")
    print(
        f"Prediction table : "
        f"{len(predictions)} rows × "
        f"{len(predictions.columns)} columns"
    )
    print(
        f"Error rows       : "
        f"{len(errors)}"
    )
    print(
        f"Confusion matrix : "
        f"{confusion_table.shape}"
    )

    data_results = run_data_checks(
        predictions,
        required_columns=REQUIRED_COLUMNS,
        missing_value_columns=REQUIRED_COLUMNS,
        max_missing_fraction=0.0,
        infinite_value_columns=[
            "true_numeric",
            "predicted_numeric",
            "pred_score",
        ],
        duplicate_subset=[
            "image_path",
        ],
        max_duplicates=0,
        allowed_values={
            "defect_type": (
                ALLOWED_DEFECT_TYPES
            ),
            "true_label": (
                ALLOWED_LABELS
            ),
            "predicted_label": (
                ALLOWED_LABELS
            ),
            "true_numeric": (
                ALLOWED_NUMERIC_LABELS
            ),
            "predicted_numeric": (
                ALLOWED_NUMERIC_LABELS
            ),
        },
        numeric_ranges={
            "true_numeric": (
                0.0,
                1.0,
            ),
            "predicted_numeric": (
                0.0,
                1.0,
            ),
            "pred_score": (
                0.0,
                1.0,
            ),
        },
        target_column="true_label",
        min_classes=2,
        min_samples_per_class=1,
    )

    metrics = calculate_metrics(
        predictions
    )

    y_true = predictions[
        "true_numeric"
    ].astype(int)

    y_pred = predictions[
        "predicted_numeric"
    ].astype(int)

    model_results = run_model_checks(
        y_true=y_true,
        y_pred=y_pred,
        allowed_labels=(
            ALLOWED_NUMERIC_LABELS
        ),
        scores=predictions[
            "pred_score"
        ],
        score_minimum=0.0,
        score_maximum=1.0,
        metrics=metrics,
        metric_minimums={
            "accuracy": 0.90,
            "precision": 0.90,
            "recall": 0.90,
            "f1_score": 0.90,
        },
        matrix=confusion_table.to_numpy(),
        confusion_matrix_labels=(
            ALLOWED_NUMERIC_LABELS
        ),
    )

    custom_results = [
        check_label_mapping(
            predictions
        ),
        check_error_table(
            predictions,
            errors,
        ),
        check_source_report(
            predictions,
            source_report,
            metrics,
        ),
        check_confusion_matrix_schema(
            confusion_table
        ),
        check_portable_paths(
            predictions,
            source_report,
        ),
    ]

    all_results = [
        *data_results,
        *model_results,
        *custom_results,
    ]

    expected_matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=ALLOWED_NUMERIC_LABELS,
    )

    save_validation_json(
        all_results,
        VALIDATION_JSON_PATH,
        report_name=(
            "PatchCore Output Validation Report"
        ),
        metadata={
            "project": (
                "industrial-image-defect-detection"
            ),
            "model": "PatchCore",
            "dataset": "MVTec AD bottle",
            "prediction_rows": int(
                len(predictions)
            ),
            "error_rows": int(
                len(errors)
            ),
            "calculated_metrics": metrics,
            "expected_confusion_matrix": (
                expected_matrix.tolist()
            ),
            "source_files": [
                str(
                    PREDICTIONS_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
                str(
                    ERRORS_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
                str(
                    SOURCE_REPORT_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
                str(
                    CONFUSION_MATRIX_PATH.relative_to(
                        PROJECT_ROOT
                    )
                ),
            ],
            "threshold_interpretation": (
                "Metric thresholds are development "
                "regression checks, not production "
                "acceptance criteria."
            ),
        },
    )

    save_validation_csv(
        all_results,
        VALIDATION_CSV_PATH,
    )

    print()

    print_validation_summary(
        all_results,
        report_name=(
            "PatchCore Output Validation"
        ),
    )

    print(
        "\nCalculated image-level metrics"
    )
    print(
        "------------------------------"
    )

    for metric_name, metric_value in metrics.items():
        print(
            f"{metric_name:10}: "
            f"{metric_value:.4f}"
        )

    print("\nSaved outputs")
    print("-------------")
    print(
        f"JSON report: "
        f"{VALIDATION_JSON_PATH}"
    )
    print(
        f"CSV checks : "
        f"{VALIDATION_CSV_PATH}"
    )

    raise_for_validation_failures(
        all_results,
        message_prefix=(
            "PatchCore output validation failed"
        ),
    )


if __name__ == "__main__":
    main()
