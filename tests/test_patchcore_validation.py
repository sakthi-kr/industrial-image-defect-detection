import numpy as np
import pandas as pd

from src.validate_patchcore_results import (
    calculate_metrics,
    check_confusion_matrix_schema,
    check_error_table,
    check_label_mapping,
    check_portable_paths,
    check_source_report,
)


def example_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_path": [
                (
                    "data/raw/mvtec_ad/bottle/"
                    "test/good/000.png"
                ),
                (
                    "data/raw/mvtec_ad/bottle/"
                    "test/contamination/000.png"
                ),
                (
                    "data/raw/mvtec_ad/bottle/"
                    "test/broken_small/000.png"
                ),
            ],
            "file_name": [
                "000.png",
                "000.png",
                "000.png",
            ],
            "defect_type": [
                "good",
                "contamination",
                "broken_small",
            ],
            "true_label": [
                "normal",
                "defective",
                "defective",
            ],
            "true_numeric": [
                0,
                1,
                1,
            ],
            "predicted_label": [
                "normal",
                "normal",
                "defective",
            ],
            "predicted_numeric": [
                0,
                0,
                1,
            ],
            "pred_score": [
                0.2,
                0.4,
                0.9,
            ],
            "correct": [
                True,
                False,
                True,
            ],
            "has_anomaly_map": [
                True,
                True,
                True,
            ],
        }
    )


def test_label_mapping_and_error_table() -> None:
    predictions = example_predictions()

    errors = predictions.loc[
        predictions["correct"] == False  # noqa: E712
    ].copy()

    assert (
        check_label_mapping(
            predictions
        ).passed
        is True
    )

    assert (
        check_error_table(
            predictions,
            errors,
        ).passed
        is True
    )


def test_source_report_consistency() -> None:
    predictions = example_predictions()
    metrics = calculate_metrics(
        predictions
    )

    report = {
        "n_test_images": 3,
        "n_normal_images": 1,
        "n_defective_images": 2,
        "false_positives": 0,
        "false_negatives": 1,
        "n_errors": 1,
        **metrics,
    }

    result = check_source_report(
        predictions,
        report,
        metrics,
    )

    assert result.passed is True


def test_matrix_schema_and_portable_paths() -> None:
    predictions = example_predictions()

    matrix = pd.DataFrame(
        np.array(
            [
                [1, 0],
                [1, 1],
            ]
        ),
        index=[
            "true_normal",
            "true_defective",
        ],
        columns=[
            "pred_normal",
            "pred_defective",
        ],
    )

    assert (
        check_confusion_matrix_schema(
            matrix
        ).passed
        is True
    )

    report = {
        "checkpoint_path": (
            "results/patchcore_anomalib/"
            "model.ckpt"
        )
    }

    assert (
        check_portable_paths(
            predictions,
            report,
        ).passed
        is True
    )


def test_absolute_checkpoint_path_fails() -> None:
    predictions = example_predictions()

    report = {
        "checkpoint_path": (
            r"C:\Users\admin\project\model.ckpt"
        )
    }

    result = check_portable_paths(
        predictions,
        report,
    )

    assert result.passed is False
