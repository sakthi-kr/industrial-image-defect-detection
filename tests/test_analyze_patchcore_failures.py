from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from scripts.analyze_patchcore_failures import (
    README_START,
    build_analysis_summary,
    build_borderline_cases,
    build_defect_type_summary,
    build_localization_gap,
    read_prediction_table,
    save_failure_case_figure,
    update_readme,
)


def sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_path": "data/raw/mvtec_ad/bottle/test/good/006.png",
                "file_name": "006.png",
                "defect_type": "good",
                "true_label": "normal",
                "true_numeric": 0,
                "predicted_label": "defective",
                "predicted_numeric": 1,
                "pred_score": 0.524,
                "correct": False,
            },
            {
                "image_path": "data/raw/mvtec_ad/bottle/test/good/000.png",
                "file_name": "000.png",
                "defect_type": "good",
                "true_label": "normal",
                "true_numeric": 0,
                "predicted_label": "normal",
                "predicted_numeric": 0,
                "pred_score": 0.42,
                "correct": True,
            },
            {
                "image_path": "data/raw/mvtec_ad/bottle/test/contamination/003.png",
                "file_name": "003.png",
                "defect_type": "contamination",
                "true_label": "defective",
                "true_numeric": 1,
                "predicted_label": "normal",
                "predicted_numeric": 0,
                "pred_score": 0.5,
                "correct": False,
            },
            {
                "image_path": "data/raw/mvtec_ad/bottle/test/broken_large/000.png",
                "file_name": "000.png",
                "defect_type": "broken_large",
                "true_label": "defective",
                "true_numeric": 1,
                "predicted_label": "defective",
                "predicted_numeric": 1,
                "pred_score": 0.9,
                "correct": True,
            },
        ]
    )


def sample_categories() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "category": "bottle",
                "image_auroc": 1.0,
                "image_f1": 0.99,
                "pixel_auroc": 0.97,
                "pixel_f1": 0.65,
            },
            {
                "category": "leather",
                "image_auroc": 1.0,
                "image_f1": 0.995,
                "pixel_auroc": 0.988,
                "pixel_f1": 0.343,
            },
        ]
    )


def test_read_prediction_table_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    pd.DataFrame({"pred_score": [0.5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        read_prediction_table(path)


def test_defect_type_summary_counts_errors() -> None:
    summary = build_defect_type_summary(sample_predictions())
    good = summary.loc[summary["defect_type"] == "good"].iloc[0]
    contamination = summary.loc[summary["defect_type"] == "contamination"].iloc[0]
    assert int(good["false_positives"]) == 1
    assert int(contamination["false_negatives"]) == 1
    assert float(good["correct_rate"]) == 0.5


def test_borderline_cases_are_sorted_by_absolute_margin() -> None:
    borderline = build_borderline_cases(
        sample_predictions(), threshold=0.5, review_margin=0.05
    )
    assert borderline.iloc[0]["defect_type"] == "contamination"
    assert bool(borderline.iloc[0]["inside_review_band"])
    assert borderline["absolute_margin"].is_monotonic_increasing


def test_localization_gap_identifies_leather() -> None:
    gaps = build_localization_gap(sample_categories())
    assert gaps.iloc[0]["category"] == "leather"
    assert gaps.iloc[0]["image_pixel_f1_gap"] == pytest.approx(0.652)


def test_analysis_summary_lists_fp_fn_and_localization() -> None:
    predictions = sample_predictions()
    defects = build_defect_type_summary(predictions)
    borderline = build_borderline_cases(
        predictions, threshold=0.5, review_margin=0.05
    )
    gaps = build_localization_gap(sample_categories())
    summary = build_analysis_summary(
        predictions,
        borderline,
        defects,
        gaps,
        threshold=0.5,
        review_margin=0.05,
    )
    assert summary["false_positives"] == 1
    assert summary["false_negatives"] == 1
    assert summary["false_positive_cases"][0]["defect_type"] == "good"
    assert summary["false_negative_cases"][0]["defect_type"] == "contamination"
    assert summary["localization"]["lowest_pixel_f1_category"] == "leather"


def test_update_readme_is_idempotent(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Project\n", encoding="utf-8")
    summary = {
        "false_positives": 1,
        "false_negatives": 1,
        "illustrative_review_margin": 0.05,
        "localization": {
            "lowest_pixel_f1_category": "leather",
            "lowest_pixel_f1": 0.343,
        },
    }
    update_readme(readme, summary)
    first = readme.read_text(encoding="utf-8")
    update_readme(readme, summary)
    second = readme.read_text(encoding="utf-8")
    assert first == second
    assert second.count(README_START) == 1


def test_failure_figure_uses_error_images_and_mask(tmp_path: Path) -> None:
    project_root = tmp_path
    good = project_root / "data/raw/mvtec_ad/bottle/test/good/006.png"
    contamination = (
        project_root / "data/raw/mvtec_ad/bottle/test/contamination/003.png"
    )
    mask = (
        project_root
        / "data/raw/mvtec_ad/bottle/ground_truth/contamination/003_mask.png"
    )
    for path in [good, contamination, mask]:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB" if path != mask else "L", (16, 16), color=128).save(path)

    output = tmp_path / "failure.png"
    save_failure_case_figure(
        sample_predictions(),
        output_path=output,
        project_root=project_root,
        threshold=0.5,
    )
    assert output.exists()
    assert output.stat().st_size > 0
