import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.compare_patchcore_categories import (
    build_comparison_table,
    extract_metrics,
    load_category_summary,
    normalize_key,
    validate_fixed_configuration,
)


def write_summary(
    root: Path,
    category: str,
    *,
    backbone: str = "resnet18",
) -> None:
    path = root / category / "run_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": "Patchcore",
        "category": category,
        "backbone": backbone,
        "layers": ["layer2", "layer3"],
        "image_size": [224, 224],
        "coreset_sampling_ratio": 0.01,
        "num_neighbors": 5,
        "accelerator": "cpu",
        "elapsed_seconds": 12.5,
        "test_results": [
            {
                "image_AUROC": 0.99,
                "image_F1Score": 0.95,
                "pixel_AUROC": 0.91,
                "pixel_F1Score": 0.62,
            }
        ],
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_normalize_key_removes_case_and_separators() -> None:
    assert normalize_key("Image_F1Score") == "imagef1score"


def test_extract_metrics_supports_anomalib_names() -> None:
    metrics = extract_metrics(
        [
            {
                "image_AUROC": 0.98,
                "image_F1Score": 0.94,
                "pixel_AUROC": 0.90,
                "pixel_F1Score": 0.61,
            }
        ]
    )

    assert metrics == {
        "image_auroc": 0.98,
        "image_f1": 0.94,
        "pixel_auroc": 0.90,
        "pixel_f1": 0.61,
    }


def test_load_category_summary_reads_metrics(
    tmp_path: Path,
) -> None:
    write_summary(tmp_path, "bottle")

    summary = load_category_summary(
        "bottle",
        results_root=tmp_path,
    )

    assert summary["category"] == "bottle"
    assert summary["image_auroc"] == 0.99
    assert summary["pixel_f1"] == 0.62


def test_build_comparison_table_has_two_rows(
    tmp_path: Path,
) -> None:
    write_summary(tmp_path, "bottle")
    write_summary(tmp_path, "cable")

    table = build_comparison_table(
        ["bottle", "cable"],
        results_root=tmp_path,
    )

    assert isinstance(table, pd.DataFrame)
    assert table["category"].tolist() == [
        "bottle",
        "cable",
    ]
    assert table["image_f1"].tolist() == [
        0.95,
        0.95,
    ]


def test_configuration_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    write_summary(tmp_path, "bottle")
    write_summary(
        tmp_path,
        "cable",
        backbone="wide_resnet50_2",
    )

    rows = [
        load_category_summary(
            category,
            results_root=tmp_path,
        )
        for category in ("bottle", "cable")
    ]

    with pytest.raises(
        ValueError,
        match="configuration mismatch",
    ):
        validate_fixed_configuration(rows)
