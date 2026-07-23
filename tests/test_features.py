from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    load_image_records,
)
from src.features import (
    build_feature_table,
    extract_edge_features,
    extract_features_from_image,
    extract_grayscale_statistics,
    extract_histogram_features,
    extract_rgb_statistics,
)
from src.preprocess import preprocess_image


def test_extract_rgb_statistics(
    synthetic_image_path: Path,
) -> None:
    processed = preprocess_image(
        synthetic_image_path
    )

    features = extract_rgb_statistics(
        processed["rgb_array"]
    )

    expected = {
        "red_mean",
        "red_std",
        "red_min",
        "red_max",
        "green_mean",
        "green_std",
        "green_min",
        "green_max",
        "blue_mean",
        "blue_std",
        "blue_min",
        "blue_max",
    }

    assert expected.issubset(
        features.keys()
    )

    assert all(
        np.isfinite(value)
        for value in features.values()
    )


def test_extract_grayscale_statistics(
    synthetic_image_path: Path,
) -> None:
    processed = preprocess_image(
        synthetic_image_path
    )

    features = extract_grayscale_statistics(
        processed["grayscale_array"]
    )

    expected = {
        "gray_mean",
        "gray_std",
        "gray_min",
        "gray_max",
        "gray_p05",
        "gray_p25",
        "gray_p50",
        "gray_p75",
        "gray_p95",
        "gray_entropy",
    }

    assert expected.issubset(
        features.keys()
    )

    assert all(
        np.isfinite(value)
        for value in features.values()
    )


def test_extract_edge_features(
    synthetic_image_path: Path,
) -> None:
    processed = preprocess_image(
        synthetic_image_path
    )

    features = extract_edge_features(
        processed["grayscale_array"]
    )

    expected = {
        "gradient_mean",
        "gradient_std",
        "gradient_max",
        "edge_density",
        "laplacian_var",
    }

    assert expected.issubset(
        features.keys()
    )

    assert all(
        np.isfinite(value)
        for value in features.values()
    )


def test_extract_histogram_features(
    synthetic_image_path: Path,
) -> None:
    processed = preprocess_image(
        synthetic_image_path
    )

    features = extract_histogram_features(
        processed["grayscale_array"],
        n_bins=16,
    )

    assert len(features) == 16

    assert abs(
        sum(features.values()) - 1.0
    ) < 1e-6


def test_extract_features_from_image(
    synthetic_image_path: Path,
) -> None:
    features = extract_features_from_image(
        synthetic_image_path
    )

    assert isinstance(
        features,
        dict,
    )

    assert len(features) > 20

    assert all(
        np.isfinite(value)
        for value in features.values()
    )


def test_build_feature_table(
    synthetic_mvtec_root: Path,
) -> None:
    records = load_image_records(
        data_root=synthetic_mvtec_root,
        category="bottle",
    )

    feature_table = build_feature_table(
        records=records,
    )

    assert isinstance(
        feature_table,
        pd.DataFrame,
    )

    assert len(feature_table) == 5

    expected_columns = {
        "gray_mean",
        "gray_std",
        "gradient_mean",
        "edge_density",
        "binary_label",
        "split",
        "defect_type",
        "image_path",
    }

    assert expected_columns.issubset(
        feature_table.columns
    )

    assert set(
        feature_table[
            "binary_label"
        ].unique()
    ) == {
        "normal",
        "defective",
    }

    numeric_columns = (
        feature_table
        .select_dtypes(include="number")
        .columns
    )

    assert np.isfinite(
        feature_table[
            numeric_columns
        ].to_numpy()
    ).all()
