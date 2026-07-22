import numpy as np
import pandas as pd

from src.data_loader import load_image_records
from src.features import (
    build_feature_table,
    extract_edge_features,
    extract_features_from_image,
    extract_grayscale_statistics,
    extract_histogram_features,
    extract_rgb_statistics,
)
from src.preprocess import preprocess_image


def get_first_image_path():
    records = load_image_records()
    return records[0].image_path


def test_extract_rgb_statistics():
    image_path = get_first_image_path()
    processed = preprocess_image(image_path)

    features = extract_rgb_statistics(processed["rgb_array"])

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

    assert expected.issubset(features.keys())

    for value in features.values():
        assert np.isfinite(value)


def test_extract_grayscale_statistics():
    image_path = get_first_image_path()
    processed = preprocess_image(image_path)

    features = extract_grayscale_statistics(processed["grayscale_array"])

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

    assert expected.issubset(features.keys())

    for value in features.values():
        assert np.isfinite(value)


def test_extract_edge_features():
    image_path = get_first_image_path()
    processed = preprocess_image(image_path)

    features = extract_edge_features(processed["grayscale_array"])

    expected = {
        "gradient_mean",
        "gradient_std",
        "gradient_max",
        "edge_density",
        "laplacian_var",
    }

    assert expected.issubset(features.keys())

    for value in features.values():
        assert np.isfinite(value)


def test_extract_histogram_features():
    image_path = get_first_image_path()
    processed = preprocess_image(image_path)

    features = extract_histogram_features(
        processed["grayscale_array"],
        n_bins=16,
    )

    assert len(features) == 16
    assert abs(sum(features.values()) - 1.0) < 1e-6


def test_extract_features_from_image():
    image_path = get_first_image_path()

    features = extract_features_from_image(image_path)

    assert isinstance(features, dict)
    assert len(features) > 20

    for value in features.values():
        assert np.isfinite(value)


def test_build_feature_table_small_subset():
    records = load_image_records()[:10]

    feature_table = build_feature_table(
        records=records,
        max_images=10,
    )

    assert isinstance(feature_table, pd.DataFrame)
    assert not feature_table.empty
    assert len(feature_table) == 10

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

    assert expected_columns.issubset(feature_table.columns)

    numeric_columns = feature_table.select_dtypes(include="number").columns
    assert np.isfinite(feature_table[numeric_columns].to_numpy()).all()
