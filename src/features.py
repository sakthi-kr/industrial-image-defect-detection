from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


try:
    # Works when running as: python -m src.features
    from src.data_loader import ImageRecord, load_image_records, records_to_dataframe
    from src.preprocess import DEFAULT_IMAGE_SIZE, preprocess_image
except ModuleNotFoundError:
    # Works when running directly from PyCharm: Run features.py
    from data_loader import ImageRecord, load_image_records, records_to_dataframe
    from preprocess import DEFAULT_IMAGE_SIZE, preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def extract_rgb_statistics(rgb_array: np.ndarray) -> dict:
    """
    Extract simple RGB channel statistics.

    Input:
    - rgb_array with shape (height, width, 3), normalized to [0, 1]
    """
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Expected RGB array with shape (H, W, 3), got {rgb_array.shape}")

    channel_names = ["red", "green", "blue"]
    features = {}

    for channel_index, channel_name in enumerate(channel_names):
        channel = rgb_array[:, :, channel_index]

        features[f"{channel_name}_mean"] = float(np.mean(channel))
        features[f"{channel_name}_std"] = float(np.std(channel))
        features[f"{channel_name}_min"] = float(np.min(channel))
        features[f"{channel_name}_max"] = float(np.max(channel))

    return features


def extract_grayscale_statistics(grayscale_array: np.ndarray) -> dict:
    """
    Extract grayscale intensity statistics.

    Input:
    - grayscale_array with shape (height, width), normalized to [0, 1]
    """
    if grayscale_array.ndim != 2:
        raise ValueError(f"Expected grayscale array with shape (H, W), got {grayscale_array.shape}")

    percentiles = np.percentile(grayscale_array, [5, 25, 50, 75, 95])

    histogram, _ = np.histogram(
        grayscale_array,
        bins=32,
        range=(0.0, 1.0),
        density=False,
    )

    histogram = histogram.astype(float)
    histogram_probability = histogram / histogram.sum()

    nonzero_probability = histogram_probability[histogram_probability > 0]
    entropy = -np.sum(nonzero_probability * np.log2(nonzero_probability))

    return {
        "gray_mean": float(np.mean(grayscale_array)),
        "gray_std": float(np.std(grayscale_array)),
        "gray_min": float(np.min(grayscale_array)),
        "gray_max": float(np.max(grayscale_array)),
        "gray_p05": float(percentiles[0]),
        "gray_p25": float(percentiles[1]),
        "gray_p50": float(percentiles[2]),
        "gray_p75": float(percentiles[3]),
        "gray_p95": float(percentiles[4]),
        "gray_entropy": float(entropy),
    }


def extract_hsv_statistics(rgb_array: np.ndarray) -> dict:
    """
    Extract simple HSV colour statistics.

    OpenCV expects uint8 image values in [0, 255], so the normalized RGB image
    is converted back to uint8 before RGB -> HSV conversion.
    """
    rgb_uint8 = np.clip(rgb_array * 255.0, 0, 255).astype(np.uint8)

    hsv_array = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Normalize HSV channels approximately:
    # H in OpenCV is [0, 179], S and V are [0, 255]
    hsv_array[:, :, 0] = hsv_array[:, :, 0] / 179.0
    hsv_array[:, :, 1] = hsv_array[:, :, 1] / 255.0
    hsv_array[:, :, 2] = hsv_array[:, :, 2] / 255.0

    channel_names = ["hue", "saturation", "value"]
    features = {}

    for channel_index, channel_name in enumerate(channel_names):
        channel = hsv_array[:, :, channel_index]

        features[f"{channel_name}_mean"] = float(np.mean(channel))
        features[f"{channel_name}_std"] = float(np.std(channel))

    return features


def extract_edge_features(grayscale_array: np.ndarray) -> dict:
    """
    Extract simple edge and texture-like features using Sobel and Laplacian filters.
    """
    gray_uint8 = np.clip(grayscale_array * 255.0, 0, 255).astype(np.uint8)

    sobel_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)

    # Simple high-gradient threshold. This is not perfect, but it is useful
    # as a baseline edge-density feature.
    edge_density = np.mean(gradient_magnitude > 50.0)

    return {
        "gradient_mean": float(np.mean(gradient_magnitude)),
        "gradient_std": float(np.std(gradient_magnitude)),
        "gradient_max": float(np.max(gradient_magnitude)),
        "edge_density": float(edge_density),
        "laplacian_var": float(np.var(laplacian)),
    }


def extract_histogram_features(
    grayscale_array: np.ndarray,
    n_bins: int = 16,
) -> dict:
    """
    Extract normalized grayscale histogram features.
    """
    histogram, _ = np.histogram(
        grayscale_array,
        bins=n_bins,
        range=(0.0, 1.0),
        density=False,
    )

    histogram = histogram.astype(float)
    histogram = histogram / histogram.sum()

    features = {
        f"gray_hist_bin_{index:02d}": float(value)
        for index, value in enumerate(histogram)
    }

    return features


def extract_features_from_image(
    image_path: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict:
    """
    Extract all baseline features from one image.
    """
    processed = preprocess_image(
        image_path=image_path,
        image_size=image_size,
    )

    rgb_array = processed["rgb_array"]
    grayscale_array = processed["grayscale_array"]

    features = {}
    features.update(extract_rgb_statistics(rgb_array))
    features.update(extract_grayscale_statistics(grayscale_array))
    features.update(extract_hsv_statistics(rgb_array))
    features.update(extract_edge_features(grayscale_array))
    features.update(extract_histogram_features(grayscale_array, n_bins=16))

    return features


def build_feature_table(
    records: list[ImageRecord],
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    max_images: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a feature table from image records.

    Each row = one image.
    Columns = extracted features + metadata + label.
    """
    rows = []

    if max_images is not None:
        records = records[:max_images]

    for index, record in enumerate(records):
        features = extract_features_from_image(
            image_path=record.image_path,
            image_size=image_size,
        )

        features["image_path"] = str(record.image_path)
        features["category"] = record.category
        features["split"] = record.split
        features["defect_type"] = record.defect_type
        features["binary_label"] = record.binary_label
        features["is_defective"] = record.is_defective
        features["has_mask"] = record.mask_path is not None

        rows.append(features)

        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1} images...")

    feature_table = pd.DataFrame(rows)

    if feature_table.empty:
        raise ValueError("Feature table is empty. Check the input records.")

    numeric_columns = feature_table.select_dtypes(include="number").columns

    if not np.isfinite(feature_table[numeric_columns].to_numpy()).all():
        raise ValueError("Feature table contains NaN or infinite values.")

    return feature_table


def save_feature_outputs(feature_table: pd.DataFrame) -> None:
    """
    Save feature table and compact summary files.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_table_path = RESULTS_DIR / "image_feature_table.csv"
    feature_preview_path = RESULTS_DIR / "image_feature_preview.csv"
    feature_summary_path = RESULTS_DIR / "image_feature_summary.csv"

    feature_table.to_csv(feature_table_path, index=False)
    feature_table.head(20).to_csv(feature_preview_path, index=False)

    numeric_columns = feature_table.select_dtypes(include="number").columns
    feature_summary = feature_table[numeric_columns].describe().T
    feature_summary.to_csv(feature_summary_path)

    print("\nSaved feature outputs:")
    print(f"  Full feature table : {feature_table_path}")
    print(f"  Feature preview    : {feature_preview_path}")
    print(f"  Feature summary    : {feature_summary_path}")


def print_feature_table_summary(feature_table: pd.DataFrame) -> None:
    """
    Print a readable summary of the feature table.
    """
    print("\nImage feature table created successfully.")
    print(f"Shape: {feature_table.shape}")

    print("\nCounts by split:")
    print(feature_table["split"].value_counts())

    print("\nCounts by binary label:")
    print(feature_table["binary_label"].value_counts())

    print("\nCounts by defect type:")
    print(feature_table["defect_type"].value_counts())

    print("\nFirst few columns:")
    for column in list(feature_table.columns[:20]):
        print(f"  - {column}")


def main() -> None:
    records = load_image_records()

    # For the first baseline, process all images in the bottle category.
    feature_table = build_feature_table(
        records=records,
        image_size=DEFAULT_IMAGE_SIZE,
        max_images=None,
    )

    print_feature_table_summary(feature_table)
    save_feature_outputs(feature_table)


if __name__ == "__main__":
    main()
