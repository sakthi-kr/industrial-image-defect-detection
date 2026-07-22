from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from anomalib.engine import Engine
from anomalib.models import Patchcore
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATEGORY_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mvtec_ad"
    / "bottle"
)

TEST_ROOT = CATEGORY_ROOT / "test"
GROUND_TRUTH_ROOT = CATEGORY_ROOT / "ground_truth"

ANOMALIB_OUTPUT_DIR = PROJECT_ROOT / "results" / "patchcore_anomalib"
METRICS_PATH = PROJECT_ROOT / "results" / "patchcore_smoke_metrics.json"

PREDICTIONS_CSV_PATH = (
    PROJECT_ROOT / "results" / "patchcore_predictions.csv"
)
ERROR_ANALYSIS_CSV_PATH = (
    PROJECT_ROOT / "results" / "patchcore_error_analysis.csv"
)
REPORT_JSON_PATH = (
    PROJECT_ROOT / "results" / "patchcore_report.json"
)
CONFUSION_MATRIX_CSV_PATH = (
    PROJECT_ROOT / "results" / "patchcore_confusion_matrix.csv"
)
HEATMAP_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "patchcore_example_heatmaps.png"
)
SCORE_DISTRIBUTION_PATH = (
    PROJECT_ROOT / "results" / "patchcore_score_distribution.png"
)


def find_checkpoint() -> Path:
    """
    Locate the checkpoint created by train_patchcore.py.
    """
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        saved_path = metrics.get("best_model_path")

        if saved_path:
            checkpoint_path = Path(saved_path)

            if checkpoint_path.exists():
                return checkpoint_path

    candidates = list(ANOMALIB_OUTPUT_DIR.rglob("model.ckpt"))

    if not candidates:
        raise FileNotFoundError(
            "PatchCore checkpoint not found.\n"
            "Run src/train_patchcore.py first."
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def create_patchcore_model() -> Patchcore:
    """
    Recreate the same PatchCore configuration used during training.
    """
    pre_processor = Patchcore.configure_pre_processor(
        image_size=(224, 224),
        center_crop_size=(224, 224),
    )

    return Patchcore(
        backbone="resnet18",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.01,
        num_neighbors=5,
        pre_processor=pre_processor,
        visualizer=False,
    )


def flatten_prediction_containers(
    predictions: Iterable[Any],
) -> list[Any]:
    """
    Flatten nested prediction lists returned by Engine.predict().
    """
    flattened = []

    for prediction in predictions:
        if isinstance(prediction, list):
            flattened.extend(
                flatten_prediction_containers(prediction)
            )
        else:
            flattened.append(prediction)

    return flattened


def path_list(value: Any) -> list[Path]:
    """
    Convert an image_path field into a list of Path objects.
    """
    if isinstance(value, (str, Path)):
        return [Path(value)]

    if isinstance(value, (list, tuple)):
        return [Path(item) for item in value]

    if isinstance(value, np.ndarray):
        return [Path(item) for item in value.tolist()]

    raise TypeError(
        f"Unsupported image_path type: {type(value)}"
    )


def split_batch_value(
    value: Any,
    batch_size: int,
) -> list[Any]:
    """
    Split a tensor, array, sequence, or scalar into one value per image.
    """
    if value is None:
        return [None] * batch_size

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()

        if batch_size == 1:
            if value.ndim > 0 and value.shape[0] == 1:
                return [value[0]]
            return [value]

        if value.ndim > 0 and value.shape[0] == batch_size:
            return [value[index] for index in range(batch_size)]

        if value.numel() == batch_size:
            flattened = value.reshape(-1)
            return [
                flattened[index]
                for index in range(batch_size)
            ]

        raise ValueError(
            "Cannot split tensor with shape "
            f"{tuple(value.shape)} into {batch_size} samples."
        )

    if isinstance(value, np.ndarray):
        if batch_size == 1:
            if value.ndim > 0 and value.shape[0] == 1:
                return [value[0]]
            return [value]

        if value.ndim > 0 and value.shape[0] == batch_size:
            return [value[index] for index in range(batch_size)]

        if value.size == batch_size:
            flattened = value.reshape(-1)
            return [
                flattened[index]
                for index in range(batch_size)
            ]

    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return list(value)

        if batch_size == 1:
            return [value]

    return [value] * batch_size


def scalar_value(value: Any) -> float | int | None:
    """
    Convert a tensor, array, or Python scalar into one scalar.
    """
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().squeeze()

        if value.numel() != 1:
            raise ValueError(
                "Expected scalar tensor, got shape "
                f"{tuple(value.shape)}"
            )

        return value.item()

    if isinstance(value, np.ndarray):
        value = np.asarray(value).squeeze()

        if value.size != 1:
            raise ValueError(
                f"Expected scalar array, got shape {value.shape}"
            )

        return value.item()

    if isinstance(value, np.generic):
        return value.item()

    return value


def map_to_numpy(value: Any) -> np.ndarray | None:
    """
    Convert an anomaly map to a 2D NumPy array.
    """
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(
            f"Expected 2D anomaly map, got shape {array.shape}"
        )

    return array.astype(np.float32)


def infer_truth(image_path: Path) -> tuple[str, int, str]:
    """
    Infer true label and defect type from MVTec folder structure.
    """
    defect_type = image_path.parent.name
    is_defective = int(defect_type != "good")
    true_label = "defective" if is_defective else "normal"

    return true_label, is_defective, defect_type


def relative_image_path(image_path: Path) -> str:
    """
    Store a portable path relative to the project root where possible.
    """
    try:
        return str(image_path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(image_path)


def extract_prediction_rows(
    predictions: list[Any],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Convert Anomalib predictions into a dataframe and anomaly-map lookup.
    """
    rows = []
    anomaly_maps = {}

    containers = flatten_prediction_containers(predictions)

    for container in containers:
        if not hasattr(container, "image_path"):
            raise AttributeError(
                "Prediction object has no image_path attribute."
            )

        image_paths = path_list(container.image_path)
        batch_size = len(image_paths)

        scores = split_batch_value(
            getattr(container, "pred_score", None),
            batch_size,
        )
        labels = split_batch_value(
            getattr(container, "pred_label", None),
            batch_size,
        )
        maps = split_batch_value(
            getattr(container, "anomaly_map", None),
            batch_size,
        )

        for index, image_path in enumerate(image_paths):
            image_path = image_path.resolve()

            pred_score = scalar_value(scores[index])
            pred_numeric = scalar_value(labels[index])

            if pred_numeric is None:
                raise ValueError(
                    f"No predicted label returned for {image_path}"
                )

            pred_numeric = int(bool(pred_numeric))
            predicted_label = (
                "defective" if pred_numeric == 1 else "normal"
            )

            true_label, true_numeric, defect_type = infer_truth(
                image_path
            )

            correct = pred_numeric == true_numeric

            anomaly_map = map_to_numpy(maps[index])

            map_key = str(image_path)

            if anomaly_map is not None:
                anomaly_maps[map_key] = anomaly_map

            rows.append(
                {
                    "image_path": relative_image_path(image_path),
                    "file_name": image_path.name,
                    "defect_type": defect_type,
                    "true_label": true_label,
                    "true_numeric": true_numeric,
                    "predicted_label": predicted_label,
                    "predicted_numeric": pred_numeric,
                    "pred_score": (
                        float(pred_score)
                        if pred_score is not None
                        else None
                    ),
                    "correct": correct,
                    "has_anomaly_map": anomaly_map is not None,
                    "absolute_path": str(image_path),
                }
            )

    prediction_table = pd.DataFrame(rows)

    if prediction_table.empty:
        raise ValueError("No prediction rows were created.")

    return prediction_table, anomaly_maps


def save_prediction_tables(
    prediction_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save full prediction and error-analysis tables.
    """
    public_columns = [
        column
        for column in prediction_table.columns
        if column != "absolute_path"
    ]

    prediction_table[public_columns].to_csv(
        PREDICTIONS_CSV_PATH,
        index=False,
    )

    errors = prediction_table[
        prediction_table["correct"] == False  # noqa: E712
    ].copy()

    errors[public_columns].to_csv(
        ERROR_ANALYSIS_CSV_PATH,
        index=False,
    )

    return errors


def save_confusion_matrix_table(
    prediction_table: pd.DataFrame,
) -> np.ndarray:
    """
    Save image-level confusion matrix as CSV.
    """
    labels = [0, 1]

    matrix = confusion_matrix(
        prediction_table["true_numeric"],
        prediction_table["predicted_numeric"],
        labels=labels,
    )

    matrix_table = pd.DataFrame(
        matrix,
        index=["true_normal", "true_defective"],
        columns=["pred_normal", "pred_defective"],
    )

    matrix_table.to_csv(CONFUSION_MATRIX_CSV_PATH)

    return matrix


def save_report(
    prediction_table: pd.DataFrame,
    errors: pd.DataFrame,
    checkpoint_path: Path,
) -> None:
    """
    Save compact image-level performance report.
    """
    y_true = prediction_table["true_numeric"]
    y_pred = prediction_table["predicted_numeric"]

    normal_scores = prediction_table.loc[
        prediction_table["true_numeric"] == 0,
        "pred_score",
    ]

    defective_scores = prediction_table.loc[
        prediction_table["true_numeric"] == 1,
        "pred_score",
    ]

    false_positives = int(
        (
            (prediction_table["true_numeric"] == 0)
            & (prediction_table["predicted_numeric"] == 1)
        ).sum()
    )

    false_negatives = int(
        (
            (prediction_table["true_numeric"] == 1)
            & (prediction_table["predicted_numeric"] == 0)
        ).sum()
    )

    report = {
        "model": "Patchcore",
        "category": "bottle",
        "backbone": "resnet18",
        "image_size": [224, 224],
        "coreset_sampling_ratio": 0.01,
        "num_neighbors": 5,
        "accelerator": "cpu",
        "checkpoint_path": relative_image_path(checkpoint_path),
        "n_test_images": int(len(prediction_table)),
        "n_normal_images": int((y_true == 0).sum()),
        "n_defective_images": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
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
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "n_errors": int(len(errors)),
        "normal_score_statistics": {
            "minimum": float(normal_scores.min()),
            "mean": float(normal_scores.mean()),
            "maximum": float(normal_scores.max()),
        },
        "defective_score_statistics": {
            "minimum": float(defective_scores.min()),
            "mean": float(defective_scores.mean()),
            "maximum": float(defective_scores.max()),
        },
        "interpretation": (
            "Image-level predictions were generated using a PatchCore "
            "model trained only on normal MVTec AD bottle images. "
            "Anomaly heatmaps indicate spatial regions that differ from "
            "the normal-image feature memory bank."
        ),
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)


def normalize_anomaly_map(
    anomaly_map: np.ndarray,
) -> np.ndarray:
    """
    Normalize one anomaly map to [0, 1] for visualization.
    """
    minimum = float(np.min(anomaly_map))
    maximum = float(np.max(anomaly_map))

    if maximum <= minimum:
        return np.zeros_like(anomaly_map, dtype=np.float32)

    return (
        (anomaly_map - minimum)
        / (maximum - minimum)
    ).astype(np.float32)


def resize_anomaly_map(
    anomaly_map: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """
    Resize normalized anomaly map to the original image size.
    """
    normalized = normalize_anomaly_map(anomaly_map)

    map_image = Image.fromarray(
        np.uint8(np.clip(normalized, 0.0, 1.0) * 255)
    )

    resized = map_image.resize(
        image_size,
        Image.Resampling.BILINEAR,
    )

    return np.asarray(resized, dtype=np.float32) / 255.0


def load_ground_truth_mask(
    image_path: Path,
    defect_type: str,
    image_size: tuple[int, int],
) -> np.ndarray:
    """
    Load MVTec ground-truth mask or return a blank mask for normal images.
    """
    if defect_type == "good":
        return np.zeros(
            (image_size[1], image_size[0]),
            dtype=np.float32,
        )

    mask_path = (
        GROUND_TRUTH_ROOT
        / defect_type
        / f"{image_path.stem}_mask.png"
    )

    if not mask_path.exists():
        return np.zeros(
            (image_size[1], image_size[0]),
            dtype=np.float32,
        )

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(
        image_size,
        Image.Resampling.NEAREST,
    )

    return np.asarray(mask, dtype=np.float32) / 255.0


def select_visualization_rows(
    prediction_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select difficult and representative cases.

    - Two normal images with the highest anomaly scores
    - Lowest-scoring image from each defect type
    """
    selected = []

    normal_rows = (
        prediction_table[
            prediction_table["defect_type"] == "good"
        ]
        .sort_values("pred_score", ascending=False)
        .head(2)
    )

    selected.append(normal_rows)

    for defect_type in [
        "broken_large",
        "broken_small",
        "contamination",
    ]:
        defect_rows = (
            prediction_table[
                prediction_table["defect_type"] == defect_type
            ]
            .sort_values("pred_score", ascending=True)
            .head(1)
        )

        if not defect_rows.empty:
            selected.append(defect_rows)

    return pd.concat(selected, ignore_index=True)


def save_heatmap_figure(
    prediction_table: pd.DataFrame,
    anomaly_maps: dict[str, np.ndarray],
) -> None:
    """
    Save original images, anomaly maps, overlays, and ground-truth masks.
    """
    selected = select_visualization_rows(prediction_table)

    n_rows = len(selected)
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 3.5 * n_rows),
    )

    if n_rows == 1:
        axes = np.asarray([axes])

    for row_index, (_, row) in enumerate(selected.iterrows()):
        image_path = Path(row["absolute_path"])
        image = Image.open(image_path).convert("RGB")

        map_key = str(image_path.resolve())

        if map_key not in anomaly_maps:
            raise KeyError(
                f"Anomaly map missing for: {image_path}"
            )

        resized_map = resize_anomaly_map(
            anomaly_maps[map_key],
            image.size,
        )

        ground_truth = load_ground_truth_mask(
            image_path=image_path,
            defect_type=row["defect_type"],
            image_size=image.size,
        )

        title = (
            f"{row['defect_type']}\n"
            f"true={row['true_label']}, "
            f"pred={row['predicted_label']}, "
            f"score={row['pred_score']:.3f}"
        )

        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(title, fontsize=9)
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(
            resized_map,
            cmap="inferno",
        )
        axes[row_index, 1].set_title("Anomaly heatmap")
        axes[row_index, 1].axis("off")

        axes[row_index, 2].imshow(image)
        axes[row_index, 2].imshow(
            resized_map,
            cmap="inferno",
            alpha=0.45,
        )
        axes[row_index, 2].set_title("Heatmap overlay")
        axes[row_index, 2].axis("off")

        axes[row_index, 3].imshow(
            ground_truth,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row_index, 3].set_title("Ground-truth mask")
        axes[row_index, 3].axis("off")

    fig.suptitle(
        "PatchCore Anomaly Detection — MVTec AD Bottle",
        fontsize=15,
        y=0.995,
    )

    plt.tight_layout()
    fig.savefig(
        HEATMAP_OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_score_distribution(
    prediction_table: pd.DataFrame,
) -> None:
    """
    Save image-level anomaly score distribution by true class.
    """
    normal_scores = prediction_table.loc[
        prediction_table["true_label"] == "normal",
        "pred_score",
    ]

    defective_scores = prediction_table.loc[
        prediction_table["true_label"] == "defective",
        "pred_score",
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot(
        [normal_scores, defective_scores],
        tick_labels=["Normal", "Defective"],
    )

    ax.set_ylabel("PatchCore anomaly score")
    ax.set_title("Image-Level Anomaly Score Distribution")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(
        SCORE_DISTRIBUTION_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    if not TEST_ROOT.exists():
        raise FileNotFoundError(
            f"MVTec test folder not found: {TEST_ROOT}"
        )

    checkpoint_path = find_checkpoint()

    print("PatchCore report generation")
    print("===========================")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Test folder: {TEST_ROOT}")
    print(f"PyTorch    : {torch.__version__}")
    print(f"CUDA       : {torch.cuda.is_available()}")

    model = create_patchcore_model()

    engine = Engine(
        accelerator="cpu",
        devices=1,
        logger=False,
        default_root_dir=ANOMALIB_OUTPUT_DIR,
    )

    print("\nRunning predictions over the full bottle test set...")

    predictions = engine.predict(
        model=model,
        data_path=str(TEST_ROOT),
        ckpt_path=str(checkpoint_path),
        return_predictions=True,
    )

    if predictions is None:
        raise RuntimeError("Engine.predict returned no predictions.")

    prediction_table, anomaly_maps = extract_prediction_rows(
        predictions
    )

    prediction_table = prediction_table.sort_values(
        ["true_numeric", "defect_type", "file_name"]
    ).reset_index(drop=True)

    errors = save_prediction_tables(prediction_table)
    save_confusion_matrix_table(prediction_table)

    save_report(
        prediction_table=prediction_table,
        errors=errors,
        checkpoint_path=checkpoint_path,
    )

    save_heatmap_figure(
        prediction_table=prediction_table,
        anomaly_maps=anomaly_maps,
    )

    save_score_distribution(prediction_table)

    accuracy = accuracy_score(
        prediction_table["true_numeric"],
        prediction_table["predicted_numeric"],
    )

    print("\nPatchCore report completed.")
    print(f"Test images       : {len(prediction_table)}")
    print(f"Correct           : {int(prediction_table['correct'].sum())}")
    print(f"Errors            : {len(errors)}")
    print(f"Accuracy          : {accuracy:.4f}")

    print("\nSaved outputs:")
    print(f"  Predictions      : {PREDICTIONS_CSV_PATH}")
    print(f"  Error analysis   : {ERROR_ANALYSIS_CSV_PATH}")
    print(f"  Report           : {REPORT_JSON_PATH}")
    print(f"  Confusion matrix : {CONFUSION_MATRIX_CSV_PATH}")
    print(f"  Heatmaps         : {HEATMAP_OUTPUT_PATH}")
    print(f"  Score plot       : {SCORE_DISTRIBUTION_PATH}")


if __name__ == "__main__":
    main()