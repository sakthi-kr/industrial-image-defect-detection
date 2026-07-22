import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


try:
    # Works when running as: python -m src.train
    from src.data_loader import load_image_records
    from src.features import build_feature_table
    from src.preprocess import DEFAULT_IMAGE_SIZE
except ModuleNotFoundError:
    # Works when running directly from PyCharm: Run train.py
    from data_loader import load_image_records
    from features import build_feature_table
    from preprocess import DEFAULT_IMAGE_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_TABLE_PATH = RESULTS_DIR / "image_feature_table.csv"
MODEL_PATH = MODELS_DIR / "baseline_random_forest_image_classifier.joblib"

RANDOM_STATE = 42


METADATA_COLUMNS = {
    "image_path",
    "category",
    "split",
    "defect_type",
    "binary_label",
    "is_defective",
    "has_mask",
}


def load_or_create_feature_table() -> pd.DataFrame:
    """
    Load the saved feature table if it exists.
    Otherwise build it from the local MVTec AD bottle images.
    """
    if FEATURE_TABLE_PATH.exists():
        print(f"Loading feature table from: {FEATURE_TABLE_PATH}")
        return pd.read_csv(FEATURE_TABLE_PATH)

    print("Feature table not found. Building feature table from images...")
    records = load_image_records()

    feature_table = build_feature_table(
        records=records,
        image_size=DEFAULT_IMAGE_SIZE,
        max_images=None,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(FEATURE_TABLE_PATH, index=False)

    return feature_table


def get_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """
    Select numerical feature columns used for model training.
    """
    feature_columns = [
        column
        for column in feature_table.columns
        if column not in METADATA_COLUMNS
    ]

    # Keep only numeric columns.
    feature_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(feature_table[column])
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found for training.")

    return feature_columns


def train_baseline_classifier(
    feature_table: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Train a Random Forest baseline classifier for normal vs defective images.

    This first version uses a stratified random split across all available bottle
    images. This is useful as a development baseline, but it is not the final
    anomaly-detection validation setup.
    """
    feature_columns = get_feature_columns(feature_table)

    X = feature_table[feature_columns]
    y = feature_table["binary_label"]

    class_counts = y.value_counts()
    print("\nClass counts:")
    print(class_counts)

    if y.nunique() < 2:
        raise ValueError(
            "Need at least two classes for this supervised baseline. "
            "Check that both normal and defective images are present."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )

    return {
        "model": model,
        "feature_columns": feature_columns,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }


def save_training_outputs(training_output: dict) -> None:
    """
    Save the trained model and training metrics.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": training_output["model"],
        "feature_columns": training_output["feature_columns"],
        "random_state": RANDOM_STATE,
        "model_type": "RandomForestClassifier",
        "task": "binary_image_defect_classification",
        "labels": ["normal", "defective"],
    }

    joblib.dump(model_bundle, MODEL_PATH)

    metrics = {
        "model_type": "RandomForestClassifier",
        "task": "binary_image_defect_classification",
        "accuracy": training_output["accuracy"],
        "n_train_samples": len(training_output["X_train"]),
        "n_test_samples": len(training_output["X_test"]),
        "feature_columns": training_output["feature_columns"],
        "classification_report": training_output["classification_report_dict"],
        "validation_note": (
            "This first baseline uses a stratified random split over the MVTec AD "
            "bottle images. It is useful for pipeline development, but it is not "
            "the final anomaly-detection validation setup. A more realistic version "
            "should train only on normal training images and evaluate on unseen normal "
            "and defective test images."
        ),
    }

    metrics_path = RESULTS_DIR / "baseline_metrics.json"
    report_path = RESULTS_DIR / "baseline_classification_report.txt"
    feature_columns_path = RESULTS_DIR / "baseline_feature_columns.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(training_output["classification_report_text"])

    with open(feature_columns_path, "w", encoding="utf-8") as file:
        json.dump(training_output["feature_columns"], file, indent=4)

    print("\nSaved training outputs:")
    print(f"  Model                 : {MODEL_PATH}")
    print(f"  Metrics               : {metrics_path}")
    print(f"  Classification report : {report_path}")
    print(f"  Feature columns       : {feature_columns_path}")


def main() -> None:
    print("Loading image feature table...")
    feature_table = load_or_create_feature_table()

    print(f"Feature table shape: {feature_table.shape}")

    print("\nTraining baseline Random Forest classifier...")
    training_output = train_baseline_classifier(feature_table)

    print("\nTraining complete.")
    print(f"Accuracy: {training_output['accuracy']:.4f}")

    print("\nClassification report:")
    print(training_output["classification_report_text"])

    save_training_outputs(training_output)


if __name__ == "__main__":
    main()
