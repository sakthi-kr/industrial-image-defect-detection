import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd


try:
    # Works when running as: python -m src.predict
    from src.data_loader import DEFAULT_DATA_ROOT, DEFAULT_CATEGORY
    from src.features import extract_features_from_image
    from src.preprocess import DEFAULT_IMAGE_SIZE
    from src.train import MODEL_PATH
except ModuleNotFoundError:
    # Works when running directly from PyCharm: Run predict.py
    from data_loader import DEFAULT_DATA_ROOT, DEFAULT_CATEGORY
    from features import extract_features_from_image
    from preprocess import DEFAULT_IMAGE_SIZE
    from train import MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_IMAGE_PATH = (
    DEFAULT_DATA_ROOT
    / DEFAULT_CATEGORY
    / "test"
    / "good"
    / "000.png"
)


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict:
    """
    Load the saved image-classification model bundle.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run `python src/train.py` first."
        )

    model_bundle = joblib.load(model_path)

    required_keys = {"model", "feature_columns"}
    missing_keys = required_keys - set(model_bundle.keys())

    if missing_keys:
        raise KeyError(
            f"Model bundle is missing keys: {missing_keys}\n"
            f"Available keys: {list(model_bundle.keys())}"
        )

    return model_bundle


def infer_label_from_path(image_path: Path) -> Optional[str]:
    """
    Infer expected label from MVTec AD folder structure.

    Examples:
    bottle/test/good/000.png -> normal
    bottle/test/broken_large/000.png -> defective
    """
    image_path = Path(image_path)

    parent_folder = image_path.parent.name

    if parent_folder == "good":
        return "normal"

    if parent_folder in {"broken_large", "broken_small", "contamination"}:
        return "defective"

    return None


def infer_defect_type_from_path(image_path: Path) -> Optional[str]:
    """
    Infer defect type from parent folder name.
    """
    image_path = Path(image_path)
    parent_folder = image_path.parent.name

    known_defect_types = {
        "good",
        "broken_large",
        "broken_small",
        "contamination",
    }

    if parent_folder in known_defect_types:
        return parent_folder

    return None


def build_prediction_features(
    image_path: Path,
    feature_columns: list[str],
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> pd.DataFrame:
    """
    Extract features from one image and return a one-row dataframe.
    """
    features = extract_features_from_image(
        image_path=image_path,
        image_size=image_size,
    )

    feature_table = pd.DataFrame([features])

    missing_columns = set(feature_columns) - set(feature_table.columns)

    if missing_columns:
        raise ValueError(
            f"Prediction feature table is missing columns: {sorted(missing_columns)}"
        )

    feature_table = feature_table[feature_columns]

    return feature_table


def predict_image(
    image_path: Path,
    model_path: Path = MODEL_PATH,
) -> dict:
    """
    Predict whether one image is normal or defective.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    model_bundle = load_model_bundle(model_path)

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    prediction_features = build_prediction_features(
        image_path=image_path,
        feature_columns=feature_columns,
    )

    predicted_label = model.predict(prediction_features)[0]

    result = {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "expected_label_from_folder": infer_label_from_path(image_path),
        "defect_type_from_folder": infer_defect_type_from_path(image_path),
        "predicted_label": predicted_label,
        "model_type": model_bundle.get("model_type", "unknown"),
        "task": model_bundle.get("task", "unknown"),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(prediction_features)[0]
        class_names = model.classes_

        probability_by_class = {
            class_name: float(probability)
            for class_name, probability in zip(class_names, probabilities)
        }

        result["probability_by_class"] = probability_by_class
        result["confidence"] = probability_by_class[predicted_label]

    return result


def save_prediction_result(result: dict) -> Path:
    """
    Save prediction result as JSON.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_stem = Path(result["image_name"]).stem
    defect_type = result["defect_type_from_folder"] or "unknown"

    output_path = RESULTS_DIR / f"prediction_{defect_type}_{image_stem}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    return output_path


def print_prediction_result(result: dict) -> None:
    """
    Print prediction result clearly.
    """
    print("\nPrediction result")
    print("-----------------")
    print(f"Image name       : {result['image_name']}")
    print(f"Expected label   : {result['expected_label_from_folder']}")
    print(f"Defect type      : {result['defect_type_from_folder']}")
    print(f"Predicted label  : {result['predicted_label']}")
    print(f"Model type       : {result['model_type']}")

    if "confidence" in result:
        print(f"Confidence       : {result['confidence']:.4f}")

    if "probability_by_class" in result:
        print("\nProbability by class:")
        for label, probability in result["probability_by_class"].items():
            print(f"  {label}: {probability:.4f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict whether one MVTec AD image is normal or defective."
    )

    parser.add_argument(
        "--image",
        type=str,
        default=str(DEFAULT_IMAGE_PATH),
        help="Path to the image file to classify.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help="Path to the trained model bundle.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    result = predict_image(
        image_path=Path(args.image),
        model_path=Path(args.model),
    )

    print_prediction_result(result)

    output_path = save_prediction_result(result)
    print(f"\nSaved prediction result to: {output_path}")


if __name__ == "__main__":
    main()
