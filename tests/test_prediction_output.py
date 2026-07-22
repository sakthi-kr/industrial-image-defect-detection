from pathlib import Path

import pytest

from src.predict import MODEL_PATH, predict_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mvtec_ad"
    / "bottle"
    / "test"
    / "good"
    / "000.png"
)


def test_prediction_output_structure():
    if not MODEL_PATH.exists():
        pytest.skip("Model file does not exist. Run `python src/train.py` first.")

    if not DEFAULT_TEST_IMAGE.exists():
        pytest.skip(f"Test image does not exist: {DEFAULT_TEST_IMAGE}")

    result = predict_image(DEFAULT_TEST_IMAGE)

    required_keys = {
        "image_path",
        "image_name",
        "expected_label_from_folder",
        "defect_type_from_folder",
        "predicted_label",
        "model_type",
        "task",
    }

    assert required_keys.issubset(result.keys())
    assert result["predicted_label"] in {"normal", "defective"}
    assert result["expected_label_from_folder"] == "normal"
    assert result["defect_type_from_folder"] == "good"


def test_prediction_probability_output():
    if not MODEL_PATH.exists():
        pytest.skip("Model file does not exist. Run `python src/train.py` first.")

    if not DEFAULT_TEST_IMAGE.exists():
        pytest.skip(f"Test image does not exist: {DEFAULT_TEST_IMAGE}")

    result = predict_image(DEFAULT_TEST_IMAGE)

    if "probability_by_class" in result:
        probabilities = result["probability_by_class"]

        assert set(probabilities.keys()).issubset({"normal", "defective"})
        assert abs(sum(probabilities.values()) - 1.0) < 1e-6
        assert 0.0 <= result["confidence"] <= 1.0
