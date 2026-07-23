from pathlib import Path

import numpy as np
import pytest

from src.preprocess import (
    DEFAULT_IMAGE_SIZE,
    image_to_array,
    load_rgb_image,
    preprocess_image,
    resize_image,
    rgb_to_grayscale_array,
)


def test_load_rgb_image(
    synthetic_image_path: Path,
) -> None:
    image = load_rgb_image(
        synthetic_image_path
    )

    assert image.mode == "RGB"
    assert image.size == (96, 96)


def test_resize_image(
    synthetic_image_path: Path,
) -> None:
    image = load_rgb_image(
        synthetic_image_path
    )

    resized = resize_image(
        image,
        image_size=DEFAULT_IMAGE_SIZE,
    )

    assert resized.size == DEFAULT_IMAGE_SIZE


def test_image_to_array_normalized(
    synthetic_image_path: Path,
) -> None:
    image = load_rgb_image(
        synthetic_image_path
    )

    resized = resize_image(
        image,
        image_size=DEFAULT_IMAGE_SIZE,
    )

    array = image_to_array(
        resized,
        normalize=True,
    )

    assert isinstance(
        array,
        np.ndarray,
    )

    assert array.shape == (
        DEFAULT_IMAGE_SIZE[1],
        DEFAULT_IMAGE_SIZE[0],
        3,
    )

    assert array.dtype == np.float32
    assert array.min() >= 0.0
    assert array.max() <= 1.0


def test_rgb_to_grayscale_array(
    synthetic_image_path: Path,
) -> None:
    processed = preprocess_image(
        synthetic_image_path
    )

    grayscale_array = rgb_to_grayscale_array(
        processed["rgb_array"]
    )

    assert grayscale_array.shape == (
        DEFAULT_IMAGE_SIZE[1],
        DEFAULT_IMAGE_SIZE[0],
    )

    assert grayscale_array.min() >= 0.0
    assert grayscale_array.max() <= 1.0


def test_preprocess_image_output_structure(
    synthetic_image_path: Path,
) -> None:
    processed = preprocess_image(
        synthetic_image_path
    )

    expected_keys = {
        "image_path",
        "original_image",
        "resized_image",
        "rgb_array",
        "grayscale_array",
    }

    assert expected_keys.issubset(
        processed.keys()
    )

    assert processed["image_path"] == (
        synthetic_image_path
    )

    assert processed[
        "rgb_array"
    ].shape == (
        DEFAULT_IMAGE_SIZE[1],
        DEFAULT_IMAGE_SIZE[0],
        3,
    )

    assert processed[
        "grayscale_array"
    ].shape == (
        DEFAULT_IMAGE_SIZE[1],
        DEFAULT_IMAGE_SIZE[0],
    )


def test_invalid_resize_dimensions_raise_error(
    synthetic_image_path: Path,
) -> None:
    image = load_rgb_image(
        synthetic_image_path
    )

    with pytest.raises(
        ValueError,
        match="Invalid image size",
    ):
        resize_image(
            image,
            image_size=(0, 128),
        )


def test_missing_image_raises_error(
    tmp_path: Path,
) -> None:
    missing_image = (
        tmp_path / "missing.png"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Image file not found",
    ):
        load_rgb_image(
            missing_image
        )
