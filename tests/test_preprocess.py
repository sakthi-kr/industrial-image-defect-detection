import numpy as np

from src.data_loader import load_image_records
from src.preprocess import (
    DEFAULT_IMAGE_SIZE,
    image_to_array,
    load_rgb_image,
    preprocess_image,
    resize_image,
    rgb_to_grayscale_array,
)


def get_first_image_path():
    records = load_image_records()
    return records[0].image_path


def test_load_rgb_image():
    image_path = get_first_image_path()
    image = load_rgb_image(image_path)

    assert image.mode == "RGB"
    assert image.size[0] > 0
    assert image.size[1] > 0


def test_resize_image():
    image_path = get_first_image_path()
    image = load_rgb_image(image_path)
    resized = resize_image(image, image_size=DEFAULT_IMAGE_SIZE)

    assert resized.size == DEFAULT_IMAGE_SIZE


def test_image_to_array_normalized():
    image_path = get_first_image_path()
    image = load_rgb_image(image_path)
    resized = resize_image(image, image_size=DEFAULT_IMAGE_SIZE)

    array = image_to_array(resized, normalize=True)

    assert isinstance(array, np.ndarray)
    assert array.shape == (DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], 3)
    assert array.min() >= 0.0
    assert array.max() <= 1.0


def test_rgb_to_grayscale_array():
    image_path = get_first_image_path()
    processed = preprocess_image(image_path)

    rgb_array = processed["rgb_array"]
    grayscale_array = rgb_to_grayscale_array(rgb_array)

    assert grayscale_array.shape == (DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0])
    assert grayscale_array.min() >= 0.0
    assert grayscale_array.max() <= 1.0


def test_preprocess_image_output_structure():
    image_path = get_first_image_path()
    processed = preprocess_image(image_path)

    expected_keys = {
        "image_path",
        "original_image",
        "resized_image",
        "rgb_array",
        "grayscale_array",
    }

    assert expected_keys.issubset(processed.keys())
    assert processed["rgb_array"].shape == (DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0], 3)
    assert processed["grayscale_array"].shape == (DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0])
