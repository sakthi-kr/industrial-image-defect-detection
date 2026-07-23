from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def create_rgb_test_image(
    output_path: Path,
    *,
    variation: int,
    defect_region: tuple[int, int, int, int] | None = None,
) -> None:
    """
    Create a deterministic synthetic RGB image.

    The generated image contains gradients and texture so preprocessing and
    handcrafted feature calculations have meaningful non-constant inputs.
    """
    height = 96
    width = 96

    y_coordinates, x_coordinates = np.indices(
        (height, width)
    )

    red = (
        x_coordinates * 2
        + y_coordinates
        + variation
    ) % 256

    green = (
        x_coordinates
        + y_coordinates * 3
        + variation * 2
    ) % 256

    blue = (
        x_coordinates * 3
        + y_coordinates * 2
        + variation * 4
    ) % 256

    image_array = np.stack(
        [red, green, blue],
        axis=-1,
    ).astype(np.uint8)

    if defect_region is not None:
        x_start, y_start, x_end, y_end = defect_region

        image_array[
            y_start:y_end,
            x_start:x_end,
        ] = np.array(
            [255, 255, 255],
            dtype=np.uint8,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        image_array,
        mode="RGB",
    ).save(output_path)


def create_mask(
    output_path: Path,
    *,
    defect_region: tuple[int, int, int, int],
) -> None:
    """Create a binary ground-truth mask."""
    height = 96
    width = 96

    mask_array = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    x_start, y_start, x_end, y_end = defect_region

    mask_array[
        y_start:y_end,
        x_start:x_end,
    ] = 255

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        mask_array,
        mode="L",
    ).save(output_path)


@pytest.fixture
def synthetic_mvtec_root(
    tmp_path: Path,
) -> Path:
    """
    Create a temporary MVTec-style bottle dataset.

    Returned path corresponds to the data root expected by data_loader.py:

        <root>/bottle/train/good
        <root>/bottle/test/good
        <root>/bottle/test/<defect_type>
        <root>/bottle/ground_truth/<defect_type>
    """
    data_root = tmp_path / "mvtec_ad"
    category_root = data_root / "bottle"

    create_rgb_test_image(
        category_root
        / "train"
        / "good"
        / "000.png",
        variation=10,
    )

    create_rgb_test_image(
        category_root
        / "test"
        / "good"
        / "000.png",
        variation=20,
    )

    defect_configuration = {
        "broken_large": (20, 55, 76, 85),
        "broken_small": (12, 42, 28, 58),
        "contamination": (62, 30, 82, 64),
    }

    for index, (
        defect_type,
        defect_region,
    ) in enumerate(
        defect_configuration.items(),
        start=1,
    ):
        create_rgb_test_image(
            category_root
            / "test"
            / defect_type
            / "000.png",
            variation=20 + index * 10,
            defect_region=defect_region,
        )

        create_mask(
            category_root
            / "ground_truth"
            / defect_type
            / "000_mask.png",
            defect_region=defect_region,
        )

    return data_root


@pytest.fixture
def synthetic_image_path(
    synthetic_mvtec_root: Path,
) -> Path:
    """Return one synthetic defective image."""
    return (
        synthetic_mvtec_root
        / "bottle"
        / "test"
        / "broken_small"
        / "000.png"
    )
