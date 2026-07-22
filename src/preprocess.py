from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


try:
    # Works when running as: python -m src.preprocess
    from src.data_loader import load_image_records, records_to_dataframe
except ModuleNotFoundError:
    # Works when running directly from PyCharm: Run preprocess.py
    from data_loader import load_image_records, records_to_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_IMAGE_SIZE = (128, 128)
PREVIEW_OUTPUT_PATH = RESULTS_DIR / "preprocessing_preview.png"


def get_resampling_filter():
    """
    Handle Pillow version differences.
    """
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def load_rgb_image(image_path: str | Path) -> Image.Image:
    """
    Load an image and convert it to RGB.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    return image


def resize_image(
    image: Image.Image,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Image.Image:
    """
    Resize image to a fixed size.

    image_size is given as (width, height), following Pillow convention.
    """
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError(f"Invalid image size: {image_size}")

    resized = image.resize(image_size, get_resampling_filter())
    return resized


def image_to_array(
    image: Image.Image,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert a PIL image to a NumPy array.

    If normalize=True, pixel values are scaled to [0, 1].
    """
    array = np.asarray(image).astype(np.float32)

    if normalize:
        array = array / 255.0

    return array


def rgb_to_grayscale_array(rgb_array: np.ndarray) -> np.ndarray:
    """
    Convert RGB image array to grayscale using standard luminance weights.

    Input shape:
    - (height, width, 3)

    Output shape:
    - (height, width)
    """
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(
            f"Expected RGB array with shape (height, width, 3), got {rgb_array.shape}"
        )

    grayscale = (
        0.299 * rgb_array[:, :, 0]
        + 0.587 * rgb_array[:, :, 1]
        + 0.114 * rgb_array[:, :, 2]
    )

    return grayscale


def preprocess_image(
    image_path: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict:
    """
    Complete preprocessing for one image.

    Returns:
    - original PIL image
    - resized PIL image
    - normalized RGB array
    - normalized grayscale array
    """
    original_image = load_rgb_image(image_path)
    resized_image = resize_image(original_image, image_size=image_size)

    rgb_array = image_to_array(resized_image, normalize=True)
    grayscale_array = rgb_to_grayscale_array(rgb_array)

    return {
        "image_path": Path(image_path),
        "original_image": original_image,
        "resized_image": resized_image,
        "rgb_array": rgb_array,
        "grayscale_array": grayscale_array,
    }


def select_preprocessing_examples(
    dataset_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select a small set of normal and defective images for preprocessing preview.
    """
    examples = []

    groups_to_show = [
        ("train", "good"),
        ("test", "good"),
        ("test", "broken_large"),
        ("test", "broken_small"),
        ("test", "contamination"),
    ]

    for split, defect_type in groups_to_show:
        group = dataset_table[
            (dataset_table["split"] == split)
            & (dataset_table["defect_type"] == defect_type)
        ]

        if group.empty:
            print(f"Warning: no example found for {split}/{defect_type}")
            continue

        examples.append(group.iloc[[0]])

    if not examples:
        raise ValueError("No preprocessing examples found.")

    return pd.concat(examples, ignore_index=True)


def save_preprocessing_preview(
    example_table: pd.DataFrame,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    output_path: Path = PREVIEW_OUTPUT_PATH,
) -> Path:
    """
    Save a preview figure showing original, resized RGB, and grayscale images.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_examples = len(example_table)
    n_cols = 3
    n_rows = n_examples

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10, 3.2 * n_rows),
    )

    if n_rows == 1:
        axes = np.asarray([axes])

    for row_index, (_, row) in enumerate(example_table.iterrows()):
        processed = preprocess_image(
            row["image_path"],
            image_size=image_size,
        )

        original_image = processed["original_image"]
        resized_image = processed["resized_image"]
        grayscale_array = processed["grayscale_array"]

        label_text = f"{row['split']} / {row['defect_type']} / {row['binary_label']}"

        axes[row_index, 0].imshow(original_image)
        axes[row_index, 0].set_title(f"Original\n{label_text}", fontsize=9)
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(resized_image)
        axes[row_index, 1].set_title(f"Resized RGB\n{image_size}", fontsize=9)
        axes[row_index, 1].axis("off")

        axes[row_index, 2].imshow(grayscale_array, cmap="gray")
        axes[row_index, 2].set_title("Grayscale array", fontsize=9)
        axes[row_index, 2].axis("off")

    fig.suptitle(
        "Image Preprocessing Preview",
        fontsize=14,
        y=0.995,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def check_preprocessed_image(
    image_path: str | Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> None:
    """
    Load and preprocess one image, then print shape and value checks.
    """
    processed = preprocess_image(
        image_path=image_path,
        image_size=image_size,
    )

    rgb_array = processed["rgb_array"]
    grayscale_array = processed["grayscale_array"]

    print("\nPreprocessing check")
    print("-------------------")
    print(f"Image path      : {processed['image_path']}")
    print(f"Original size   : {processed['original_image'].size}")
    print(f"Resized size    : {processed['resized_image'].size}")
    print(f"RGB array shape : {rgb_array.shape}")
    print(f"Gray array shape: {grayscale_array.shape}")
    print(f"RGB min/max     : {rgb_array.min():.4f} / {rgb_array.max():.4f}")
    print(f"Gray min/max    : {grayscale_array.min():.4f} / {grayscale_array.max():.4f}")


def main() -> None:
    records = load_image_records()
    dataset_table = records_to_dataframe(records)

    example_table = select_preprocessing_examples(dataset_table)

    first_image_path = example_table.iloc[0]["image_path"]
    check_preprocessed_image(first_image_path)

    output_path = save_preprocessing_preview(example_table)

    print("\nPreprocessing preview created.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
