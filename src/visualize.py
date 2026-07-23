from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


try:
    # Works when running as: python -m src.visualize
    from src.data_loader import load_image_records, records_to_dataframe
except ModuleNotFoundError:
    # Works when running directly from PyCharm: Run visualize.py
    from data_loader import load_image_records, records_to_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

SAMPLE_OUTPUT_PATH = RESULTS_DIR / "sample_images.jpg"


def load_dataset_table() -> pd.DataFrame:
    """
    Load MVTec AD bottle records and convert them into a dataframe.
    """
    records = load_image_records()
    dataset_table = records_to_dataframe(records)
    return dataset_table


def select_sample_images(
    dataset_table: pd.DataFrame,
    samples_per_group: int = 3,
) -> pd.DataFrame:
    """
    Select a small set of sample images for visualization.

    The grid includes:
    - train/good examples
    - test/good examples
    - defective examples from each defect type
    """
    selected_rows = []

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
            print(f"Warning: no images found for {split}/{defect_type}")
            continue

        selected_rows.append(
            group.head(samples_per_group)
        )

    if not selected_rows:
        raise ValueError("No sample images selected. Check dataset_table.")

    sample_table = pd.concat(selected_rows, ignore_index=True)
    return sample_table


def load_image(image_path: str) -> Image.Image:
    """
    Load an image as RGB.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path).convert("RGB")
    return image


def make_sample_grid(
    sample_table: pd.DataFrame,
    output_path: Path = SAMPLE_OUTPUT_PATH,
) -> Path:
    """
    Create and save a grid of sample images.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_images = len(sample_table)
    n_cols = 3
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 4 * n_rows),
    )

    # Make axes always iterable as a flat list
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    axes_flat = axes.flatten()

    for ax, (_, row) in zip(axes_flat, sample_table.iterrows()):
        image = load_image(row["image_path"])

        ax.imshow(image)
        ax.axis("off")

        title = (
            f"{row['split']} / {row['defect_type']}\n"
            f"{row['binary_label']}"
        )
        ax.set_title(title, fontsize=10)

    # Hide unused axes if grid is not completely filled
    for ax in axes_flat[n_images:]:
        ax.axis("off")

    fig.suptitle(
        "MVTec AD Bottle Samples: Normal and Defective Images",
        fontsize=14,
        y=0.995,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def main() -> None:
    dataset_table = load_dataset_table()

    sample_table = select_sample_images(
        dataset_table=dataset_table,
        samples_per_group=3,
    )

    output_path = make_sample_grid(sample_table)

    print("\nSample image grid created.")
    print(f"Saved to: {output_path}")
    print(f"Number of images shown: {len(sample_table)}")


if __name__ == "__main__":
    main()
