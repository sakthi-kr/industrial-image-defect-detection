from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_ad"
DEFAULT_CATEGORY = "bottle"

RESULTS_DIR = PROJECT_ROOT / "results"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageRecord:
    """
    Container for one image sample from MVTec AD.
    """
    image_path: Path
    category: str
    split: str
    defect_type: str
    binary_label: str
    is_defective: bool
    mask_path: Optional[Path] = None


def check_category_exists(
    data_root: Path = DEFAULT_DATA_ROOT,
    category: str = DEFAULT_CATEGORY,
) -> Path:
    """
    Check that the requested MVTec AD category exists locally.

    Expected structure:

    data/raw/mvtec_ad/bottle/
        train/good/
        test/good/
        test/broken_large/
        test/broken_small/
        test/contamination/
        ground_truth/
    """
    category_dir = Path(data_root) / category

    if not category_dir.exists():
        raise FileNotFoundError(
            f"Category folder not found: {category_dir}\n\n"
            "Expected local structure:\n"
            "data/raw/mvtec_ad/bottle/\n\n"
            "Check that you downloaded and extracted the MVTec AD bottle category."
        )

    return category_dir


def is_image_file(path: Path) -> bool:
    """
    Return True if the file has an image extension.
    """
    return path.suffix.lower() in IMAGE_EXTENSIONS


def find_image_paths(
    data_root: Path = DEFAULT_DATA_ROOT,
    category: str = DEFAULT_CATEGORY,
) -> list[Path]:
    """
    Find all image files in train/ and test/ folders.

    Ground-truth mask images are intentionally excluded here because they are
    annotations, not input images.
    """
    category_dir = check_category_exists(data_root=data_root, category=category)

    image_paths = []

    for split in ["train", "test"]:
        split_dir = category_dir / split

        if not split_dir.exists():
            raise FileNotFoundError(f"Expected split folder not found: {split_dir}")

        for path in split_dir.rglob("*"):
            if path.is_file() and is_image_file(path):
                image_paths.append(path)

    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(
            f"No image files found inside train/test folders of: {category_dir}"
        )

    return image_paths


def infer_split_and_defect_type(image_path: Path, category_dir: Path) -> tuple[str, str]:
    """
    Infer split and defect type from image path.

    Example:
    bottle/train/good/000.png -> split=train, defect_type=good
    bottle/test/broken_large/000.png -> split=test, defect_type=broken_large
    """
    relative_parts = image_path.relative_to(category_dir).parts

    if len(relative_parts) < 3:
        raise ValueError(
            f"Unexpected image path structure: {image_path}\n"
            "Expected structure like: bottle/test/good/000.png"
        )

    split = relative_parts[0]
    defect_type = relative_parts[1]

    if split not in {"train", "test"}:
        raise ValueError(
            f"Unknown split '{split}' from image path: {image_path}"
        )

    return split, defect_type


def find_mask_path(
    image_path: Path,
    category_dir: Path,
    split: str,
    defect_type: str,
) -> Optional[Path]:
    """
    Find the ground-truth mask path for defective test images.

    MVTec AD mask convention is usually:

    test/broken_large/000.png
    ground_truth/broken_large/000_mask.png

    Normal images do not have masks.
    """
    if split != "test" or defect_type == "good":
        return None

    expected_mask_path = (
        category_dir
        / "ground_truth"
        / defect_type
        / f"{image_path.stem}_mask.png"
    )

    if expected_mask_path.exists():
        return expected_mask_path

    return None


def create_image_record(
    image_path: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
    category: str = DEFAULT_CATEGORY,
) -> ImageRecord:
    """
    Create one ImageRecord from an image path.
    """
    category_dir = check_category_exists(data_root=data_root, category=category)

    split, defect_type = infer_split_and_defect_type(
        image_path=image_path,
        category_dir=category_dir,
    )

    is_defective = defect_type != "good"
    binary_label = "defective" if is_defective else "normal"

    mask_path = find_mask_path(
        image_path=image_path,
        category_dir=category_dir,
        split=split,
        defect_type=defect_type,
    )

    return ImageRecord(
        image_path=image_path,
        category=category,
        split=split,
        defect_type=defect_type,
        binary_label=binary_label,
        is_defective=is_defective,
        mask_path=mask_path,
    )


def load_image_records(
    data_root: Path = DEFAULT_DATA_ROOT,
    category: str = DEFAULT_CATEGORY,
) -> list[ImageRecord]:
    """
    Load all image records for one MVTec AD category.
    """
    image_paths = find_image_paths(data_root=data_root, category=category)

    records = [
        create_image_record(
            image_path=image_path,
            data_root=data_root,
            category=category,
        )
        for image_path in image_paths
    ]

    return records


def records_to_dataframe(records: list[ImageRecord]) -> pd.DataFrame:
    """
    Convert ImageRecord objects into a pandas DataFrame.
    """
    rows = []

    for record in records:
        rows.append(
            {
                "image_path": str(record.image_path),
                "category": record.category,
                "split": record.split,
                "defect_type": record.defect_type,
                "binary_label": record.binary_label,
                "is_defective": record.is_defective,
                "mask_path": str(record.mask_path) if record.mask_path else "",
                "has_mask": record.mask_path is not None,
            }
        )

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        raise ValueError("Dataset dataframe is empty.")

    return dataframe


def save_dataset_outputs(dataset_table: pd.DataFrame) -> None:
    """
    Save dataset index and summary files.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_index_path = RESULTS_DIR / "dataset_index.csv"
    dataset_summary_path = RESULTS_DIR / "dataset_summary.csv"

    dataset_table.to_csv(dataset_index_path, index=False)

    summary = (
        dataset_table
        .groupby(["split", "defect_type", "binary_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "binary_label", "defect_type"])
    )

    summary.to_csv(dataset_summary_path, index=False)

    print(f"\nSaved dataset index to  : {dataset_index_path}")
    print(f"Saved dataset summary to: {dataset_summary_path}")


def print_dataset_summary(dataset_table: pd.DataFrame) -> None:
    """
    Print a readable dataset summary.
    """
    print("\nDataset loaded successfully.")
    print(f"Total images: {len(dataset_table)}")

    print("\nCounts by split:")
    print(dataset_table["split"].value_counts())

    print("\nCounts by binary label:")
    print(dataset_table["binary_label"].value_counts())

    print("\nCounts by split and defect type:")
    print(
        dataset_table
        .groupby(["split", "defect_type"])
        .size()
        .reset_index(name="count")
        .to_string(index=False)
    )

    print("\nMask availability:")
    print(dataset_table["has_mask"].value_counts())


def main() -> None:
    records = load_image_records()
    dataset_table = records_to_dataframe(records)

    print_dataset_summary(dataset_table)
    save_dataset_outputs(dataset_table)


if __name__ == "__main__":
    main()
