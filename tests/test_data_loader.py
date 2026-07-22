from pathlib import Path

import pandas as pd

from src.data_loader import (
    DEFAULT_CATEGORY,
    DEFAULT_DATA_ROOT,
    IMAGE_EXTENSIONS,
    check_category_exists,
    find_image_paths,
    load_image_records,
    records_to_dataframe,
)


def test_category_folder_exists():
    category_dir = check_category_exists(
        data_root=DEFAULT_DATA_ROOT,
        category=DEFAULT_CATEGORY,
    )

    assert category_dir.exists()
    assert category_dir.name == "bottle"


def test_find_image_paths():
    image_paths = find_image_paths(
        data_root=DEFAULT_DATA_ROOT,
        category=DEFAULT_CATEGORY,
    )

    assert len(image_paths) > 0
    assert all(path.suffix.lower() in IMAGE_EXTENSIONS for path in image_paths)


def test_load_image_records():
    records = load_image_records(
        data_root=DEFAULT_DATA_ROOT,
        category=DEFAULT_CATEGORY,
    )

    assert len(records) > 0

    splits = {record.split for record in records}
    labels = {record.binary_label for record in records}
    defect_types = {record.defect_type for record in records}

    assert "train" in splits
    assert "test" in splits

    assert "normal" in labels
    assert "defective" in labels

    assert "good" in defect_types
    assert any(defect_type != "good" for defect_type in defect_types)


def test_records_to_dataframe():
    records = load_image_records()
    dataset_table = records_to_dataframe(records)

    assert isinstance(dataset_table, pd.DataFrame)
    assert not dataset_table.empty

    expected_columns = {
        "image_path",
        "category",
        "split",
        "defect_type",
        "binary_label",
        "is_defective",
        "mask_path",
        "has_mask",
    }

    assert expected_columns.issubset(dataset_table.columns)
    assert dataset_table["binary_label"].isin(["normal", "defective"]).all()
    assert dataset_table["split"].isin(["train", "test"]).all()
