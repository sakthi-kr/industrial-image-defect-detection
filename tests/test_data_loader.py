from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    DEFAULT_CATEGORY,
    DEFAULT_DATA_ROOT,
    IMAGE_EXTENSIONS,
    check_category_exists,
    find_image_paths,
    load_image_records,
    records_to_dataframe,
)


def test_default_data_root_is_path() -> None:
    """
    The real MVTec dataset is external, so CI checks only the path type.
    """
    assert isinstance(
        DEFAULT_DATA_ROOT,
        Path,
    )


def test_category_folder_exists(
    synthetic_mvtec_root: Path,
) -> None:
    category_directory = check_category_exists(
        data_root=synthetic_mvtec_root,
        category=DEFAULT_CATEGORY,
    )

    assert category_directory.exists()
    assert category_directory.name == "bottle"


def test_find_image_paths(
    synthetic_mvtec_root: Path,
) -> None:
    image_paths = find_image_paths(
        data_root=synthetic_mvtec_root,
        category=DEFAULT_CATEGORY,
    )

    assert len(image_paths) == 5

    assert all(
        path.suffix.lower() in IMAGE_EXTENSIONS
        for path in image_paths
    )

    assert all(
        "ground_truth" not in path.parts
        for path in image_paths
    )


def test_load_image_records(
    synthetic_mvtec_root: Path,
) -> None:
    records = load_image_records(
        data_root=synthetic_mvtec_root,
        category=DEFAULT_CATEGORY,
    )

    assert len(records) == 5

    splits = {
        record.split
        for record in records
    }

    labels = {
        record.binary_label
        for record in records
    }

    defect_types = {
        record.defect_type
        for record in records
    }

    assert splits == {
        "train",
        "test",
    }

    assert labels == {
        "normal",
        "defective",
    }

    assert defect_types == {
        "good",
        "broken_large",
        "broken_small",
        "contamination",
    }

    defective_records = [
        record
        for record in records
        if record.is_defective
    ]

    assert len(defective_records) == 3

    assert all(
        record.mask_path is not None
        for record in defective_records
    )

    assert all(
        record.mask_path.exists()
        for record in defective_records
        if record.mask_path is not None
    )


def test_records_to_dataframe(
    synthetic_mvtec_root: Path,
) -> None:
    records = load_image_records(
        data_root=synthetic_mvtec_root,
        category=DEFAULT_CATEGORY,
    )

    dataset_table = records_to_dataframe(
        records
    )

    assert isinstance(
        dataset_table,
        pd.DataFrame,
    )

    assert len(dataset_table) == 5

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

    assert expected_columns.issubset(
        dataset_table.columns
    )

    assert dataset_table[
        "binary_label"
    ].isin(
        ["normal", "defective"]
    ).all()

    assert dataset_table[
        "split"
    ].isin(
        ["train", "test"]
    ).all()

    assert int(
        dataset_table["has_mask"].sum()
    ) == 3


def test_missing_category_raises_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Category folder not found",
    ):
        check_category_exists(
            data_root=tmp_path,
            category="missing_category",
        )
