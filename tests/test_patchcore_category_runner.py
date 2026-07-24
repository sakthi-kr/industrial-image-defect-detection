from pathlib import Path

import pytest

from src.train_patchcore_category import (
    SUPPORTED_CATEGORIES,
    build_run_paths,
    check_dataset,
    parse_arguments,
    validate_category,
)


def make_category_tree(
    data_root: Path,
    category: str,
) -> Path:
    category_path = data_root / category

    for relative_path in (
        Path("train/good"),
        Path("test/good"),
        Path("ground_truth"),
    ):
        (
            category_path
            / relative_path
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    return category_path


def test_supported_categories_are_fixed() -> None:
    assert SUPPORTED_CATEGORIES == (
        "bottle",
        "cable",
        "leather",
        "metal_nut",
        "tile",
    )


def test_validate_category() -> None:
    assert validate_category(" Cable ") == "cable"

    with pytest.raises(
        ValueError,
        match="Unsupported category",
    ):
        validate_category("wood")


def test_build_run_paths_are_category_specific(
    tmp_path: Path,
) -> None:
    paths = build_run_paths(
        "tile",
        project_root=tmp_path,
    )

    assert paths.anomalib_output_dir == (
        tmp_path
        / "results"
        / "patchcore_anomalib"
        / "tile"
    )

    assert paths.summary_path == (
        tmp_path
        / "results"
        / "patchcore_multicategory"
        / "tile"
        / "run_summary.json"
    )


def test_check_dataset_accepts_complete_tree(
    tmp_path: Path,
) -> None:
    expected_path = make_category_tree(
        tmp_path,
        "leather",
    )

    assert check_dataset(
        "leather",
        data_root=tmp_path,
    ) == expected_path


def test_check_dataset_rejects_missing_folder(
    tmp_path: Path,
) -> None:
    category_path = make_category_tree(
        tmp_path,
        "bottle",
    )

    (
        category_path
        / "ground_truth"
    ).rmdir()

    with pytest.raises(
        FileNotFoundError,
        match="Missing required folders",
    ):
        check_dataset(
            "bottle",
            data_root=tmp_path,
        )


def test_parse_check_only_arguments() -> None:
    arguments = parse_arguments(
        [
            "--category",
            "metal_nut",
            "--check-only",
        ]
    )

    assert arguments.category == "metal_nut"
    assert arguments.check_only is True
