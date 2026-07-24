from pathlib import Path

import pandas as pd

from scripts.finalize_multicategory_patchcore import (
    END_MARKER,
    START_MARKER,
    build_rankings,
    build_summary,
    tied_categories,
    update_readme,
)


def make_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": [
                "bottle",
                "cable",
                "leather",
                "metal_nut",
                "tile",
            ],
            "image_auroc": [1.0, 0.9, 0.95, 0.9, 0.8],
            "image_f1": [0.99, 0.85, 0.9, 0.85, 0.8],
            "pixel_auroc": [0.97, 0.9, 0.96, 0.91, 0.88],
            "pixel_f1": [0.65, 0.55, 0.7, 0.55, 0.5],
            "elapsed_seconds": [60.0, 80.0, 70.0, 80.0, 50.0],
        }
    )


def make_payload() -> dict:
    return {
        "fixed_configuration": {
            "model": "Patchcore",
            "backbone": "resnet18",
            "layers": ["layer2", "layer3"],
            "image_size": [224, 224],
            "coreset_sampling_ratio": 0.01,
            "num_neighbors": 5,
            "accelerator": "cpu",
        }
    }


def test_tied_categories_returns_all_ties() -> None:
    table = make_table()
    assert tied_categories(
        table,
        "image_auroc",
        best=False,
    ) == ["tile"]
    assert tied_categories(
        table,
        "elapsed_seconds",
        best=True,
    ) == ["cable", "metal_nut"]


def test_rankings_contains_every_category_for_every_metric() -> None:
    rankings = build_rankings(make_table())
    assert len(rankings) == 20
    assert set(rankings["metric"]) == {
        "image_auroc",
        "image_f1",
        "pixel_auroc",
        "pixel_f1",
    }


def test_summary_is_tie_aware() -> None:
    summary = build_summary(make_table(), make_payload())
    assert summary["n_categories"] == 5
    assert summary["runtime"]["slowest_categories"] == [
        "cable",
        "metal_nut",
    ]
    assert summary["metric_extremes"]["image_auroc"][
        "weakest_categories"
    ] == ["tile"]


def test_update_readme_inserts_section_once() -> None:
    original = (
        "# Industrial Image Defect Detection\n\n"
        "# Repository Structure\n\nContent\n"
    )
    section = (
        f"{START_MARKER}\n"
        "# Multi-Category PatchCore Benchmark\n"
        f"{END_MARKER}"
    )

    updated = update_readme(original, section)
    assert updated.count(START_MARKER) == 1
    assert updated.count(END_MARKER) == 1
    assert updated.index(START_MARKER) < updated.index(
        "# Repository Structure"
    )

    updated_again = update_readme(updated, section)
    assert updated_again.count(START_MARKER) == 1
    assert updated_again.count(END_MARKER) == 1


def test_update_readme_replaces_stale_bottle_claim() -> None:
    original = (
        "This project develops a pipeline using the MVTec AD `bottle` category.\n\n"
        "# Repository Structure\n"
    )
    section = f"{START_MARKER}\nbenchmark\n{END_MARKER}"
    updated = update_readme(original, section)
    assert "using five selected MVTec AD categories" in updated
