from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_patchcore_coreset_experiment import (
    DEFAULT_RATIOS,
    best_ratio_by_metric,
    build_experiment_paths,
    normalize_ratios,
    parse_arguments,
    ratio_slug,
    save_experiment_outputs,
    summarize_experiment,
    validate_coreset_ratio,
)


def write_summary(
    project_root: Path,
    *,
    category: str,
    ratio: float,
    image_auroc: float,
    image_f1: float,
    pixel_auroc: float,
    pixel_f1: float,
    elapsed_seconds: float,
    memory_bank_embeddings: int,
) -> None:
    path = build_experiment_paths(
        category,
        ratio,
        project_root=project_root,
    ).summary_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "category": category,
        "model": "Patchcore",
        "backbone": "resnet18",
        "layers": ["layer2", "layer3"],
        "image_size": [224, 224],
        "coreset_sampling_ratio": ratio,
        "num_neighbors": 5,
        "accelerator": "cpu",
        "elapsed_seconds": elapsed_seconds,
        "memory_bank_embeddings": memory_bank_embeddings,
        "embedding_dimension": 384,
        "metrics": {
            "image_auroc": image_auroc,
            "image_f1": image_f1,
            "pixel_auroc": pixel_auroc,
            "pixel_f1": pixel_f1,
        },
        "test_results": [],
    }
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "ratio",
    [0.005, 0.01, 0.05, 1.0],
)
def test_validate_coreset_ratio_accepts_valid_values(
    ratio: float,
) -> None:
    assert validate_coreset_ratio(ratio) == ratio


@pytest.mark.parametrize(
    "ratio",
    [0.0, -0.1, 1.1, float("inf")],
)
def test_validate_coreset_ratio_rejects_invalid_values(
    ratio: float,
) -> None:
    with pytest.raises(ValueError):
        validate_coreset_ratio(ratio)


def test_normalize_ratios_sorts_and_deduplicates() -> None:
    assert normalize_ratios(
        [0.05, 0.01, 0.005, 0.01]
    ) == [0.005, 0.01, 0.05]


def test_ratio_slug_and_paths_are_stable(
    tmp_path: Path,
) -> None:
    assert ratio_slug(0.005) == "0p005"
    paths = build_experiment_paths(
        "leather",
        0.005,
        project_root=tmp_path,
    )
    assert paths.summary_path == (
        tmp_path
        / "results"
        / "patchcore_coreset_experiment"
        / "leather"
        / "ratio_0p005"
        / "run_summary.json"
    )


def test_parse_arguments_uses_planned_defaults() -> None:
    parsed = parse_arguments([])
    assert parsed.category == "leather"
    assert parsed.ratios == list(DEFAULT_RATIOS)
    assert parsed.check_only is False
    assert parsed.summarize_only is False


def test_best_ratio_handles_ties() -> None:
    table = pd.DataFrame(
        {
            "coreset_sampling_ratio": [0.005, 0.01, 0.05],
            "pixel_f1": [0.4, 0.6, 0.6],
        }
    )
    assert best_ratio_by_metric(
        table,
        "pixel_f1",
    ) == [0.01, 0.05]


def test_summarize_experiment_creates_outputs(
    tmp_path: Path,
) -> None:
    rows = [
        (0.005, 0.99, 0.96, 0.97, 0.30, 50.0, 100),
        (0.01, 1.00, 0.98, 0.98, 0.34, 60.0, 200),
        (0.05, 1.00, 0.99, 0.99, 0.40, 90.0, 1000),
    ]
    for row in rows:
        write_summary(
            tmp_path,
            category="leather",
            ratio=row[0],
            image_auroc=row[1],
            image_f1=row[2],
            pixel_auroc=row[3],
            pixel_f1=row[4],
            elapsed_seconds=row[5],
            memory_bank_embeddings=row[6],
        )

    table, paths = summarize_experiment(
        "leather",
        [0.005, 0.01, 0.05],
        project_root=tmp_path,
    )

    assert table["coreset_sampling_ratio"].tolist() == [
        0.005,
        0.01,
        0.05,
    ]
    assert table["pixel_f1"].tolist() == [
        0.30,
        0.34,
        0.40,
    ]
    assert all(path.exists() for path in paths.values())

    payload = json.loads(
        paths["json"].read_text(encoding="utf-8")
    )
    assert payload["category"] == "leather"
    assert payload["best_ratio_by_metric"]["pixel_f1"] == [
        0.05
    ]
