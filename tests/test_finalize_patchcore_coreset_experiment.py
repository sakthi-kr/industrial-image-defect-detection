from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.finalize_patchcore_coreset_experiment import (
    README_END,
    README_START,
    build_summary,
    load_comparison_table,
    select_efficiency_ratio,
    update_marked_section,
    write_outputs,
)


def sample_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "category": "leather",
                "coreset_sampling_ratio": 0.005,
                "image_auroc": 0.99,
                "image_f1": 0.97,
                "pixel_auroc": 0.96,
                "pixel_f1": 0.500,
                "elapsed_seconds": 80.0,
                "memory_bank_embeddings": 100,
            },
            {
                "category": "leather",
                "coreset_sampling_ratio": 0.01,
                "image_auroc": 1.00,
                "image_f1": 0.98,
                "pixel_auroc": 0.97,
                "pixel_f1": 0.504,
                "elapsed_seconds": 90.0,
                "memory_bank_embeddings": 200,
            },
            {
                "category": "leather",
                "coreset_sampling_ratio": 0.05,
                "image_auroc": 1.00,
                "image_f1": 0.98,
                "pixel_auroc": 0.98,
                "pixel_f1": 0.505,
                "elapsed_seconds": 130.0,
                "memory_bank_embeddings": 1000,
            },
        ]
    )


def write_csv(root: Path, table: pd.DataFrame) -> None:
    directory = root / "leather"
    directory.mkdir(parents=True)
    table.to_csv(directory / "coreset_comparison.csv", index=False)


def test_selects_smallest_ratio_within_tolerance() -> None:
    assert select_efficiency_ratio(sample_table(), 0.005) == 0.005


def test_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="Tolerance"):
        select_efficiency_ratio(sample_table(), -0.1)


def test_summary_records_best_and_reference() -> None:
    summary = build_summary(sample_table(), "leather", 0.005)
    assert summary["reference_ratio"] == 0.01
    assert summary["efficiency_oriented_ratio"] == 0.005
    assert summary["best_by_metric"]["pixel_f1"]["ratios"] == [0.05]


def test_load_comparison_sorts_ratios(tmp_path: Path) -> None:
    table = sample_table().iloc[::-1]
    write_csv(tmp_path, table)
    loaded = load_comparison_table("leather", tmp_path)
    assert loaded["coreset_sampling_ratio"].tolist() == [0.005, 0.01, 0.05]


def test_update_marked_section_is_idempotent() -> None:
    original = "# Project\n\n<!-- MULTICATEGORY_PATCHCORE_END -->\n\n# Experiment Roadmap\n"
    section = f"{README_START}\ncontent\n{README_END}"
    first = update_marked_section(original, section)
    second = update_marked_section(first, section)
    assert first == second
    assert second.count(README_START) == 1
    assert second.count(README_END) == 1


def test_write_outputs_creates_report_summary_and_readme(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "# Project\n\n<!-- MULTICATEGORY_PATCHCORE_END -->\n",
        encoding="utf-8",
    )
    table = sample_table()
    summary = build_summary(table, "leather", 0.005)
    outputs = write_outputs(table, summary, project_root=tmp_path)

    assert outputs["summary"].exists()
    assert outputs["report"].exists()
    readme = outputs["readme"].read_text(encoding="utf-8")
    assert README_START in readme
    assert "Highest observed pixel F1" in readme

    payload = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert payload["category"] == "leather"
