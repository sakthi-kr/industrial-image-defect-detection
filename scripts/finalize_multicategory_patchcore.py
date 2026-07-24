from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "patchcore_multicategory"
COMPARISON_CSV = RESULTS_DIR / "category_comparison.csv"
COMPARISON_JSON = RESULTS_DIR / "category_comparison.json"
COMPARISON_PLOT = RESULTS_DIR / "category_comparison.png"
BENCHMARK_SUMMARY = RESULTS_DIR / "benchmark_summary.json"
RANKINGS_CSV = RESULTS_DIR / "category_rankings.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "patchcore_multicategory_benchmark.md"
README_PATH = PROJECT_ROOT / "README.md"

EXPECTED_CATEGORIES = [
    "bottle",
    "cable",
    "leather",
    "metal_nut",
    "tile",
]

METRICS = [
    "image_auroc",
    "image_f1",
    "pixel_auroc",
    "pixel_f1",
]

METRIC_LABELS = {
    "image_auroc": "Image AUROC",
    "image_f1": "Image F1",
    "pixel_auroc": "Pixel AUROC",
    "pixel_f1": "Pixel F1",
}

START_MARKER = "<!-- MULTICATEGORY_PATCHCORE_START -->"
END_MARKER = "<!-- MULTICATEGORY_PATCHCORE_END -->"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def finite_number(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Expected a finite number, received {value!r}")
    return number


def load_and_validate_results() -> tuple[pd.DataFrame, dict[str, Any]]:
    for path in [
        COMPARISON_CSV,
        COMPARISON_JSON,
        COMPARISON_PLOT,
        README_PATH,
    ]:
        require_file(path)

    if COMPARISON_PLOT.stat().st_size == 0:
        raise ValueError("The category comparison plot is empty.")

    table = pd.read_csv(COMPARISON_CSV)
    payload = json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))

    required_columns = {
        "category",
        *METRICS,
        "elapsed_seconds",
        "model",
        "backbone",
        "layers",
        "image_size",
        "coreset_sampling_ratio",
        "num_neighbors",
        "accelerator",
        "summary_path",
    }

    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(
            "Comparison table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    observed_categories = table["category"].astype(str).tolist()
    if set(observed_categories) != set(EXPECTED_CATEGORIES):
        raise ValueError(
            "Expected exactly the five benchmark categories. "
            f"Observed: {observed_categories}"
        )

    table = (
        table.set_index("category")
        .loc[EXPECTED_CATEGORIES]
        .reset_index()
    )

    for metric in METRICS:
        numeric = pd.to_numeric(table[metric], errors="coerce")
        if numeric.isna().any() or not numeric.between(0.0, 1.0).all():
            raise ValueError(
                f"Metric '{metric}' contains missing values or values outside [0, 1]."
            )
        table[metric] = numeric.astype(float)

    elapsed = pd.to_numeric(table["elapsed_seconds"], errors="coerce")
    if elapsed.isna().any() or not (elapsed > 0.0).all():
        raise ValueError("Every category must have a positive elapsed_seconds value.")
    table["elapsed_seconds"] = elapsed.astype(float)

    if int(payload.get("n_completed_categories", -1)) != 5:
        raise ValueError("Comparison JSON must report five completed categories.")

    if set(payload.get("completed_categories", [])) != set(EXPECTED_CATEGORIES):
        raise ValueError("Comparison JSON category list does not match the benchmark.")

    fixed_configuration = payload.get("fixed_configuration")
    if not isinstance(fixed_configuration, dict):
        raise ValueError("Comparison JSON has no fixed_configuration object.")

    required_configuration = {
        "model": "Patchcore",
        "backbone": "resnet18",
        "coreset_sampling_ratio": 0.01,
        "num_neighbors": 5,
        "accelerator": "cpu",
    }

    for key, expected in required_configuration.items():
        observed = fixed_configuration.get(key)
        if observed != expected:
            raise ValueError(
                f"Configuration mismatch for '{key}': "
                f"expected {expected!r}, received {observed!r}"
            )

    for summary_value in table["summary_path"].astype(str):
        summary_path = Path(summary_value)
        if summary_path.is_absolute():
            raise ValueError(
                "summary_path must be project-relative, received: "
                f"{summary_value}"
            )
        require_file(PROJECT_ROOT / summary_path)

    return table, payload


def tied_categories(
    table: pd.DataFrame,
    metric: str,
    *,
    best: bool,
    tolerance: float = 1e-12,
) -> list[str]:
    values = table[metric].astype(float)
    target = values.max() if best else values.min()
    mask = (values - target).abs() <= tolerance
    return table.loc[mask, "category"].astype(str).tolist()


def build_rankings(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for metric in METRICS:
        ordered = table.sort_values(
            [metric, "category"],
            ascending=[False, True],
        ).reset_index(drop=True)

        for position, (_, row) in enumerate(ordered.iterrows(), start=1):
            rows.append(
                {
                    "metric": metric,
                    "metric_name": METRIC_LABELS[metric],
                    "rank": position,
                    "category": row["category"],
                    "score": float(row[metric]),
                }
            )

    return pd.DataFrame(rows)


def format_categories(categories: list[str]) -> str:
    return ", ".join(f"`{category}`" for category in categories)


def markdown_table(table: pd.DataFrame) -> str:
    headers = [
        "Category",
        "Image AUROC",
        "Image F1",
        "Pixel AUROC",
        "Pixel F1",
        "Runtime (s)",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * 5) + " |",
    ]

    for _, row in table.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["category"]),
                    f"{row['image_auroc']:.3f}",
                    f"{row['image_f1']:.3f}",
                    f"{row['pixel_auroc']:.3f}",
                    f"{row['pixel_f1']:.3f}",
                    f"{row['elapsed_seconds']:.1f}",
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def build_summary(table: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    macro_average = {
        metric: float(table[metric].mean())
        for metric in METRICS
    }

    metric_extremes = {
        metric: {
            "best_categories": tied_categories(table, metric, best=True),
            "best_score": float(table[metric].max()),
            "weakest_categories": tied_categories(table, metric, best=False),
            "weakest_score": float(table[metric].min()),
        }
        for metric in METRICS
    }

    return {
        "benchmark": "Fixed-configuration five-category PatchCore benchmark",
        "categories": EXPECTED_CATEGORIES,
        "n_categories": len(EXPECTED_CATEGORIES),
        "fixed_configuration": payload["fixed_configuration"],
        "macro_average": macro_average,
        "metric_extremes": metric_extremes,
        "runtime": {
            "total_seconds": float(table["elapsed_seconds"].sum()),
            "mean_seconds_per_category": float(table["elapsed_seconds"].mean()),
            "fastest_categories": tied_categories(
                table,
                "elapsed_seconds",
                best=False,
            ),
            "fastest_seconds": float(table["elapsed_seconds"].min()),
            "slowest_categories": tied_categories(
                table,
                "elapsed_seconds",
                best=True,
            ),
            "slowest_seconds": float(table["elapsed_seconds"].max()),
        },
        "limitations": [
            "Only five of the fifteen MVTec AD categories are evaluated.",
            "One fixed lightweight configuration is used without category-specific tuning.",
            "The benchmark uses controlled laboratory images rather than production camera data.",
            "AUROC and F1 do not by themselves establish deployment readiness.",
        ],
    }


def build_report(table: pd.DataFrame, summary: dict[str, Any]) -> str:
    config = summary["fixed_configuration"]
    macro = summary["macro_average"]
    extremes = summary["metric_extremes"]

    strongest_lines = []
    for metric in METRICS:
        result = extremes[metric]
        strongest_lines.append(
            "- **"
            + METRIC_LABELS[metric]
            + ":** best "
            + format_categories(result["best_categories"])
            + f" at {result['best_score']:.3f}; weakest "
            + format_categories(result["weakest_categories"])
            + f" at {result['weakest_score']:.3f}."
        )

    lines = [
        "# Five-Category PatchCore Benchmark",
        "",
        "## Purpose",
        "",
        (
            "This experiment tests whether one unchanged PatchCore configuration "
            "generalizes across five visually different MVTec AD categories: "
            "three object categories (`bottle`, `cable`, and `metal_nut`) and "
            "two texture categories (`leather` and `tile`)."
        ),
        "",
        "## Fixed Configuration",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| Model | {config.get('model')} |",
        f"| Backbone | {config.get('backbone')} |",
        f"| Feature layers | {', '.join(config.get('layers', []))} |",
        f"| Input size | {' × '.join(str(value) for value in config.get('image_size', []))} |",
        f"| Coreset sampling ratio | {config.get('coreset_sampling_ratio')} |",
        f"| Nearest neighbours | {config.get('num_neighbors')} |",
        f"| Execution | {config.get('accelerator')} |",
        "",
        "No category-specific hyperparameter tuning was performed. This makes the comparison fair, but it may not give the best possible result for every category.",
        "",
        "## Category Results",
        "",
        markdown_table(table),
        "",
        "![Five-category PatchCore comparison](../results/patchcore_multicategory/category_comparison.png)",
        "",
        "## Macro Averages",
        "",
        "| Metric | Macro average |",
        "| --- | ---: |",
        f"| Image AUROC | {macro['image_auroc']:.3f} |",
        f"| Image F1 | {macro['image_f1']:.3f} |",
        f"| Pixel AUROC | {macro['pixel_auroc']:.3f} |",
        f"| Pixel F1 | {macro['pixel_f1']:.3f} |",
        "",
        "## Strongest and Weakest Categories",
        "",
        *strongest_lines,
        "",
        "The strongest category depends on the chosen metric. Image-level detection and pixel-level localization answer different questions, so no single category is labelled as universally best or worst.",
        "",
        "## Runtime",
        "",
        f"- Total CPU runtime: **{summary['runtime']['total_seconds']:.1f} seconds**.",
        f"- Mean CPU runtime: **{summary['runtime']['mean_seconds_per_category']:.1f} seconds per category**.",
        f"- Fastest: {format_categories(summary['runtime']['fastest_categories'])} at {summary['runtime']['fastest_seconds']:.1f} seconds.",
        f"- Slowest: {format_categories(summary['runtime']['slowest_categories'])} at {summary['runtime']['slowest_seconds']:.1f} seconds.",
        "",
        "Runtime is machine-specific and is reported as a local engineering measurement, not as a universal benchmark.",
        "",
        "## Interpretation",
        "",
        (
            "The experiment broadens the project from a single-category demonstration "
            "to a reusable anomaly-detection benchmark across objects and textures. "
            "Differences between image-level and pixel-level scores also show why "
            "detecting that an image is anomalous can be easier than localizing the "
            "defect boundary precisely."
        ),
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in summary["limitations"]],
        "",
        "## Reproduction",
        "",
        "```bash",
        "for category in bottle cable leather metal_nut tile",
        "do",
        "  python -u src/train_patchcore_category.py --category \"$category\"",
        "done",
        "",
        "python scripts/compare_patchcore_categories.py \\",
        "  --categories bottle cable leather metal_nut tile",
        "",
        "python scripts/finalize_multicategory_patchcore.py",
        "```",
        "",
        "## Generated Outputs",
        "",
        "- `results/patchcore_multicategory/category_comparison.csv`",
        "- `results/patchcore_multicategory/category_comparison.json`",
        "- `results/patchcore_multicategory/category_comparison.png`",
        "- `results/patchcore_multicategory/category_rankings.csv`",
        "- `results/patchcore_multicategory/benchmark_summary.json`",
    ]

    return "\n".join(lines).rstrip() + "\n"


def build_readme_section(table: pd.DataFrame, summary: dict[str, Any]) -> str:
    macro = summary["macro_average"]

    lines = [
        START_MARKER,
        "# Multi-Category PatchCore Benchmark",
        "",
        (
            "The fixed PatchCore configuration was evaluated on five selected "
            "MVTec AD categories spanning rigid objects, complex assemblies, "
            "small mechanical parts, and textures."
        ),
        "",
        markdown_table(table),
        "",
        "Macro averages:",
        "",
        f"- Image AUROC: **{macro['image_auroc']:.3f}**",
        f"- Image F1: **{macro['image_f1']:.3f}**",
        f"- Pixel AUROC: **{macro['pixel_auroc']:.3f}**",
        f"- Pixel F1: **{macro['pixel_f1']:.3f}**",
        "",
        "![Five-category PatchCore comparison](results/patchcore_multicategory/category_comparison.png)",
        "",
        (
            "The same ResNet-18 PatchCore settings were used for all five "
            "categories without category-specific tuning. Detailed rankings, "
            "runtime measurements, interpretation, and limitations are available "
            "in [`docs/patchcore_multicategory_benchmark.md`](docs/patchcore_multicategory_benchmark.md)."
        ),
        END_MARKER,
    ]

    return "\n".join(lines)


def update_readme(readme_text: str, section: str) -> str:
    if START_MARKER in readme_text or END_MARKER in readme_text:
        if START_MARKER not in readme_text or END_MARKER not in readme_text:
            raise ValueError("README contains only one multi-category marker.")

        start = readme_text.index(START_MARKER)
        end = readme_text.index(END_MARKER, start) + len(END_MARKER)
        updated = readme_text[:start] + section + readme_text[end:]
    else:
        insertion_heading = "# Repository Structure"
        if insertion_heading not in readme_text:
            raise ValueError(
                "README does not contain the expected '# Repository Structure' heading."
            )
        updated = readme_text.replace(
            insertion_heading,
            section + "\n\n" + insertion_heading,
            1,
        )

    exact_replacements = {
        "using the MVTec AD `bottle` category.": (
            "using five selected MVTec AD categories, with `bottle` retained "
            "for the supervised development baseline."
        ),
        "trained only on normal bottle images.": "trained only on normal images.",
        "* only the MVTec AD `bottle` category has been evaluated": (
            "* five selected MVTec AD categories have been evaluated; "
            "the complete 15-category benchmark has not been run"
        ),
        "3. Evaluate PatchCore on additional MVTec categories": (
            "3. Evaluate PatchCore on additional MVTec categories — completed for five selected categories"
        ),
    }

    for old, new in exact_replacements.items():
        updated = updated.replace(old, new)

    return updated.rstrip() + "\n"


def write_outputs(
    table: pd.DataFrame,
    payload: dict[str, Any],
) -> dict[str, Any]:
    rankings = build_rankings(table)
    summary = build_summary(table, payload)
    report = build_report(table, summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rankings.to_csv(RANKINGS_CSV, index=False)
    BENCHMARK_SUMMARY.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    readme_text = README_PATH.read_text(encoding="utf-8")
    readme_section = build_readme_section(table, summary)
    updated_readme = update_readme(readme_text, readme_section)
    README_PATH.write_text(updated_readme, encoding="utf-8")

    return summary


def validate_generated_outputs(summary: dict[str, Any]) -> None:
    for path in [
        RANKINGS_CSV,
        BENCHMARK_SUMMARY,
        REPORT_PATH,
        README_PATH,
    ]:
        require_file(path)
        if path.stat().st_size == 0:
            raise ValueError(f"Generated file is empty: {path}")

    report = REPORT_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    required_report_text = [
        "# Five-Category PatchCore Benchmark",
        "## Strongest and Weakest Categories",
        "## Runtime",
        "## Limitations",
    ]
    for text in required_report_text:
        if text not in report:
            raise ValueError(f"Generated report is missing: {text}")

    if readme.count(START_MARKER) != 1 or readme.count(END_MARKER) != 1:
        raise ValueError("README must contain exactly one multi-category section.")

    if summary.get("n_categories") != 5:
        raise ValueError("Benchmark summary must report five categories.")


def main() -> int:
    table, payload = load_and_validate_results()
    summary = write_outputs(table, payload)
    validate_generated_outputs(summary)

    print("=" * 84)
    print("FIVE-CATEGORY PATCHCORE BENCHMARK")
    print("=" * 84)
    print(
        table[
            [
                "category",
                "image_auroc",
                "image_f1",
                "pixel_auroc",
                "pixel_f1",
                "elapsed_seconds",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print()
    print("Macro averages:")
    for metric in METRICS:
        print(
            f"  {METRIC_LABELS[metric]:<14}: "
            f"{summary['macro_average'][metric]:.4f}"
        )
    print()
    print(f"Saved report   : {REPORT_PATH}")
    print(f"Saved summary  : {BENCHMARK_SUMMARY}")
    print(f"Saved rankings : {RANKINGS_CSV}")
    print("Updated README : README.md")
    print()
    print("PASS: Five-category PatchCore results were analyzed and documented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
