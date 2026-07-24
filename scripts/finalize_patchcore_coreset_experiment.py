from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORY = "leather"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "patchcore_coreset_experiment"
DEFAULT_TOLERANCE = 0.005
README_START = "<!-- PATCHCORE_CORESET_EXPERIMENT_START -->"
README_END = "<!-- PATCHCORE_CORESET_EXPERIMENT_END -->"
METRIC_COLUMNS = ("image_auroc", "image_f1", "pixel_auroc", "pixel_f1")
REQUIRED_COLUMNS = (
    "category",
    "coreset_sampling_ratio",
    *METRIC_COLUMNS,
    "elapsed_seconds",
    "memory_bank_embeddings",
)


def load_comparison_table(category: str, results_root: Path) -> pd.DataFrame:
    path = Path(results_root) / category / "coreset_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {path}")

    table = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"Comparison CSV is missing columns: {missing}")
    if table.empty:
        raise ValueError("Comparison CSV is empty.")
    if set(table["category"].astype(str)) != {category}:
        raise ValueError("Comparison CSV contains an unexpected category.")

    numeric_columns = [
        "coreset_sampling_ratio",
        *METRIC_COLUMNS,
        "elapsed_seconds",
    ]
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
        if table[column].isna().any():
            raise ValueError(f"Column '{column}' contains missing or invalid values.")

    for metric in METRIC_COLUMNS:
        if ((table[metric] < 0.0) | (table[metric] > 1.0)).any():
            raise ValueError(f"Metric '{metric}' must lie in [0, 1].")
    if (table["coreset_sampling_ratio"] <= 0.0).any():
        raise ValueError("Coreset ratios must be positive.")
    if (table["elapsed_seconds"] <= 0.0).any():
        raise ValueError("Elapsed times must be positive.")

    table = table.sort_values("coreset_sampling_ratio").reset_index(drop=True)
    if table["coreset_sampling_ratio"].duplicated().any():
        raise ValueError("Coreset ratios must be unique.")
    return table


def tied_ratios(table: pd.DataFrame, column: str, *, maximize: bool) -> list[float]:
    target = float(table[column].max() if maximize else table[column].min())
    mask = table[column].sub(target).abs() <= 1e-12
    return [float(value) for value in table.loc[mask, "coreset_sampling_ratio"].tolist()]


def select_efficiency_ratio(table: pd.DataFrame, tolerance: float) -> float:
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("Tolerance must be a finite non-negative number.")

    best_pixel_f1 = float(table["pixel_f1"].max())
    candidates = table[table["pixel_f1"] >= best_pixel_f1 - tolerance]
    if candidates.empty:
        raise ValueError("No efficiency candidate could be selected.")

    return float(candidates["coreset_sampling_ratio"].min())


def percent_change(observed: float, reference: float) -> float | None:
    if math.isclose(reference, 0.0, abs_tol=1e-15):
        return None
    return 100.0 * (observed - reference) / reference


def find_reference_row(table: pd.DataFrame, preferred_ratio: float = 0.01) -> pd.Series:
    mask = table["coreset_sampling_ratio"].sub(preferred_ratio).abs() <= 1e-12
    if mask.any():
        return table.loc[mask].iloc[0]
    return table.iloc[0]


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and callable(value.item):
        try:
            return make_json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return str(value)


def build_summary(table: pd.DataFrame, category: str, tolerance: float) -> dict[str, Any]:
    reference = find_reference_row(table)
    reference_ratio = float(reference["coreset_sampling_ratio"])
    efficiency_ratio = select_efficiency_ratio(table, tolerance)
    efficiency = table[
        table["coreset_sampling_ratio"].sub(efficiency_ratio).abs() <= 1e-12
    ].iloc[0]

    metric_best = {
        metric: {
            "score": float(table[metric].max()),
            "ratios": tied_ratios(table, metric, maximize=True),
        }
        for metric in METRIC_COLUMNS
    }

    memory_available = table["memory_bank_embeddings"].notna().all()
    memory_details: dict[str, Any] = {"available": bool(memory_available)}
    if memory_available:
        memory_table = table.copy()
        memory_table["memory_bank_embeddings"] = pd.to_numeric(
            memory_table["memory_bank_embeddings"], errors="raise"
        )
        memory_details.update(
            {
                "smallest_ratios": tied_ratios(
                    memory_table, "memory_bank_embeddings", maximize=False
                ),
                "largest_ratios": tied_ratios(
                    memory_table, "memory_bank_embeddings", maximize=True
                ),
                "reference_embeddings": int(reference["memory_bank_embeddings"]),
                "efficiency_embeddings": int(efficiency["memory_bank_embeddings"]),
                "efficiency_vs_reference_percent": percent_change(
                    float(efficiency["memory_bank_embeddings"]),
                    float(reference["memory_bank_embeddings"]),
                ),
            }
        )

    row_records = []
    for _, row in table.iterrows():
        record = row.to_dict()
        record["pixel_f1_delta_vs_reference"] = (
            float(row["pixel_f1"]) - float(reference["pixel_f1"])
        )
        record["runtime_change_vs_reference_percent"] = percent_change(
            float(row["elapsed_seconds"]),
            float(reference["elapsed_seconds"]),
        )
        row_records.append(record)

    return make_json_safe(
        {
            "experiment": "PatchCore coreset-ratio sensitivity",
            "category": category,
            "controlled_variable": "coreset_sampling_ratio",
            "primary_metric": "pixel_f1",
            "selection_rule": (
                "Choose the smallest coreset ratio whose pixel F1 is within "
                f"{tolerance:.3f} absolute of the best observed pixel F1."
            ),
            "efficiency_tolerance": tolerance,
            "efficiency_oriented_ratio": efficiency_ratio,
            "reference_ratio": reference_ratio,
            "best_by_metric": metric_best,
            "fastest_ratios": tied_ratios(table, "elapsed_seconds", maximize=False),
            "slowest_ratios": tied_ratios(table, "elapsed_seconds", maximize=True),
            "memory_bank": memory_details,
            "results": row_records,
            "scope_limit": (
                "This conclusion applies to the leather category and the tested "
                "configuration only; it is not a universal PatchCore optimum."
            ),
        }
    )


def format_ratio(value: float) -> str:
    return f"{value:g}"


def format_ratio_list(values: Sequence[float]) -> str:
    return ", ".join(f"`{format_ratio(value)}`" for value in values)


def markdown_table(table: pd.DataFrame) -> str:
    lines = [
        "| Coreset ratio | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 | Runtime (s) | Memory embeddings |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        memory = row["memory_bank_embeddings"]
        memory_text = "not reported" if pd.isna(memory) else f"{int(memory):,}"
        lines.append(
            "| "
            f"{float(row['coreset_sampling_ratio']):.3f} | "
            f"{float(row['image_auroc']):.4f} | "
            f"{float(row['image_f1']):.4f} | "
            f"{float(row['pixel_auroc']):.4f} | "
            f"{float(row['pixel_f1']):.4f} | "
            f"{float(row['elapsed_seconds']):.1f} | "
            f"{memory_text} |"
        )
    return "\n".join(lines)


def build_interpretation(summary: dict[str, Any]) -> list[str]:
    best = summary["best_by_metric"]
    efficient = float(summary["efficiency_oriented_ratio"])
    reference = float(summary["reference_ratio"])
    lines = [
        (
            f"The highest pixel F1 was {best['pixel_f1']['score']:.4f}, "
            f"observed at ratio(s) {format_ratio_list(best['pixel_f1']['ratios'])}."
        ),
        (
            f"Using the documented efficiency rule, ratio `{format_ratio(efficient)}` "
            "is the smallest tested memory-bank fraction that remains within "
            f"{summary['efficiency_tolerance']:.3f} absolute pixel F1 of the best run."
        ),
        (
            f"Ratio `{format_ratio(reference)}` is treated as the reference because it "
            "matches the five-category benchmark configuration when available."
        ),
        (
            "The experiment is a single-category sensitivity study; it supports a "
            "configuration trade-off for leather rather than a universal optimum."
        ),
    ]

    memory = summary["memory_bank"]
    if memory.get("available"):
        change = memory.get("efficiency_vs_reference_percent")
        if change is not None:
            direction = "more" if change > 0 else "fewer"
            lines.insert(
                2,
                (
                    f"The efficiency-oriented run used {abs(change):.1f}% {direction} "
                    "memory-bank embeddings than the reference run."
                ),
            )
    return lines


def build_report(table: pd.DataFrame, summary: dict[str, Any]) -> str:
    category = summary["category"]
    interpretation = "\n".join(f"- {line}" for line in build_interpretation(summary))
    memory_plot = ""
    if summary["memory_bank"].get("available"):
        memory_plot = (
            f"\n![Memory-bank sensitivity](../results/patchcore_coreset_experiment/"
            f"{category}/coreset_memory_bank.png)\n"
        )
    return f"""# PatchCore Coreset-Ratio Sensitivity Experiment

## Objective

Evaluate how the PatchCore coreset sampling ratio affects anomaly-detection quality, localization quality, runtime, and memory-bank size while holding the remaining configuration fixed.

## Controlled design

- **Category:** `{category}`
- **Controlled variable:** coreset sampling ratio
- **Ratios:** {format_ratio_list(table['coreset_sampling_ratio'].tolist())}
- **Backbone:** ResNet-18
- **Feature layers:** `layer2`, `layer3`
- **Image size:** 224 × 224
- **Nearest neighbours:** 5
- **Execution:** CPU
- **Primary diagnostic metric:** pixel F1, used here to assess localization sensitivity

## Results

{markdown_table(table)}

![Metric sensitivity](../results/patchcore_coreset_experiment/{category}/coreset_metrics.png)

![Runtime sensitivity](../results/patchcore_coreset_experiment/{category}/coreset_efficiency.png)
{memory_plot}
## Interpretation

{interpretation}

## Reproduction

```bash
python -u -m scripts.run_patchcore_coreset_experiment \\
  --category {category} \\
  --ratios {' '.join(format_ratio(value) for value in table['coreset_sampling_ratio'].tolist())}

python -m scripts.finalize_patchcore_coreset_experiment \\
  --category {category}
```

## Limitations

- The experiment covers one MVTec AD category and three coreset ratios.
- Runtime measurements are hardware- and system-load-dependent.
- The same test set is reused across configurations, so this is a sensitivity comparison rather than independent model selection and confirmation.
- Results should not be generalized to every MVTec AD category or industrial imaging system without additional validation.
"""


def build_readme_section(table: pd.DataFrame, summary: dict[str, Any]) -> str:
    category = summary["category"]
    efficient = float(summary["efficiency_oriented_ratio"])
    best_pixel = summary["best_by_metric"]["pixel_f1"]
    return f"""{README_START}
## Controlled PatchCore experiment

A controlled sensitivity experiment on `{category}` varied only the PatchCore coreset sampling ratio (`{', '.join(format_ratio(value) for value in table['coreset_sampling_ratio'].tolist())}`) while keeping the backbone, feature layers, image size, neighbour count, and CPU execution fixed.

- Highest observed pixel F1: **{best_pixel['score']:.4f}** at ratio(s) {format_ratio_list(best_pixel['ratios'])}
- Efficiency-oriented ratio under the documented {summary['efficiency_tolerance']:.3f} pixel-F1 tolerance: **`{format_ratio(efficient)}`**
- Detailed report: [`docs/patchcore_coreset_sensitivity.md`](docs/patchcore_coreset_sensitivity.md)
- Comparison data: [`results/patchcore_coreset_experiment/{category}/coreset_comparison.csv`](results/patchcore_coreset_experiment/{category}/coreset_comparison.csv)

The result is interpreted as a category-specific performance–efficiency trade-off, not a universal PatchCore optimum.
{README_END}"""


def update_marked_section(readme: str, section: str) -> str:
    start_count = readme.count(README_START)
    end_count = readme.count(README_END)
    if start_count != end_count or start_count > 1:
        raise ValueError("README contains malformed coreset-experiment markers.")

    if start_count == 1:
        start = readme.index(README_START)
        end = readme.index(README_END, start) + len(README_END)
        prefix = readme[:start].rstrip()
        suffix = readme[end:].strip()
        if suffix:
            return prefix + "\n\n" + section + "\n\n" + suffix + "\n"
        return prefix + "\n\n" + section + "\n"

    anchor = "<!-- MULTICATEGORY_PATCHCORE_END -->"
    if anchor in readme:
        position = readme.index(anchor) + len(anchor)
        return readme[:position].rstrip() + "\n\n" + section + "\n\n" + readme[position:].lstrip("\n")

    roadmap = "# Experiment Roadmap"
    if roadmap in readme:
        position = readme.index(roadmap)
        return readme[:position].rstrip() + "\n\n" + section + "\n\n" + readme[position:]

    return readme.rstrip() + "\n\n" + section + "\n"


def write_outputs(
    table: pd.DataFrame,
    summary: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Path]:
    category = str(summary["category"])
    result_dir = Path(project_root) / "results" / "patchcore_coreset_experiment" / category
    docs_dir = Path(project_root) / "docs"
    readme_path = Path(project_root) / "README.md"
    summary_path = result_dir / "experiment_summary.json"
    report_path = docs_dir / "patchcore_coreset_sensitivity.md"

    if not readme_path.exists():
        raise FileNotFoundError(f"README not found: {readme_path}")

    result_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(build_report(table, summary), encoding="utf-8")

    readme = readme_path.read_text(encoding="utf-8")
    updated = update_marked_section(readme, build_readme_section(table, summary))
    readme_path.write_text(updated, encoding="utf-8")

    return {"summary": summary_path, "report": report_path, "readme": readme_path}


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize the controlled PatchCore coreset-ratio experiment."
    )
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    table = load_comparison_table(parsed.category, DEFAULT_RESULTS_ROOT)
    summary = build_summary(table, parsed.category, parsed.tolerance)
    outputs = write_outputs(table, summary, project_root=PROJECT_ROOT)

    print("=" * 76)
    print("PATCHCORE CORESET EXPERIMENT FINALIZED")
    print("=" * 76)
    print(f"Category                   : {parsed.category}")
    print(f"Efficiency-oriented ratio  : {summary['efficiency_oriented_ratio']:g}")
    print(f"Highest pixel F1            : {summary['best_by_metric']['pixel_f1']['score']:.4f}")
    for name, path in outputs.items():
        print(f"{name:26}: {path}")
    print("PASS: Controlled PatchCore experiment analyzed and documented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
