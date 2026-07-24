from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = PROJECT_ROOT / "results" / "patchcore_predictions.csv"
DEFAULT_CATEGORY_COMPARISON = (
    PROJECT_ROOT
    / "results"
    / "patchcore_multicategory"
    / "category_comparison.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "patchcore_failure_analysis" / "bottle"
)
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "patchcore_failure_analysis.md"
DEFAULT_README = PROJECT_ROOT / "README.md"
DEFAULT_THRESHOLD = 0.5
DEFAULT_REVIEW_MARGIN = 0.05

README_START = "<!-- PATCHCORE_FAILURE_ANALYSIS_START -->"
README_END = "<!-- PATCHCORE_FAILURE_ANALYSIS_END -->"

REQUIRED_PREDICTION_COLUMNS = {
    "image_path",
    "file_name",
    "defect_type",
    "true_label",
    "true_numeric",
    "predicted_label",
    "predicted_numeric",
    "pred_score",
    "correct",
}

REQUIRED_CATEGORY_COLUMNS = {
    "category",
    "image_auroc",
    "image_f1",
    "pixel_auroc",
    "pixel_f1",
}


def read_prediction_table(path: Path) -> pd.DataFrame:
    """Load and validate the saved image-level PatchCore predictions."""
    table = pd.read_csv(path)
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(table.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    if table.empty:
        raise ValueError("Prediction table is empty.")

    numeric_columns = [
        "true_numeric",
        "predicted_numeric",
        "pred_score",
    ]
    for column in numeric_columns:
        table[column] = pd.to_numeric(table[column], errors="raise")

    if table["pred_score"].isna().any():
        raise ValueError("Prediction scores contain missing values.")
    if not table["true_numeric"].isin([0, 1]).all():
        raise ValueError("true_numeric must contain only 0 and 1.")
    if not table["predicted_numeric"].isin([0, 1]).all():
        raise ValueError("predicted_numeric must contain only 0 and 1.")

    table["correct"] = (
        table["true_numeric"].astype(int)
        == table["predicted_numeric"].astype(int)
    )
    return table.copy()


def build_defect_type_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize errors and score distributions for every defect type."""
    rows: list[dict[str, float | int | str]] = []
    for defect_type, group in table.groupby("defect_type", sort=True):
        false_positives = int(
            ((group["true_numeric"] == 0) & (group["predicted_numeric"] == 1)).sum()
        )
        false_negatives = int(
            ((group["true_numeric"] == 1) & (group["predicted_numeric"] == 0)).sum()
        )
        rows.append(
            {
                "defect_type": str(defect_type),
                "n_images": int(len(group)),
                "n_correct": int(group["correct"].sum()),
                "correct_rate": float(group["correct"].mean()),
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "score_min": float(group["pred_score"].min()),
                "score_mean": float(group["pred_score"].mean()),
                "score_max": float(group["pred_score"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("defect_type").reset_index(drop=True)


def build_borderline_cases(
    table: pd.DataFrame,
    *,
    threshold: float,
    review_margin: float,
) -> pd.DataFrame:
    """Return cases nearest to the classification threshold."""
    if review_margin <= 0:
        raise ValueError("review_margin must be positive.")
    result = table.copy()
    result["score_margin"] = result["pred_score"] - float(threshold)
    result["absolute_margin"] = result["score_margin"].abs()
    result["inside_review_band"] = result["absolute_margin"] <= float(review_margin)
    columns = [
        "image_path",
        "file_name",
        "defect_type",
        "true_label",
        "predicted_label",
        "pred_score",
        "score_margin",
        "absolute_margin",
        "correct",
        "inside_review_band",
    ]
    return result.sort_values(
        ["absolute_margin", "pred_score", "image_path"],
        ascending=[True, True, True],
    )[columns].reset_index(drop=True)


def read_category_comparison(path: Path) -> pd.DataFrame:
    """Load and validate the five-category benchmark table."""
    table = pd.read_csv(path)
    missing = sorted(REQUIRED_CATEGORY_COLUMNS - set(table.columns))
    if missing:
        raise ValueError(f"Category comparison is missing columns: {missing}")
    if table.empty:
        raise ValueError("Category comparison table is empty.")
    return table.copy()


def build_localization_gap(table: pd.DataFrame) -> pd.DataFrame:
    """Quantify the gap between image detection and pixel localization."""
    result = table.copy()
    result["image_pixel_f1_gap"] = result["image_f1"] - result["pixel_f1"]
    result["image_pixel_auroc_gap"] = result["image_auroc"] - result["pixel_auroc"]
    columns = [
        "category",
        "image_auroc",
        "pixel_auroc",
        "image_pixel_auroc_gap",
        "image_f1",
        "pixel_f1",
        "image_pixel_f1_gap",
    ]
    return result[columns].sort_values(
        ["image_pixel_f1_gap", "category"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _case_records(table: pd.DataFrame, mask: pd.Series, threshold: float) -> list[dict]:
    cases = table.loc[mask].copy()
    cases["score_margin"] = cases["pred_score"] - float(threshold)
    records: list[dict] = []
    for row in cases.sort_values("pred_score").to_dict(orient="records"):
        records.append(
            {
                "image_path": str(row["image_path"]).replace("\\", "/"),
                "defect_type": str(row["defect_type"]),
                "pred_score": float(row["pred_score"]),
                "score_margin": float(row["score_margin"]),
            }
        )
    return records


def build_analysis_summary(
    predictions: pd.DataFrame,
    borderline: pd.DataFrame,
    defect_summary: pd.DataFrame,
    localization_gap: pd.DataFrame,
    *,
    threshold: float,
    review_margin: float,
) -> dict:
    """Create a JSON-serializable failure-analysis summary."""
    false_positive_mask = (
        (predictions["true_numeric"] == 0)
        & (predictions["predicted_numeric"] == 1)
    )
    false_negative_mask = (
        (predictions["true_numeric"] == 1)
        & (predictions["predicted_numeric"] == 0)
    )
    errors = predictions["true_numeric"] != predictions["predicted_numeric"]
    largest_gap_row = localization_gap.iloc[0]
    lowest_pixel_row = localization_gap.sort_values(
        ["pixel_f1", "category"], ascending=[True, True]
    ).iloc[0]

    defective_only = defect_summary[defect_summary["defect_type"] != "good"]
    if defective_only.empty:
        hardest_defect_types: list[str] = []
    else:
        minimum_rate = float(defective_only["correct_rate"].min())
        hardest_defect_types = sorted(
            defective_only.loc[
                np.isclose(defective_only["correct_rate"], minimum_rate),
                "defect_type",
            ].astype(str).tolist()
        )

    return {
        "analysis": "PatchCore failure and localization analysis",
        "category": "bottle",
        "decision_threshold": float(threshold),
        "illustrative_review_margin": float(review_margin),
        "n_test_images": int(len(predictions)),
        "n_correct": int((~errors).sum()),
        "n_errors": int(errors.sum()),
        "false_positives": int(false_positive_mask.sum()),
        "false_negatives": int(false_negative_mask.sum()),
        "false_positive_cases": _case_records(
            predictions, false_positive_mask, threshold
        ),
        "false_negative_cases": _case_records(
            predictions, false_negative_mask, threshold
        ),
        "n_inside_illustrative_review_band": int(
            borderline["inside_review_band"].sum()
        ),
        "hardest_bottle_defect_types_by_correct_rate": hardest_defect_types,
        "localization": {
            "largest_image_pixel_f1_gap_category": str(largest_gap_row["category"]),
            "largest_image_pixel_f1_gap": float(
                largest_gap_row["image_pixel_f1_gap"]
            ),
            "lowest_pixel_f1_category": str(lowest_pixel_row["category"]),
            "lowest_pixel_f1": float(lowest_pixel_row["pixel_f1"]),
        },
        "interpretation_limits": [
            "The review band is descriptive and has not been tuned on an independent validation set.",
            "The bottle error analysis uses one benchmark test set and should not be treated as a production error rate.",
            "Image-level detection and pixel-level localization measure different capabilities.",
        ],
    }


def resolve_image_path(raw_path: str, project_root: Path) -> Path:
    """Resolve project-relative paths saved with Windows or POSIX separators."""
    normalized = str(raw_path).replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate
    return Path(project_root) / candidate


def load_ground_truth_mask(image_path: Path, defect_type: str) -> Image.Image:
    """Load a MVTec mask, or return a blank mask for a normal image."""
    if defect_type == "good":
        with Image.open(image_path) as image:
            return Image.new("L", image.size, color=0)
    category_root = image_path.parents[2]
    mask_path = (
        category_root
        / "ground_truth"
        / defect_type
        / f"{image_path.stem}_mask.png"
    )
    if not mask_path.exists():
        raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")
    return Image.open(mask_path).convert("L")


def save_failure_case_figure(
    predictions: pd.DataFrame,
    *,
    output_path: Path,
    project_root: Path,
    threshold: float,
) -> None:
    """Visualize every image-level false positive and false negative."""
    errors = predictions[
        predictions["true_numeric"] != predictions["predicted_numeric"]
    ].copy()
    if errors.empty:
        raise ValueError("No image-level errors are available to visualize.")

    fig, axes = plt.subplots(len(errors), 2, figsize=(10, 4 * len(errors)))
    if len(errors) == 1:
        axes = np.asarray([axes])

    for row_index, (_, row) in enumerate(errors.iterrows()):
        image_path = resolve_image_path(str(row["image_path"]), project_root)
        if not image_path.exists():
            raise FileNotFoundError(f"Test image not found: {image_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            axes[row_index, 0].imshow(image)
        error_kind = "False positive" if int(row["true_numeric"]) == 0 else "False negative"
        axes[row_index, 0].set_title(
            f"{error_kind}: {row['defect_type']}/{row['file_name']}\n"
            f"true={row['true_label']}, pred={row['predicted_label']}, "
            f"score={float(row['pred_score']):.4f}, threshold={threshold:.2f}"
        )
        axes[row_index, 0].axis("off")

        mask = load_ground_truth_mask(image_path, str(row["defect_type"]))
        axes[row_index, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 1].set_title(
            "Ground-truth mask" if str(row["defect_type"]) != "good" else "Normal image: no defect mask"
        )
        axes[row_index, 1].axis("off")

    fig.suptitle("PatchCore image-level failure cases — MVTec AD bottle", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_score_margin_plot(
    predictions: pd.DataFrame,
    *,
    output_path: Path,
    threshold: float,
    review_margin: float,
) -> None:
    """Plot image scores by defect type and emphasize misclassifications."""
    order = [
        item
        for item in ["good", "broken_large", "broken_small", "contamination"]
        if item in set(predictions["defect_type"])
    ]
    extras = sorted(set(predictions["defect_type"]) - set(order))
    order.extend(extras)
    positions = {name: index for index, name in enumerate(order)}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, row in predictions.iterrows():
        x = positions[str(row["defect_type"])]
        marker = "x" if not bool(row["correct"]) else "o"
        size = 80 if not bool(row["correct"]) else 28
        ax.scatter(x, float(row["pred_score"]), marker=marker, s=size, alpha=0.8)

    ax.axhline(threshold, linestyle="--", label=f"Decision threshold ({threshold:.2f})")
    ax.axhspan(
        threshold - review_margin,
        threshold + review_margin,
        alpha=0.12,
        label=f"Illustrative ±{review_margin:.2f} review band",
    )
    ax.set_xticks(range(len(order)), [item.replace("_", " ") for item in order])
    ax.set_ylabel("PatchCore anomaly score")
    ax.set_title("Bottle anomaly scores and image-level errors")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_localization_gap_plot(gaps: pd.DataFrame, *, output_path: Path) -> None:
    """Plot the image-F1 minus pixel-F1 gap for each benchmark category."""
    ordered = gaps.sort_values("image_pixel_f1_gap", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ordered["category"], ordered["image_pixel_f1_gap"])
    ax.set_ylabel("Image F1 − pixel F1")
    ax.set_title("Detection–localization gap across MVTec AD categories")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    """Render selected dataframe columns without requiring tabulate."""
    selected = frame[list(columns)].copy()
    headers = [str(column) for column in selected.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in selected.iterrows():
        values: list[str] = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    *,
    path: Path,
    summary: dict,
    defect_summary: pd.DataFrame,
    borderline: pd.DataFrame,
    localization_gap: pd.DataFrame,
) -> None:
    """Write a concise evidence-based failure-analysis report."""
    fp_cases = summary["false_positive_cases"]
    fn_cases = summary["false_negative_cases"]
    fp_text = ", ".join(
        f"`{case['defect_type']}/{Path(case['image_path']).name}` (score {case['pred_score']:.4f})"
        for case in fp_cases
    ) or "none"
    fn_text = ", ".join(
        f"`{case['defect_type']}/{Path(case['image_path']).name}` (score {case['pred_score']:.4f})"
        for case in fn_cases
    ) or "none"

    top_borderline = borderline.head(10).copy()
    report = f"""# PatchCore Failure and Localization Analysis

## Scope

This analysis reuses the saved `bottle` prediction table and the completed five-category benchmark. No model was retrained and the test threshold was not retuned.

## Image-level errors

- Test images: **{summary['n_test_images']}**
- Correct predictions: **{summary['n_correct']}**
- False positives: **{summary['false_positives']}**
- False negatives: **{summary['false_negatives']}**
- Decision threshold used by the saved predictions: **{summary['decision_threshold']:.2f}**

Observed false-positive case(s): {fp_text}.

Observed false-negative case(s): {fn_text}.

![Image-level failure cases](../results/patchcore_failure_analysis/bottle/failure_case_images.png)

The false positive shows that normal appearance variation can cross the decision threshold. The false negative is a contamination example at the threshold, illustrating that subtle anomalies can remain difficult at the selected operating point.

## Performance by bottle defect type

{markdown_table(defect_summary, ['defect_type', 'n_images', 'n_correct', 'correct_rate', 'false_positives', 'false_negatives', 'score_mean'])}

Broken-large and broken-small defects were detected consistently in this run. Contamination was the hardest defective subtype because one contamination image was missed. This is a benchmark observation, not a general production failure rate.

## Borderline cases

The following table lists the ten images closest to the threshold. The ±{summary['illustrative_review_margin']:.2f} band is an **illustrative review band**, not a tuned deployment policy.

{markdown_table(top_borderline, ['defect_type', 'file_name', 'true_label', 'predicted_label', 'pred_score', 'score_margin', 'correct', 'inside_review_band'])}

![Score margins](../results/patchcore_failure_analysis/bottle/score_margin_by_defect_type.png)

A production system could route low-margin cases for manual review, but the width of such a band must be selected using independent validation data and operational costs.

## Localization limitation across categories

{markdown_table(localization_gap, ['category', 'image_f1', 'pixel_f1', 'image_pixel_f1_gap', 'image_auroc', 'pixel_auroc'])}

![Detection-localization gap](../results/patchcore_failure_analysis/bottle/localization_gap.png)

The largest image-F1 versus pixel-F1 gap occurs for **{summary['localization']['largest_image_pixel_f1_gap_category']}** ({summary['localization']['largest_image_pixel_f1_gap']:.4f}). The lowest pixel F1 is also observed for **{summary['localization']['lowest_pixel_f1_category']}** ({summary['localization']['lowest_pixel_f1']:.4f}). This shows that classifying an image as anomalous can be substantially easier than identifying the exact defective pixels.

The existing PatchCore visualization below includes difficult normal and defective examples with anomaly heatmaps and ground-truth masks:

![PatchCore heatmaps](../results/patchcore_example_heatmaps.png)

## Practical conclusions

1. Image-level performance is strong, but threshold-adjacent examples still produce both false alarms and missed defects.
2. Contamination deserves targeted validation because it contains the observed false negative.
3. Pixel-level localization should be evaluated separately from image-level detection; high image AUROC does not imply accurate defect boundaries.
4. Threshold or manual-review policies must be selected on independent validation data, not optimized on this test set.
5. Before industrial use, the model needs robustness testing under lighting, blur, camera-position, and product-batch changes.

## Reproducibility

```bash
python -m scripts.analyze_patchcore_failures
```

Generated outputs are stored under:

```text
results/patchcore_failure_analysis/bottle/
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def update_readme(path: Path, summary: dict) -> None:
    """Insert or replace the marked README failure-analysis section."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    section = f"""{README_START}
## Focused failure analysis

The saved `bottle` predictions contain **{summary['false_positives']} false positive** and **{summary['false_negatives']} false negative**. The missed case is a contamination image at the classification threshold. Across the five-category benchmark, **{summary['localization']['lowest_pixel_f1_category']}** has the lowest pixel F1 ({summary['localization']['lowest_pixel_f1']:.3f}), despite strong image-level detection.

The analysis includes score-margin diagnostics, defect-type performance, representative failure images, and the image-detection versus pixel-localization gap. The ±{summary['illustrative_review_margin']:.2f} review band is illustrative and was not tuned on the test set.

Detailed report: [`docs/patchcore_failure_analysis.md`](docs/patchcore_failure_analysis.md)
{README_END}"""

    if README_START in text and README_END in text:
        before = text.split(README_START, 1)[0].rstrip()
        after = text.split(README_END, 1)[1].lstrip()
        updated = before + "\n\n" + section
        if after:
            updated += "\n\n" + after.rstrip()
    else:
        updated = text.rstrip() + "\n\n" + section
    path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze saved PatchCore errors and localization limitations."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--category-comparison", type=Path, default=DEFAULT_CATEGORY_COMPARISON
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--review-margin", type=float, default=DEFAULT_REVIEW_MARGIN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = read_prediction_table(args.predictions)
    defect_summary = build_defect_type_summary(predictions)
    borderline = build_borderline_cases(
        predictions,
        threshold=args.threshold,
        review_margin=args.review_margin,
    )
    category_table = read_category_comparison(args.category_comparison)
    localization_gap = build_localization_gap(category_table)
    summary = build_analysis_summary(
        predictions,
        borderline,
        defect_summary,
        localization_gap,
        threshold=args.threshold,
        review_margin=args.review_margin,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    defect_summary.to_csv(output_dir / "defect_type_performance.csv", index=False)
    borderline.to_csv(output_dir / "borderline_cases.csv", index=False)
    localization_gap.to_csv(output_dir / "localization_gap.csv", index=False)
    save_json(summary, output_dir / "failure_analysis_summary.json")
    save_failure_case_figure(
        predictions,
        output_path=output_dir / "failure_case_images.png",
        project_root=args.project_root,
        threshold=args.threshold,
    )
    save_score_margin_plot(
        predictions,
        output_path=output_dir / "score_margin_by_defect_type.png",
        threshold=args.threshold,
        review_margin=args.review_margin,
    )
    save_localization_gap_plot(
        localization_gap,
        output_path=output_dir / "localization_gap.png",
    )
    write_report(
        path=args.report,
        summary=summary,
        defect_summary=defect_summary,
        borderline=borderline,
        localization_gap=localization_gap,
    )
    update_readme(args.readme, summary)

    print("PASS: PatchCore failure and localization analysis completed.")
    print(f"Report : {args.report}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
