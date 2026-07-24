from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "patchcore_multicategory"
SUPPORTED_CATEGORIES = (
    "bottle",
    "cable",
    "leather",
    "metal_nut",
    "tile",
)

METRIC_ALIASES = {
    "image_auroc": {
        "imageauroc",
        "imageaurocmetric",
        "aurocimage",
    },
    "image_f1": {
        "imagef1",
        "imagef1score",
        "f1image",
        "f1scoreimage",
    },
    "pixel_auroc": {
        "pixelauroc",
        "pixelaurocmetric",
        "aurocpixel",
    },
    "pixel_f1": {
        "pixelf1",
        "pixelf1score",
        "f1pixel",
        "f1scorepixel",
    },
}

CONFIG_FIELDS = (
    "model",
    "backbone",
    "layers",
    "image_size",
    "coreset_sampling_ratio",
    "num_neighbors",
    "accelerator",
)


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    if hasattr(value, "item") and callable(value.item):
        try:
            number = float(value.item())
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    return None


def iter_named_values(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from iter_named_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_named_values(item)


def extract_metrics(test_results: Any) -> dict[str, float | None]:
    observed: dict[str, float | None] = {
        metric_name: None
        for metric_name in METRIC_ALIASES
    }

    for key, value in iter_named_values(test_results):
        normalized_key = normalize_key(key)
        numeric_value = to_float(value)

        if numeric_value is None:
            continue

        for metric_name, aliases in METRIC_ALIASES.items():
            if (
                observed[metric_name] is None
                and normalized_key in aliases
            ):
                observed[metric_name] = numeric_value

    return observed


def portable_project_path(path: Path) -> str:
    """Return a project-relative path when possible.

    Test fixtures and external result directories may live outside the
    repository. In those cases, preserve a normalized absolute path instead
    of raising ValueError.
    """
    resolved_path = Path(path).resolve()
    resolved_root = PROJECT_ROOT.resolve()

    try:
        return resolved_path.relative_to(
            resolved_root
        ).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def load_category_summary(
    category: str,
    results_root: Path = RESULTS_ROOT,
) -> dict[str, Any]:
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(
            f"Unsupported category '{category}'. "
            f"Choose from: {', '.join(SUPPORTED_CATEGORIES)}"
        )

    summary_path = (
        Path(results_root)
        / category
        / "run_summary.json"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Run summary not found for '{category}': "
            f"{summary_path}"
        )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    if summary.get("category") != category:
        raise ValueError(
            f"Category mismatch in {summary_path}: "
            f"expected '{category}', received "
            f"'{summary.get('category')}'"
        )

    metrics = extract_metrics(
        summary.get("test_results", [])
    )

    return {
        "category": category,
        "model": summary.get("model"),
        "backbone": summary.get("backbone"),
        "layers": summary.get("layers"),
        "image_size": summary.get("image_size"),
        "coreset_sampling_ratio": summary.get(
            "coreset_sampling_ratio"
        ),
        "num_neighbors": summary.get("num_neighbors"),
        "accelerator": summary.get("accelerator"),
        "elapsed_seconds": to_float(
            summary.get("elapsed_seconds")
        ),
        **metrics,
        "summary_path": portable_project_path(
            summary_path
        ),
    }


def validate_fixed_configuration(
    rows: list[dict[str, Any]],
) -> None:
    if len(rows) < 2:
        raise ValueError(
            "At least two completed categories are required "
            "for comparison."
        )

    reference = rows[0]

    for row in rows[1:]:
        mismatches = [
            field
            for field in CONFIG_FIELDS
            if row.get(field) != reference.get(field)
        ]

        if mismatches:
            raise ValueError(
                "PatchCore configuration mismatch between "
                f"'{reference['category']}' and "
                f"'{row['category']}': {mismatches}"
            )


def build_comparison_table(
    categories: Sequence[str],
    results_root: Path = RESULTS_ROOT,
) -> pd.DataFrame:
    normalized_categories = list(dict.fromkeys(categories))

    rows = [
        load_category_summary(
            category,
            results_root=results_root,
        )
        for category in normalized_categories
    ]

    validate_fixed_configuration(rows)

    table = pd.DataFrame(rows)

    column_order = [
        "category",
        "image_auroc",
        "image_f1",
        "pixel_auroc",
        "pixel_f1",
        "elapsed_seconds",
        "model",
        "backbone",
        "layers",
        "image_size",
        "coreset_sampling_ratio",
        "num_neighbors",
        "accelerator",
        "summary_path",
    ]

    return table[column_order]


def save_comparison_plot(
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    metric_columns = [
        "image_auroc",
        "image_f1",
        "pixel_auroc",
        "pixel_f1",
    ]

    available_metrics = [
        column
        for column in metric_columns
        if table[column].notna().any()
    ]

    if not available_metrics:
        raise ValueError(
            "No supported PatchCore metrics were found "
            "in the completed run summaries."
        )

    plot_table = (
        table.set_index("category")[available_metrics]
    )

    axis = plot_table.plot(
        kind="bar",
        figsize=(11, 6),
    )

    axis.set_title(
        "Fixed-configuration PatchCore comparison"
    )
    axis.set_xlabel("MVTec AD category")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.05)
    axis.tick_params(axis="x", rotation=0)
    axis.grid(axis="y", alpha=0.3)

    figure = axis.get_figure()
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


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
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    return str(value)


def save_outputs(
    table: pd.DataFrame,
    output_dir: Path = RESULTS_ROOT,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "category_comparison.csv"
    json_path = output_dir / "category_comparison.json"
    plot_path = output_dir / "category_comparison.png"

    table.to_csv(csv_path, index=False)

    json_payload = {
        "completed_categories": table[
            "category"
        ].tolist(),
        "n_completed_categories": int(len(table)),
        "fixed_configuration": make_json_safe({
            field: table.iloc[0][field]
            for field in CONFIG_FIELDS
        }),
        "category_results": make_json_safe(
            table.to_dict(orient="records")
        ),
        "macro_average": {
            metric: (
                None
                if table[metric].dropna().empty
                else float(table[metric].mean())
            )
            for metric in (
                "image_auroc",
                "image_f1",
                "pixel_auroc",
                "pixel_f1",
            )
        },
    }

    json_path.write_text(
        json.dumps(json_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    save_comparison_plot(table, plot_path)

    return {
        "csv": csv_path,
        "json": json_path,
        "plot": plot_path,
    }


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate completed fixed-configuration "
            "PatchCore category runs."
        )
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        choices=SUPPORTED_CATEGORIES,
        required=True,
        help="Completed MVTec AD categories to compare.",
    )

    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)

    table = build_comparison_table(
        parsed.categories
    )
    paths = save_outputs(table)

    print("=" * 88)
    print("PATCHCORE CATEGORY COMPARISON")
    print("=" * 88)
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
    print(f"Saved CSV  : {paths['csv']}")
    print(f"Saved JSON : {paths['json']}")
    print(f"Saved plot : {paths['plot']}")
    print()
    print(
        "PASS: Completed PatchCore categories were "
        "standardized and compared."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
