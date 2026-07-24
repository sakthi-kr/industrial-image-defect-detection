from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from scripts.compare_patchcore_categories import extract_metrics
from src.train_patchcore_category import (
    RANDOM_SEED,
    SUPPORTED_CATEGORIES,
    check_dataset,
    create_datamodule,
    create_engine,
    make_json_serializable,
    portable_project_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    PROJECT_ROOT / "results" / "patchcore_coreset_experiment"
)
DEFAULT_CATEGORY = "leather"
DEFAULT_RATIOS = (0.005, 0.01, 0.05)
BACKBONE = "resnet18"
LAYERS = ("layer2", "layer3")
IMAGE_SIZE = (224, 224)
NUM_NEIGHBORS = 5


@dataclass(frozen=True)
class ExperimentRunPaths:
    anomalib_output_dir: Path
    summary_path: Path


def validate_coreset_ratio(value: float) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError(
            "Coreset ratio must be a finite number in the interval "
            "(0, 1]."
        )
    return ratio


def normalize_ratios(values: Sequence[float]) -> list[float]:
    ratios: list[float] = []
    for value in values:
        ratio = validate_coreset_ratio(value)
        if ratio not in ratios:
            ratios.append(ratio)

    if len(ratios) < 2:
        raise ValueError(
            "At least two distinct coreset ratios are required for "
            "a controlled comparison."
        )

    return sorted(ratios)


def ratio_slug(ratio: float) -> str:
    validated = validate_coreset_ratio(ratio)
    text = f"{validated:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def build_experiment_paths(
    category: str,
    ratio: float,
    project_root: Path = PROJECT_ROOT,
) -> ExperimentRunPaths:
    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(
            f"Unsupported category '{category}'. "
            f"Choose from: {', '.join(SUPPORTED_CATEGORIES)}"
        )

    slug = f"ratio_{ratio_slug(ratio)}"
    result_root = (
        Path(project_root)
        / "results"
        / "patchcore_coreset_experiment"
        / category
    )

    return ExperimentRunPaths(
        anomalib_output_dir=(
            Path(project_root)
            / "results"
            / "patchcore_anomalib_experiments"
            / "coreset_ratio"
            / category
            / slug
        ),
        summary_path=result_root / slug / "run_summary.json",
    )


def create_patchcore_model(coreset_ratio: float):
    from anomalib.models import Patchcore

    ratio = validate_coreset_ratio(coreset_ratio)
    pre_processor = Patchcore.configure_pre_processor(
        image_size=IMAGE_SIZE,
        center_crop_size=IMAGE_SIZE,
    )

    return Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        pre_trained=True,
        coreset_sampling_ratio=ratio,
        num_neighbors=NUM_NEIGHBORS,
        pre_processor=pre_processor,
        visualizer=False,
    )


def get_memory_bank_metadata(model: Any) -> dict[str, int | None]:
    memory_bank = getattr(model, "memory_bank", None)
    if memory_bank is None:
        inner_model = getattr(model, "model", None)
        memory_bank = getattr(inner_model, "memory_bank", None)

    shape = getattr(memory_bank, "shape", None)

    if shape is None or len(shape) == 0:
        return {
            "memory_bank_embeddings": None,
            "embedding_dimension": None,
        }

    return {
        "memory_bank_embeddings": int(shape[0]),
        "embedding_dimension": (
            int(shape[1]) if len(shape) > 1 else None
        ),
    }


def save_run_summary(
    *,
    category: str,
    ratio: float,
    test_results: list[dict],
    engine: Any,
    elapsed_seconds: float,
    model: Any,
    pytorch_version: str,
    cuda_available: bool,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = extract_metrics(test_results)

    summary = {
        "experiment": "PatchCore coreset-ratio sensitivity",
        "controlled_variable": "coreset_sampling_ratio",
        "category": category,
        "model": "Patchcore",
        "backbone": BACKBONE,
        "layers": list(LAYERS),
        "image_size": list(IMAGE_SIZE),
        "coreset_sampling_ratio": float(ratio),
        "num_neighbors": NUM_NEIGHBORS,
        "accelerator": "cpu",
        "random_seed": RANDOM_SEED,
        "pytorch_version": pytorch_version,
        "cuda_available": bool(cuda_available),
        "elapsed_seconds": float(elapsed_seconds),
        **get_memory_bank_metadata(model),
        "best_model_path": portable_project_path(
            getattr(engine, "best_model_path", None)
        ),
        "metrics": metrics,
        "test_results": make_json_serializable(test_results),
        "interpretation_scope": (
            "Single-category controlled experiment. All settings except "
            "the coreset sampling ratio are fixed."
        ),
    }

    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def run_one_configuration(
    category: str,
    ratio: float,
    *,
    overwrite: bool = False,
) -> Path:
    import torch

    check_dataset(category)
    paths = build_experiment_paths(category, ratio)

    if paths.summary_path.exists() and not overwrite:
        print(
            "SKIP: Existing summary found for "
            f"{category}, ratio={ratio:g}: {paths.summary_path}"
        )
        return paths.summary_path

    if paths.anomalib_output_dir.exists():
        shutil.rmtree(paths.anomalib_output_dir)
    if paths.summary_path.parent.exists():
        shutil.rmtree(paths.summary_path.parent)

    torch.manual_seed(RANDOM_SEED)

    print()
    print("=" * 76)
    print("PATCHCORE CORESET-RATIO EXPERIMENT")
    print("=" * 76)
    print(f"Category      : {category}")
    print(f"Coreset ratio : {ratio:g}")
    print(f"Backbone      : {BACKBONE}")
    print(f"Layers        : {', '.join(LAYERS)}")
    print(f"Image size    : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
    print(f"Neighbours    : {NUM_NEIGHBORS}")
    print("Execution     : CPU")

    datamodule = create_datamodule(category)
    model = create_patchcore_model(ratio)
    engine = create_engine(paths.anomalib_output_dir)

    start_time = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
    )
    elapsed_seconds = time.perf_counter() - start_time

    save_run_summary(
        category=category,
        ratio=ratio,
        test_results=test_results,
        engine=engine,
        elapsed_seconds=elapsed_seconds,
        model=model,
        pytorch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        output_path=paths.summary_path,
    )

    print(f"Elapsed time  : {elapsed_seconds:.1f} seconds")
    print(f"Run summary   : {paths.summary_path}")
    return paths.summary_path


def load_experiment_row(
    category: str,
    ratio: float,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    summary_path = build_experiment_paths(
        category,
        ratio,
        project_root=project_root,
    ).summary_path

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Experiment summary not found: {summary_path}"
        )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    expected = {
        "category": category,
        "backbone": BACKBONE,
        "layers": list(LAYERS),
        "image_size": list(IMAGE_SIZE),
        "num_neighbors": NUM_NEIGHBORS,
        "accelerator": "cpu",
    }

    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            raise ValueError(
                f"Configuration mismatch in {summary_path}: "
                f"{key}={summary.get(key)!r}, "
                f"expected {expected_value!r}"
            )

    observed_ratio = float(
        summary.get("coreset_sampling_ratio")
    )
    if not math.isclose(
        observed_ratio,
        ratio,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Ratio mismatch in {summary_path}: "
            f"{observed_ratio!r} != {ratio!r}"
        )

    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        metrics = extract_metrics(
            summary.get("test_results", [])
        )

    return {
        "category": category,
        "coreset_sampling_ratio": observed_ratio,
        "image_auroc": metrics.get("image_auroc"),
        "image_f1": metrics.get("image_f1"),
        "pixel_auroc": metrics.get("pixel_auroc"),
        "pixel_f1": metrics.get("pixel_f1"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "memory_bank_embeddings": summary.get(
            "memory_bank_embeddings"
        ),
        "embedding_dimension": summary.get(
            "embedding_dimension"
        ),
        "summary_path": portable_project_path(summary_path),
    }


def build_experiment_table(
    category: str,
    ratios: Sequence[float],
    *,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    normalized_ratios = normalize_ratios(ratios)
    rows = [
        load_experiment_row(
            category,
            ratio,
            project_root=project_root,
        )
        for ratio in normalized_ratios
    ]

    table = pd.DataFrame(rows)
    metric_columns = [
        "image_auroc",
        "image_f1",
        "pixel_auroc",
        "pixel_f1",
    ]

    missing = [
        column
        for column in metric_columns
        if column not in table.columns
        or table[column].isna().any()
    ]
    if missing:
        raise ValueError(
            f"Missing required experiment metrics: {missing}"
        )

    return table


def best_ratio_by_metric(
    table: pd.DataFrame,
    metric: str,
) -> list[float]:
    maximum = float(table[metric].max())
    return [
        float(value)
        for value in table.loc[
            table[metric].sub(maximum).abs() <= 1e-12,
            "coreset_sampling_ratio",
        ].tolist()
    ]


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


def save_experiment_outputs(
    table: pd.DataFrame,
    category: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Path]:
    output_dir = (
        Path(project_root)
        / "results"
        / "patchcore_coreset_experiment"
        / category
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "coreset_comparison.csv"
    json_path = output_dir / "coreset_comparison.json"
    metric_plot_path = output_dir / "coreset_metrics.png"
    efficiency_plot_path = output_dir / "coreset_efficiency.png"

    table.to_csv(csv_path, index=False)

    metric_columns = [
        "image_auroc",
        "image_f1",
        "pixel_auroc",
        "pixel_f1",
    ]
    payload = {
        "experiment": "PatchCore coreset-ratio sensitivity",
        "category": category,
        "controlled_variable": "coreset_sampling_ratio",
        "fixed_configuration": {
            "backbone": BACKBONE,
            "layers": list(LAYERS),
            "image_size": list(IMAGE_SIZE),
            "num_neighbors": NUM_NEIGHBORS,
            "accelerator": "cpu",
            "random_seed": RANDOM_SEED,
        },
        "ratios": table[
            "coreset_sampling_ratio"
        ].tolist(),
        "best_ratio_by_metric": {
            metric: best_ratio_by_metric(table, metric)
            for metric in metric_columns
        },
        "results": table.to_dict(orient="records"),
        "important_note": (
            "This is a sensitivity experiment on one MVTec AD "
            "category, not a universal hyperparameter optimum."
        ),
    }
    json_path.write_text(
        json.dumps(make_json_safe(payload), indent=2) + "\n",
        encoding="utf-8",
    )

    axis = table.plot(
        x="coreset_sampling_ratio",
        y=metric_columns,
        marker="o",
        figsize=(10, 6),
    )
    axis.set_title(
        f"PatchCore coreset-ratio sensitivity — {category}"
    )
    axis.set_xlabel("Coreset sampling ratio")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.3)
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(
        metric_plot_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    efficiency_columns = ["elapsed_seconds"]
    memory_plot_path = output_dir / "coreset_memory_bank.png"
    if table["memory_bank_embeddings"].notna().any():
        efficiency_columns.append("memory_bank_embeddings")

    for column in efficiency_columns:
        axis = table.plot(
            x="coreset_sampling_ratio",
            y=column,
            marker="o",
            legend=False,
            figsize=(8, 5),
        )
        axis.set_title(
            f"{column.replace('_', ' ').title()} "
            f"versus coreset ratio — {category}"
        )
        axis.set_xlabel("Coreset sampling ratio")
        axis.set_ylabel(column.replace("_", " ").title())
        axis.grid(alpha=0.3)
        figure = axis.get_figure()
        figure.tight_layout()
        target = (
            efficiency_plot_path
            if column == "elapsed_seconds"
            else memory_plot_path
        )
        figure.savefig(
            target,
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(figure)

    outputs = {
        "csv": csv_path,
        "json": json_path,
        "metrics_plot": metric_plot_path,
        "efficiency_plot": efficiency_plot_path,
    }
    if memory_plot_path.exists():
        outputs["memory_plot"] = memory_plot_path
    return outputs


def summarize_experiment(
    category: str,
    ratios: Sequence[float],
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    table = build_experiment_table(
        category,
        ratios,
        project_root=project_root,
    )
    paths = save_experiment_outputs(
        table,
        category,
        project_root=project_root,
    )
    return table, paths


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled PatchCore coreset-ratio experiment "
            "while keeping all other settings fixed."
        )
    )
    parser.add_argument(
        "--category",
        choices=SUPPORTED_CATEGORIES,
        default=DEFAULT_CATEGORY,
        help=(
            "MVTec AD category. Defaults to leather because it had "
            "the lowest pixel F1 in the five-category benchmark."
        ),
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=list(DEFAULT_RATIOS),
        help="Coreset ratios to compare.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the dataset and experiment plan only.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Regenerate comparison outputs from existing summaries.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun configurations even when summaries already exist.",
    )
    return parser.parse_args(arguments)


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    parsed = parse_arguments(arguments)
    ratios = normalize_ratios(parsed.ratios)
    category_path = check_dataset(parsed.category)

    print(f"Dataset validated: {category_path}")
    print(
        "Controlled variable: coreset_sampling_ratio = "
        + ", ".join(f"{value:g}" for value in ratios)
    )
    print(
        "Fixed settings: resnet18, layer2/layer3, "
        "224x224, 5 neighbours, CPU"
    )

    if parsed.check_only:
        for ratio in ratios:
            paths = build_experiment_paths(
                parsed.category,
                ratio,
            )
            print(
                f"  ratio={ratio:g} -> "
                f"{paths.summary_path}"
            )
        return 0

    if not parsed.summarize_only:
        for ratio in ratios:
            run_one_configuration(
                parsed.category,
                ratio,
                overwrite=parsed.overwrite,
            )

    table, paths = summarize_experiment(
        parsed.category,
        ratios,
    )

    print()
    print("=" * 90)
    print("CORESET-RATIO COMPARISON")
    print("=" * 90)
    print(
        table[
            [
                "coreset_sampling_ratio",
                "image_auroc",
                "image_f1",
                "pixel_auroc",
                "pixel_f1",
                "elapsed_seconds",
                "memory_bank_embeddings",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print()
    for name, path in paths.items():
        print(f"{name:16}: {path}")
    print()
    print(
        "PASS: Controlled PatchCore coreset-ratio "
        "experiment completed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
