from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_ad"
RESULTS_ROOT = PROJECT_ROOT / "results"
RANDOM_SEED = 42

SUPPORTED_CATEGORIES = (
    "bottle",
    "cable",
    "leather",
    "metal_nut",
    "tile",
)


@dataclass(frozen=True)
class CategoryRunPaths:
    anomalib_output_dir: Path
    summary_path: Path


def validate_category(category: str) -> str:
    normalized = category.strip().lower()

    if normalized not in SUPPORTED_CATEGORIES:
        raise ValueError(
            f"Unsupported category '{category}'. "
            f"Choose from: {', '.join(SUPPORTED_CATEGORIES)}"
        )

    return normalized


def build_run_paths(
    category: str,
    project_root: Path = PROJECT_ROOT,
) -> CategoryRunPaths:
    normalized = validate_category(category)
    results_root = Path(project_root) / "results"

    return CategoryRunPaths(
        anomalib_output_dir=(
            results_root
            / "patchcore_anomalib"
            / normalized
        ),
        summary_path=(
            results_root
            / "patchcore_multicategory"
            / normalized
            / "run_summary.json"
        ),
    )


def check_dataset(
    category: str,
    data_root: Path = DATA_ROOT,
) -> Path:
    normalized = validate_category(category)
    category_path = Path(data_root) / normalized

    required_paths = (
        category_path / "train" / "good",
        category_path / "test" / "good",
        category_path / "ground_truth",
    )

    missing_paths = [
        path
        for path in required_paths
        if not path.is_dir()
    ]

    if missing_paths:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing_paths
        )
        raise FileNotFoundError(
            f"MVTec AD category '{normalized}' is incomplete.\n"
            f"Missing required folders:\n{formatted}"
        )

    return category_path


def make_json_serializable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()

    if hasattr(value, "numel") and callable(value.numel):
        if value.numel() == 1:
            return value.item()
        return value.tolist()

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    return value


def portable_project_path(path_value: str | Path | None) -> str | None:
    if not path_value:
        return None

    path = Path(path_value).resolve()
    project_root = PROJECT_ROOT.resolve()

    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def create_datamodule(category: str):
    from anomalib.data import MVTecAD

    normalized = validate_category(category)

    return MVTecAD(
        root=DATA_ROOT,
        category=normalized,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        seed=RANDOM_SEED,
    )


def create_patchcore_model():
    from anomalib.models import Patchcore

    pre_processor = Patchcore.configure_pre_processor(
        image_size=(224, 224),
        center_crop_size=(224, 224),
    )

    return Patchcore(
        backbone="resnet18",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.01,
        num_neighbors=5,
        pre_processor=pre_processor,
        visualizer=False,
    )


def create_engine(output_dir: Path):
    from anomalib.engine import Engine

    return Engine(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        default_root_dir=output_dir,
        num_sanity_val_steps=0,
    )


def save_run_summary(
    *,
    category: str,
    test_results: list[dict],
    engine: Any,
    elapsed_seconds: float,
    pytorch_version: str,
    cuda_available: bool,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "model": "Patchcore",
        "category": validate_category(category),
        "backbone": "resnet18",
        "layers": ["layer2", "layer3"],
        "image_size": [224, 224],
        "coreset_sampling_ratio": 0.01,
        "num_neighbors": 5,
        "accelerator": "cpu",
        "pytorch_version": pytorch_version,
        "cuda_available": bool(cuda_available),
        "elapsed_seconds": float(elapsed_seconds),
        "best_model_path": portable_project_path(
            getattr(engine, "best_model_path", None)
        ),
        "test_results": make_json_serializable(test_results),
        "run_type": (
            "Fixed-configuration multi-category PatchCore benchmark. "
            "The same model settings are used for every category."
        ),
    }

    output_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def run_category(category: str) -> Path:
    import torch

    normalized = validate_category(category)
    category_path = check_dataset(normalized)
    run_paths = build_run_paths(normalized)

    torch.manual_seed(RANDOM_SEED)

    print("PatchCore category run")
    print("======================")
    print(f"Dataset path : {category_path}")
    print(f"Category     : {normalized}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    print("Backbone     : resnet18")
    print("Layers       : layer2, layer3")
    print("Image size   : 224 x 224")
    print("Coreset ratio: 0.01")
    print("Neighbours   : 5")

    datamodule = create_datamodule(normalized)
    model = create_patchcore_model()
    engine = create_engine(
        run_paths.anomalib_output_dir
    )

    start_time = time.perf_counter()

    print("\nBuilding the PatchCore memory bank...")
    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\nEvaluating MVTec AD test images...")
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    print("\nTest results:")
    for result in test_results:
        print(
            make_json_serializable(result)
        )

    save_run_summary(
        category=normalized,
        test_results=test_results,
        engine=engine,
        elapsed_seconds=elapsed_seconds,
        pytorch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        output_path=run_paths.summary_path,
    )

    print(
        f"\nElapsed time : {elapsed_seconds:.1f} seconds"
    )
    print(
        f"Run summary  : {run_paths.summary_path}"
    )
    print("PatchCore category run completed successfully.")

    return run_paths.summary_path


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the fixed PatchCore "
            "configuration for one MVTec AD category."
        )
    )

    parser.add_argument(
        "--category",
        required=True,
        choices=SUPPORTED_CATEGORIES,
        help="MVTec AD category to evaluate.",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate the category folder structure "
            "without training PatchCore."
        ),
    )

    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    category_path = check_dataset(parsed.category)

    if parsed.check_only:
        print(
            "PASS: Dataset structure is valid for "
            f"'{parsed.category}': {category_path}"
        )
        return 0

    run_category(parsed.category)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
