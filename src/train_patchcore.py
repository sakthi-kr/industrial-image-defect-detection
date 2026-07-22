from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_ad"
CATEGORY = "bottle"

ANOMALIB_OUTPUT_DIR = PROJECT_ROOT / "results" / "patchcore_anomalib"
METRICS_OUTPUT_PATH = PROJECT_ROOT / "results" / "patchcore_smoke_metrics.json"

RANDOM_SEED = 42


def make_json_serializable(value: Any) -> Any:
    """
    Convert tensors, paths, nested mappings, and sequences into values that
    can be written to JSON.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()

        if value.numel() == 1:
            return value.item()

        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


def check_dataset() -> None:
    """
    Confirm that the local MVTec AD bottle category exists.
    """
    category_path = DATA_ROOT / CATEGORY

    if not category_path.exists():
        raise FileNotFoundError(
            f"MVTec AD category not found: {category_path}"
        )

    required_paths = [
        category_path / "train" / "good",
        category_path / "test" / "good",
        category_path / "ground_truth",
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        formatted_paths = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Required MVTec AD folders are missing:\n"
            f"{formatted_paths}"
        )


def create_datamodule() -> MVTecAD:
    """
    Create a low-memory MVTec AD datamodule.
    """
    return MVTecAD(
        root=DATA_ROOT,
        category=CATEGORY,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        seed=RANDOM_SEED,
    )


def create_patchcore_model() -> Patchcore:
    """
    Create a lightweight PatchCore configuration for the first CPU run.
    """
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


def create_engine() -> Engine:
    """
    Create a CPU-only Anomalib engine.
    """
    return Engine(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        default_root_dir=ANOMALIB_OUTPUT_DIR,
        num_sanity_val_steps=0,
    )


def save_run_summary(
    test_results: list[dict],
    engine: Engine,
    elapsed_seconds: float,
) -> None:
    """
    Save run configuration, runtime, checkpoint location, and test metrics.
    """
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": "Patchcore",
        "category": CATEGORY,
        "backbone": "resnet18",
        "layers": ["layer2", "layer3"],
        "image_size": [224, 224],
        "coreset_sampling_ratio": 0.01,
        "num_neighbors": 5,
        "accelerator": "cpu",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "elapsed_seconds": elapsed_seconds,
        "best_model_path": (
            str(Path(engine.best_model_path).resolve().relative_to(PROJECT_ROOT.resolve()))
            if engine.best_model_path
            else None
        ),
        "test_results": make_json_serializable(test_results),
        "run_type": (
            "Initial low-memory PatchCore run. Results are intended to verify "
            "the full anomaly-detection pipeline before final tuning."
        ),
    }

    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"\nSaved summary to: {METRICS_OUTPUT_PATH}")


def main() -> None:
    torch.manual_seed(RANDOM_SEED)

    check_dataset()

    print("PatchCore CPU run")
    print("=================")
    print(f"Dataset root : {DATA_ROOT}")
    print(f"Category     : {CATEGORY}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    print("Backbone     : resnet18")
    print("Image size   : 224 x 224")
    print("Coreset ratio: 0.01")

    datamodule = create_datamodule()
    model = create_patchcore_model()
    engine = create_engine()

    start_time = time.perf_counter()

    print("\nBuilding the PatchCore memory bank...")
    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\nEvaluating on MVTec AD test images...")
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    elapsed_seconds = time.perf_counter() - start_time

    print("\nTest results:")
    for result in test_results:
        print(make_json_serializable(result))

    print(f"\nTotal elapsed time: {elapsed_seconds:.1f} seconds")
    print(f"Best model path   : {engine.best_model_path}")

    save_run_summary(
        test_results=test_results,
        engine=engine,
        elapsed_seconds=elapsed_seconds,
    )

    print("\nPatchCore run completed successfully.")


if __name__ == "__main__":
    main()