from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mvtec_ad"
    / "bottle"
    / "test"
    / "broken_small"
    / "000.png"
)

METRICS_PATH = PROJECT_ROOT / "results" / "patchcore_smoke_metrics.json"
ANOMALIB_OUTPUT_DIR = PROJECT_ROOT / "results" / "patchcore_anomalib"


def find_checkpoint() -> Path:
    """
    Find the PatchCore checkpoint created by train_patchcore.py.

    First tries the checkpoint path stored in patchcore_smoke_metrics.json.
    If that does not work, searches the Anomalib output directory.
    """
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        saved_path = metrics.get("best_model_path")

        if saved_path:
            checkpoint_path = Path(saved_path)

            if checkpoint_path.exists():
                return checkpoint_path

    candidates = list(ANOMALIB_OUTPUT_DIR.rglob("model.ckpt"))

    if not candidates:
        raise FileNotFoundError(
            "Could not find a PatchCore checkpoint.\n"
            "Run src/train_patchcore.py first."
        )

    # Select the most recently modified checkpoint.
    return max(candidates, key=lambda path: path.stat().st_mtime)


def create_patchcore_model() -> Patchcore:
    """
    Recreate the same PatchCore configuration used during training.
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


def summarize_value(value: Any) -> str:
    """
    Return a concise description of a prediction attribute.
    """
    if isinstance(value, torch.Tensor):
        value_cpu = value.detach().cpu()

        summary = (
            f"Tensor(shape={tuple(value_cpu.shape)}, "
            f"dtype={value_cpu.dtype}"
        )

        if value_cpu.numel() > 0 and torch.is_floating_point(value_cpu):
            summary += (
                f", min={value_cpu.min().item():.6f}, "
                f"max={value_cpu.max().item():.6f}"
            )

        if value_cpu.numel() <= 10:
            summary += f", values={value_cpu.tolist()}"

        return summary + ")"

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (list, tuple)):
        preview = list(value[:3])
        return (
            f"{type(value).__name__}"
            f"(length={len(value)}, preview={preview})"
        )

    return repr(value)


def inspect_prediction_item(item: Any, item_index: int) -> None:
    """
    Print the important fields returned by Anomalib.
    """
    print(f"\nPrediction item {item_index}")
    print("-" * 40)
    print(f"Object type: {type(item)}")

    attributes = [
        "image_path",
        "pred_score",
        "pred_label",
        "anomaly_map",
        "pred_mask",
        "gt_label",
        "gt_mask",
        "mask_path",
    ]

    for attribute in attributes:
        if hasattr(item, attribute):
            value = getattr(item, attribute)
            print(f"{attribute:12}: {summarize_value(value)}")
        else:
            print(f"{attribute:12}: not available")


def main() -> None:
    if not DEFAULT_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: {DEFAULT_IMAGE_PATH}"
        )

    checkpoint_path = find_checkpoint()

    print("PatchCore prediction inspection")
    print("===============================")
    print(f"Image      : {DEFAULT_IMAGE_PATH}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"PyTorch    : {torch.__version__}")
    print(f"CUDA       : {torch.cuda.is_available()}")

    model = create_patchcore_model()

    engine = Engine(
        accelerator="cpu",
        devices=1,
        logger=False,
        default_root_dir=ANOMALIB_OUTPUT_DIR,
    )

    print("\nRunning prediction...")

    predictions = engine.predict(
        model=model,
        data_path=str(DEFAULT_IMAGE_PATH),
        ckpt_path=str(checkpoint_path),
        return_predictions=True,
    )

    if predictions is None:
        raise RuntimeError("Engine.predict returned None.")

    print("\nPrediction container")
    print("--------------------")
    print(f"Type  : {type(predictions)}")
    print(f"Length: {len(predictions)}")

    for index, prediction_item in enumerate(predictions):
        inspect_prediction_item(prediction_item, index)

    print("\nPrediction inspection completed successfully.")


if __name__ == "__main__":
    main()