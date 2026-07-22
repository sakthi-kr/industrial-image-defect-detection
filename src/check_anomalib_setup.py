from pathlib import Path

import anomalib
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_ad"
CATEGORY = "bottle"


def print_batch_information(batch, batch_name: str) -> None:
    """
    Print basic information about an Anomalib image batch.
    """
    print(f"\n{batch_name}")
    print("-" * len(batch_name))
    print(f"Batch type: {type(batch)}")

    if hasattr(batch, "image"):
        print(f"Image shape: {tuple(batch.image.shape)}")
        print(f"Image dtype: {batch.image.dtype}")
        print(f"Image min  : {batch.image.min().item():.4f}")
        print(f"Image max  : {batch.image.max().item():.4f}")
    else:
        print("No `image` attribute found on batch.")

    if hasattr(batch, "gt_label"):
        print(f"Ground-truth labels: {batch.gt_label}")
    else:
        print("No `gt_label` attribute found on batch.")

    if hasattr(batch, "image_path"):
        print(f"Image paths: {batch.image_path}")


def main() -> None:
    print("Anomalib environment check")
    print("==========================")
    print(f"Anomalib version : {anomalib.__version__}")
    print(f"PyTorch version  : {torch.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    print(f"Dataset root     : {DATA_ROOT}")
    print(f"Category         : {CATEGORY}")

    category_path = DATA_ROOT / CATEGORY

    if not category_path.exists():
        raise FileNotFoundError(
            f"MVTec category folder not found: {category_path}"
        )

    print("\nCreating MVTecAD datamodule...")

    datamodule = MVTecAD(
        root=DATA_ROOT,
        category=CATEGORY,
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
    )

    print("Preparing dataset...")
    datamodule.prepare_data()

    print("Setting up dataset...")
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    print_batch_information(train_batch, "Training batch")
    print_batch_information(test_batch, "Test batch")

    print("\nCreating lightweight PatchCore model...")

    model = Patchcore(
        backbone="resnet18",
        layers=("layer2", "layer3"),
        coreset_sampling_ratio=0.01,
        num_neighbors=9,
    )

    print(f"Model type: {type(model).__name__}")

    print("\nCreating CPU engine...")

    engine = Engine(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
    )

    print(f"Engine type: {type(engine).__name__}")

    print("\nSmoke test completed successfully.")
    print("Dataset loading, PatchCore initialization, and CPU engine setup all work.")


if __name__ == "__main__":
    main()
