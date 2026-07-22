# Dataset Instructions

This project uses the MVTec AD industrial anomaly-detection dataset.

The raw dataset is not included in this repository.

Download the dataset from the official MVTec source and comply with its dataset terms.

## Current Category

The current implementation uses:

```text
bottle
```

## Expected Local Structure

```text
data/
└── raw/
    └── mvtec_ad/
        └── bottle/
            ├── train/
            │   └── good/
            ├── test/
            │   ├── good/
            │   ├── broken_large/
            │   ├── broken_small/
            │   └── contamination/
            └── ground_truth/
                ├── broken_large/
                ├── broken_small/
                └── contamination/
```

## Label Convention

```text
train/good -> normal training image
test/good  -> normal test image
test/*     -> defective test image, except test/good
```

Binary labels used by the project:

```text
good             -> normal
broken_large     -> defective
broken_small     -> defective
contamination    -> defective
```

## Ground-Truth Masks

Defective test images have corresponding masks.

Example:

```text
test/broken_small/000.png
ground_truth/broken_small/000_mask.png
```

Normal images do not have defect masks.

## Dataset Use

### Random Forest Baseline

The supervised development baseline uses manually extracted features from normal and defective images.

### PatchCore

PatchCore uses:

```text
Training:
bottle/train/good
```

and evaluates on:

```text
bottle/test/good
bottle/test/broken_large
bottle/test/broken_small
bottle/test/contamination
```

## Important

Do not commit the dataset to GitHub.

The repository `.gitignore` should exclude:

```text
data/raw/
data/processed/
*.zip
*.tar
*.tar.gz
*.tar.xz
```

Verify before committing:

```bash
git status --short
```

Raw images should not appear in the output.
