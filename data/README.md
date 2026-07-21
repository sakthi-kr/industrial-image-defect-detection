# Dataset Instructions

This project uses the MVTec AD industrial anomaly detection dataset.

The raw dataset is not included in this repository because image datasets can be large and should not be committed to GitHub.

## Dataset Source

Use the official MVTec AD dataset page to download the dataset.

MVTec AD is designed for benchmarking anomaly detection methods in industrial inspection. Each category contains defect-free training images and test images containing both normal and defective examples.

## First Project Category

For the first version of this project, use only:

```text
bottle
```

Starting with one category keeps the first pipeline easier to debug before expanding to more categories.

## Expected Local Folder Structure

After downloading and extracting the dataset, the expected local structure is:

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
```

## Label Convention

For the first baseline:

```text
train/good      -> normal training images
test/good       -> normal test images
test/*          -> defective test images, except test/good
```

The baseline classification label will be:

```text
good      -> normal
not good  -> defective
```

## Important

Do not upload raw dataset files to GitHub.

The `.gitignore` file should exclude:

```text
data/raw/
data/processed/
*.zip
*.tar
*.tar.gz
```

## Planned Dataset Use

Initial version:

- use only the `bottle` category
- build a simple image loading and preprocessing pipeline
- generate a sample image grid
- train a simple baseline model
- evaluate normal vs defective classification

Later versions:

- add more MVTec AD categories
- compare baseline model with anomaly-detection methods
- add PatchCore or PaDiM using Anomalib
- generate anomaly heatmaps
