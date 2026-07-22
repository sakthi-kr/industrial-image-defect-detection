# Industrial Image Defect Detection

## Summary

This project develops an industrial image anomaly-detection pipeline for visual defect detection. The first version will use the MVTec AD `bottle` category to build a reproducible workflow for image loading, preprocessing, baseline modelling, evaluation, and validation.

## Motivation

Industrial inspection often relies on image data to identify defective parts, surface damage, contamination, or production issues. This project explores image-based machine learning as a practical AI/ML workflow for industrial quality control and visual inspection.

## Dataset

Planned dataset:

```text
MVTec AD industrial anomaly detection dataset
```

The dataset is not included in this repository. Download and folder-structure instructions are provided in:

```text
data/README.md
```

The first implementation will use only the `bottle` category.

## Sample Images

The first version uses the MVTec AD `bottle` category, including normal images and several defect types.

![Sample Images](results/sample_images.png)

## Preprocessing Preview

Images are converted to RGB, resized to `128 × 128`, normalized to `[0, 1]`, and converted to grayscale for simple baseline feature extraction.

![Preprocessing Preview](results/preprocessing_preview.png)

## Problem Definition

The first baseline will treat the task as binary classification:

```text
normal image     -> good
defective image  -> anomaly/defect
```

Later versions may use anomaly-detection methods that train only on normal images and detect deviations at test time.

## Planned Methods

Version 1: Simple baseline

- load image paths and labels
- inspect normal and defective image examples
- resize images
- extract simple image features
- train a baseline classifier
- evaluate with classification metrics and confusion matrix

Version 2: Industrial anomaly detection

- use PatchCore or PaDiM
- train using normal images
- predict anomaly scores on test images
- generate anomaly localization heatmaps

Version 3: Testing and validation

- add preprocessing tests
- add prediction-output tests
- document model limitations
- add validation strategy and experiment plan

## Project Structure

```text
industrial-image-defect-detection/
├── data/
│   └── README.md
├── docs/
│   └── model_card.md
├── notebooks/
├── results/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
├── requirements.txt
└── README.md
```

## Current Status

Project skeleton created. Phase 3 begins with dataset setup, image loading, and sample visualization.

## Planned Results

The final project should include:

- dataset summary
- sample normal and defective images
- baseline classification metrics
- confusion matrix
- example predictions
- validation limitations
- model card
- reproducible scripts
- basic tests

## Baseline Results

The first baseline uses simple image statistics, colour features, grayscale histogram features, and edge-based features with a Random Forest classifier.

This is a supervised development baseline for checking the full image-processing and model-training pipeline. It should not be interpreted as the final industrial anomaly-detection model.

### Baseline Confusion Matrix

![Baseline Confusion Matrix](results/confusion_matrix_baseline.png)

### Baseline Feature Importance

![Baseline Feature Importance](results/feature_importance_baseline.png)

Current baseline result files:

```text
results/baseline_metrics.json
results/baseline_classification_report.txt
results/baseline_evaluation_summary.json
results/confusion_matrix_baseline.png
results/feature_importance_baseline.png

## Validation Note

The first version will be a development baseline, not a production-quality inspection system.

Important limitations to document:

- small initial scope using only one category
- possible dataset-specific performance
- limited robustness testing
- no real industrial deployment data
- no camera/sensor calibration checks
- no production monitoring

## Future Improvements

Planned extensions:

- add more MVTec AD categories
- compare classical image-feature baselines with anomaly-detection models
- add PatchCore or PaDiM
- generate anomaly heatmaps
- add stronger validation tests
- build a simple inference demo
