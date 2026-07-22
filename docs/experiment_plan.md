# Experiment Plan

## Goal

Develop a reproducible industrial image anomaly-detection workflow and evaluate both image-level defect detection and pixel-level defect localization.

## Completed Experiment 1: Dataset Pipeline

Completed:

- MVTec AD bottle loading
- split and defect-type indexing
- binary-label generation
- mask-path discovery
- dataset summary generation
- normal and defective sample visualization

## Completed Experiment 2: Image Preprocessing

Completed:

- RGB conversion
- image resizing
- normalization
- grayscale conversion
- preprocessing preview generation

## Completed Experiment 3: Supervised Baseline

Model:

```text
RandomForestClassifier
```

Features:

- RGB statistics
- HSV statistics
- grayscale statistics
- histogram bins
- Sobel gradients
- edge density
- Laplacian variance

Purpose:

- verify the end-to-end classical ML pipeline
- create an interpretable development benchmark

Limitation:

The model uses both normal and defective examples and is not the primary anomaly-detection method.

## Completed Experiment 4: PatchCore

Configuration:

| Parameter | Value |
|---|---|
| Dataset | MVTec AD bottle |
| Backbone | ResNet-18 |
| Layers | `layer2`, `layer3` |
| Input size | 224 × 224 |
| Coreset ratio | 1% |
| Nearest neighbours | 5 |
| Accelerator | CPU |

Training:

- only normal images
- pretrained CNN feature extraction
- memory-bank construction
- reduced coreset sampling

Results:

| Metric | Result |
|---|---:|
| Image AUROC | 1.000 |
| Image F1-score | 0.992 |
| Pixel AUROC | 0.976 |
| Pixel F1-score | 0.654 |
| Image accuracy | 0.976 |

## Completed Experiment 5: Prediction and Error Analysis

Completed:

- full-test-set prediction
- prediction CSV
- error-analysis CSV
- confusion-matrix table
- anomaly-score distribution
- false-positive analysis
- false-negative analysis

Observed errors:

- one normal false positive
- one contamination false negative

## Completed Experiment 6: Localization Visualization

Completed:

- anomaly maps
- heatmap overlays
- ground-truth mask comparison
- representative normal and defective examples

## Planned Experiment 7: Baseline Feature Ablation

Compare:

- colour features only
- grayscale features only
- histogram features only
- edge and texture features only
- all features combined

Question:

Which manually engineered feature groups contribute most to the supervised baseline?

## Planned Experiment 8: Classical Model Comparison

Compare:

- Logistic Regression
- Random Forest
- Support Vector Machine
- k-Nearest Neighbours
- gradient boosting

Use identical feature tables and evaluation splits.

## Planned Experiment 9: Independent Threshold Validation

Develop a validation procedure that does not tune on the final test set.

Possible approaches:

- hold out a subset of normal training images
- create controlled synthetic anomalies
- use a separate MVTec category for threshold calibration
- compare threshold stability across categories

## Planned Experiment 10: Additional Categories

Possible categories:

- metal nut
- cable
- capsule
- screw

Questions:

- does the lightweight PatchCore configuration generalize?
- which object structures are more difficult?
- does one decision threshold transfer across categories?

## Planned Experiment 11: Model Comparison

Compare PatchCore with:

- PaDiM
- another lightweight anomaly detector
- optional autoencoder baseline

Compare:

- image AUROC
- image F1-score
- pixel AUROC
- pixel F1-score
- inference time
- memory use
- qualitative localization

## Planned Experiment 12: Robustness Testing

Apply controlled transformations:

- brightness changes
- contrast changes
- Gaussian noise
- blur
- JPEG compression
- small rotations
- translations

Measure:

- anomaly-score shift for normal images
- false-positive rate
- false-negative rate
- threshold stability

## Planned Experiment 13: Validation Toolkit Integration

Use reusable validation utilities for:

- required result columns
- missing and infinite values
- prediction-label checks
- probability and score-range checks
- metric-schema checks
- report generation
- reproducibility checks

## Planned Experiment 14: Inference Prototype

Build a simple interface that:

- accepts one image
- returns normal or defective
- reports anomaly score
- displays anomaly heatmap
- documents model limitations

## Final Deliverables

The mature project should contain:

- reproducible data loading
- supervised baseline
- normal-only anomaly detector
- image-level evaluation
- pixel-level evaluation
- heatmaps
- error analysis
- robustness checks
- reusable validation checks
- model card
- experiment documentation
- inference prototype
