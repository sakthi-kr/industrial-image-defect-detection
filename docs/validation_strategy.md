# Validation Strategy

## Purpose

This document describes the validation strategy for the industrial image defect-detection project.

The project contains:

1. a supervised Random Forest development baseline
2. a PatchCore model trained only on normal images

PatchCore is the primary industrial anomaly-detection workflow.

## Dataset Structure

Dataset:

```text
MVTec AD bottle
```

Training:

```text
train/good
```

Testing:

```text
test/good
test/broken_large
test/broken_small
test/contamination
```

Pixel-level ground-truth masks are available for defective test images.

## Supervised Baseline Validation

The Random Forest baseline uses a random image-level train/test split across the available images.

Purpose:

- verify data loading
- verify preprocessing
- test handcrafted features
- test training and prediction scripts
- establish an interpretable benchmark

Limitation:

The baseline uses defective examples during training and does not follow the normal-only MVTec anomaly-detection protocol.

It should not be interpreted as the primary benchmark result.

## PatchCore Validation

PatchCore is trained only on normal training images.

The evaluation uses:

- unseen normal test images
- broken-large defects
- broken-small defects
- contamination defects

Outputs include:

- image-level anomaly score
- image-level predicted label
- pixel-level anomaly map
- pixel-level predicted mask

## Current Metrics

| Metric | Result |
|---|---:|
| Image AUROC | 1.000 |
| Image F1-score | 0.992 |
| Pixel AUROC | 0.976 |
| Pixel F1-score | 0.654 |
| Image-level accuracy | 0.976 |
| Correct test images | 81 / 83 |

## Error Analysis

Two image-level errors were observed:

### False Positive

One normal image was classified as defective.

Interpretation:

- normal appearance variation produced a comparatively high anomaly score
- the selected threshold is sensitive to some valid normal variation

### False Negative

One contamination image was classified as normal.

Interpretation:

- the contamination defect was comparatively subtle
- its anomaly score remained close to the decision threshold

## Threshold Validation

The decision threshold must not be tuned directly on the final test set.

A stronger approach would create a separate validation strategy using:

- held-out normal images
- synthetic anomalies
- an additional validation category
- cross-category threshold analysis

The final test set should remain untouched until the threshold and model configuration have been selected.

## Image-Level Validation

Current image-level checks:

- AUROC
- precision
- recall
- F1-score
- accuracy
- confusion matrix
- false positives
- false negatives
- anomaly-score distribution

## Pixel-Level Validation

Current pixel-level checks:

- pixel AUROC
- pixel F1-score
- anomaly heatmaps
- overlay visualization
- comparison with ground-truth masks

The lower pixel F1-score indicates that localization thresholding and mask precision require improvement.

## Defect-Type Validation

The current report includes examples from:

- broken large
- broken small
- contamination

Future reports should calculate metrics separately for each defect type rather than only overall.

## Robustness Validation

Future robustness experiments should test:

- brightness changes
- contrast changes
- colour shifts
- Gaussian noise
- blur
- image compression
- small rotations
- translations
- crop changes
- camera-position changes

These tests should distinguish acceptable imaging variation from actual defects.

## Deployment-Relevant Validation

Before production use, the following would be required:

- representative production images
- camera calibration
- lighting control
- threshold calibration
- latency measurements
- failure handling
- human review workflow
- monitoring for drift
- audit logging
- periodic revalidation

## Current Conclusion

PatchCore provides strong image-level anomaly detection on the MVTec AD bottle benchmark.

The main unresolved issues are:

- independent threshold validation
- exact pixel-level localization
- robustness beyond the controlled benchmark
- generalization to additional object categories
- production deployment validation
