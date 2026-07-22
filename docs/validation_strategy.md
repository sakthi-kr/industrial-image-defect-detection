# Validation Strategy

## Purpose

This document describes the validation approach for the industrial image defect detection project.

The first baseline model uses simple image-level features and a supervised Random Forest classifier. This is useful for checking that the full image-processing and machine-learning pipeline works, but it is not the final industrial anomaly-detection approach.

## Current Baseline Validation

Current setup:

- Dataset: MVTec AD
- Category: bottle
- Task: binary classification
- Labels: normal and defective
- Features: colour, grayscale, histogram, and edge-based image features
- Model: Random Forest classifier
- Split: stratified random train/test split over image-level features

## Main Limitation

The current baseline is a supervised classifier trained using both normal and defective images.

This is useful as a development baseline, but many industrial anomaly-detection systems are designed differently:

```text
train only on normal images
detect deviations from normal behaviour at test time
```

A future version should therefore use anomaly-detection methods such as PatchCore or PaDiM.

## Why This Baseline Still Matters

The first baseline confirms that the full pipeline works:

```text
load images -> preprocess -> extract features -> train model -> evaluate -> predict
```

It also creates a simple benchmark before moving to stronger industrial anomaly-detection methods.

## Improved Validation Plan

### Stage 1: Supervised Baseline

Purpose:

- confirm end-to-end image pipeline
- test image loading and preprocessing
- establish a simple benchmark

Status:

- completed as first baseline

### Stage 2: Train/Test Split by MVTec Structure

Purpose:

- train using `train/good`
- evaluate using `test/good` and test defect folders

Expected benefit:

- closer to the official anomaly-detection structure of MVTec AD

### Stage 3: Unsupervised Anomaly Detection

Purpose:

- train only on normal images
- detect defective images as anomalies

Candidate methods:

- PatchCore
- PaDiM
- autoencoder baseline

Expected benefit:

- closer to industrial visual-inspection use cases where all defect types may not be known during training

### Stage 4: Defect-Type Evaluation

Purpose:

- evaluate performance separately for each defect type

For bottle:

- broken_large
- broken_small
- contamination

Expected benefit:

- shows which defect types are easy or difficult for the model

### Stage 5: More MVTec Categories

Purpose:

- test whether the approach generalizes beyond one object category

Possible next categories:

- metal_nut
- screw
- cable
- capsule

Expected benefit:

- stronger evidence of general image-defect detection ability

## Validation Metrics

Classification metrics:

- accuracy
- precision
- recall
- F1-score
- confusion matrix

Anomaly-detection metrics for future versions:

- image-level AUROC
- pixel-level AUROC
- anomaly score distribution
- false-positive examples
- false-negative examples
- defect localization heatmaps

## Deployment-Relevant Checks

Before any real industrial use, additional checks would be needed:

- camera calibration
- lighting variation tests
- unseen product batches
- image blur/noise robustness
- threshold stability
- monitoring for data drift
- human expert review
- integration with inspection workflow

## Summary

The current baseline is a working proof of pipeline. The next goal is not only better accuracy, but a more realistic industrial anomaly-detection validation setup.
