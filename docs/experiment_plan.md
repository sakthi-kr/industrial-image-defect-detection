# Experiment Plan

## Goal

The goal is to build a reproducible machine-learning workflow for industrial image defect detection and progressively move from a simple supervised baseline to a more realistic anomaly-detection setup.

## Current Experiment: Baseline Random Forest Classifier

### Dataset

MVTec AD `bottle` category.

### Task

Binary classification:

```text
normal image    -> normal
defective image -> defective
```

### Features

The current baseline uses manually extracted image features.

Colour features:

- RGB channel means
- RGB channel standard deviations
- RGB channel min/max values
- HSV statistics

Grayscale features:

- mean
- standard deviation
- percentiles
- entropy
- histogram bins

Edge/texture features:

- Sobel gradient statistics
- edge density
- Laplacian variance

### Model

Random Forest classifier.

### Purpose

This experiment checks whether simple image statistics can separate normal and defective bottle images and confirms that the full pipeline works.

### Interpretation

The result is useful as a baseline, but it should not be treated as final industrial anomaly-detection performance.

---

## Experiment 1: Baseline Feature Groups

### Question

Which simple image features contribute most?

### Comparisons

- colour features only
- grayscale/intensity features only
- edge/texture features only
- all features combined

### Purpose

This helps understand whether defects are mainly captured by colour, intensity, texture, or edge changes.

---

## Experiment 2: Classical Model Comparison

### Question

How do different classical models perform on the same extracted features?

### Models

- Logistic Regression
- Random Forest
- Support Vector Machine
- k-Nearest Neighbors
- Gradient Boosting or XGBoost

### Metrics

- accuracy
- precision
- recall
- F1-score
- confusion matrix

---

## Experiment 3: MVTec-Style Anomaly Detection

### Question

Can the model detect defects after training only on normal images?

### Method

Train on:

```text
bottle/train/good
```

Evaluate on:

```text
bottle/test/good
bottle/test/broken_large
bottle/test/broken_small
bottle/test/contamination
```

### Candidate Models

- PatchCore
- PaDiM
- autoencoder baseline

### Purpose

This is closer to real industrial inspection, where training data may contain mostly normal examples.

---

## Experiment 4: Defect-Type Error Analysis

### Question

Which defect types are most difficult?

### Defect Types

- broken_large
- broken_small
- contamination

### Analysis

For each defect type:

- number of correct predictions
- number of missed defects
- confidence or anomaly-score distribution
- visual examples of failures

---

## Experiment 5: Heatmap-Based Localization

### Question

Can the model localize defective regions, not just classify the image?

### Method

Use anomaly-detection methods that generate heatmaps.

Candidate methods:

- PatchCore
- PaDiM

### Expected Outputs

- anomaly heatmaps
- image-level anomaly scores
- comparison with ground-truth masks

---

## Final Deliverables

The final project should include:

- clean image data loading
- preprocessing pipeline
- sample image visualizations
- baseline classifier
- confusion matrix
- feature importance
- prediction script
- basic tests
- validation strategy
- experiment plan
- model card
- future anomaly-detection extension
