# Project Documentation

This folder contains the validation, experiment, and model documentation for the industrial image defect-detection project.

## Documents

### Model Card

```text
model_card.md
```

Documents:

- intended use
- model inputs and outputs
- supervised baseline
- PatchCore anomaly detector
- evaluation metrics
- current results
- known limitations
- possible failure cases

### Validation Strategy

```text
validation_strategy.md
```

Documents:

- current evaluation design
- supervised-baseline limitations
- normal-only PatchCore evaluation
- image-level and pixel-level metrics
- threshold-validation requirements
- robustness and deployment checks

### Experiment Plan

```text
experiment_plan.md
```

Documents:

- completed experiments
- baseline feature experiments
- PatchCore configuration
- error analysis
- additional-category experiments
- future model comparisons
- robustness experiments

## Current Primary Model

The primary industrial anomaly-detection model is PatchCore.

Configuration:

```text
Dataset: MVTec AD bottle
Backbone: ResNet-18
Layers: layer2 and layer3
Input size: 224  224
Coreset ratio: 1%
Nearest neighbours: 5
Execution: CPU
```

Main results:

| Metric | Result |
|---|---:|
| Image AUROC | 1.000 |
| Image F1-score | 0.992 |
| Pixel AUROC | 0.976 |
| Pixel F1-score | 0.654 |
| Image-level accuracy | 0.976 |
| Correct test images | 81 / 83 |

The main remaining weaknesses are exact pixel-level localization, threshold validation, and robustness beyond the controlled MVTec benchmark.

<!-- PATCHCORE_VALIDATION_DOC_START -->
### PatchCore Output Validation

```text
patchcore_validation.md
```

Documents:

- reusable validation-toolkit integration
- saved prediction-table checks
- source-report and confusion-matrix consistency
- recalculated image-level metrics
- portable-path validation
- interpretation and remaining limitations
<!-- PATCHCORE_VALIDATION_DOC_END -->
