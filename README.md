# Industrial Image Defect Detection

## Summary

This project develops an industrial image defect-detection pipeline using the MVTec AD `bottle` category. The current version includes image loading, preprocessing, sample visualization, feature extraction, a Random Forest baseline classifier, evaluation, prediction scripting, tests, and validation planning.

## Motivation

Industrial inspection often relies on image data to identify defective parts, surface damage, contamination, or production issues. This project explores image-based machine learning as a practical AI/ML workflow for industrial visual inspection and quality control.

## Dataset

Dataset:

```text
MVTec AD industrial anomaly detection dataset
```

Current category:

```text
bottle
```

The dataset is not included in this repository. Download and folder-structure instructions are provided in:

```text
data/README.md
```

## Problem Definition

The first baseline treats the task as binary classification:

```text
normal image     -> normal
defective image  -> defective
```

Later versions will move toward industrial anomaly detection:

```text
train only on normal images
detect deviations from normal appearance at test time
```

## Sample Images

The first version uses the MVTec AD `bottle` category, including normal images and several defect types.

![Sample Images](results/sample_images.png)

## Preprocessing Preview

Images are converted to RGB, resized to `128 × 128`, normalized to `[0, 1]`, and converted to grayscale for simple baseline feature extraction.

![Preprocessing Preview](results/preprocessing_preview.png)

## Current Baseline Method

The current baseline uses manually extracted image features and a Random Forest classifier.

Feature groups:

- RGB colour statistics
- HSV colour statistics
- grayscale intensity statistics
- grayscale histogram features
- Sobel edge features
- Laplacian variance

Model:

```text
RandomForestClassifier
```

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
```


## PatchCore Anomaly-Detection Upgrade

The second version uses PatchCore, an industrial anomaly-detection method trained only on normal MVTec AD bottle images.

Unlike the supervised baseline, PatchCore does not require defective training examples. It extracts patch-level features using a pretrained CNN backbone, builds a representative memory bank from normal images, and assigns anomaly scores by comparing test-image patches with that normal-feature memory.

### Configuration

| Parameter | Value |
|---|---|
| Dataset | MVTec AD bottle |
| Normal training images | 209 |
| Test images | 83 |
| Backbone | ResNet-18 |
| Feature layers | `layer2`, `layer3` |
| Input size | 224 × 224 |
| Coreset ratio | 1% |
| Nearest neighbours | 5 |
| Execution | CPU |
| Training and evaluation time | Approximately 59 seconds |

### Quantitative Results

| Metric | Result |
|---|---:|
| Image AUROC | 1.000 |
| Image F1-score | 0.992 |
| Pixel AUROC | 0.976 |
| Pixel F1-score | 0.654 |
| Image-level accuracy from prediction report | 0.976 |
| Correct test images | 81 / 83 |

The image-level results are strong, but exact defect localization remains less accurate than image-level detection. The prediction report found one false positive and one false negative.

### Anomaly Heatmaps

The figure compares original images, predicted anomaly maps, heatmap overlays, and pixel-level ground-truth masks.

![PatchCore Anomaly Heatmaps](results/patchcore_example_heatmaps.png)

### Anomaly-Score Distribution

Normal and defective images are largely separated, with a small overlap near the selected classification threshold.

![PatchCore Score Distribution](results/patchcore_score_distribution.png)

### Error Analysis

The two observed image-level errors were:

- one normal image classified as defective
- one contamination image classified as normal

This indicates sensitivity to normal appearance variation and difficulty detecting a comparatively subtle contamination example.

Detailed outputs:

```text
results/patchcore_predictions.csv
results/patchcore_error_analysis.csv
results/patchcore_report.json
results/patchcore_confusion_matrix.csv
```

### Interpretation

The supervised Random Forest baseline and PatchCore results are not directly comparable. The baseline trains using both normal and defective examples, while PatchCore trains only on normal images and therefore represents a more realistic industrial anomaly-detection workflow.

## Project Structure

```text
industrial-image-defect-detection/
├── data/
│   └── README.md
├── docs/
│   ├── model_card.md
│   ├── validation_strategy.md
│   └── experiment_plan.md
├── notebooks/
├── results/
├── src/
│   ├── data_loader.py
│   ├── visualize.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocess.py
│   ├── test_features.py
│   └── test_prediction_output.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Load dataset index

```bash
python src/data_loader.py
```

This creates:

```text
results/dataset_index.csv
results/dataset_summary.csv
```

### 4. Generate sample image grid

```bash
python src/visualize.py
```

This creates:

```text
results/sample_images.png
```

### 5. Run preprocessing preview

```bash
python src/preprocess.py
```

This creates:

```text
results/preprocessing_preview.png
```

### 6. Extract image features

```bash
python src/features.py
```

This creates:

```text
results/image_feature_table.csv
results/image_feature_preview.csv
results/image_feature_summary.csv
```

### 7. Train baseline classifier

```bash
python src/train.py
```

This creates:

```text
models/baseline_random_forest_image_classifier.joblib
results/baseline_metrics.json
results/baseline_classification_report.txt
results/baseline_feature_columns.json
```

### 8. Evaluate baseline classifier

```bash
python src/evaluate.py
```

This creates:

```text
results/confusion_matrix_baseline.png
results/feature_importance_baseline.png
results/baseline_evaluation_summary.json
```

### 9. Predict one image

```bash
python src/predict.py --image data/raw/mvtec_ad/bottle/test/good/000.png
```

Example output:

```text
Predicted label: normal
```

### 10. Run tests

```bash
pytest
```

## Testing

Tests are included for:

- dataset loading
- image path discovery
- label extraction
- RGB image loading
- resizing
- normalization
- grayscale conversion
- feature extraction
- prediction output structure

## Validation Note

The current baseline is a first development version. It is not yet a full industrial anomaly-detection solution.

Detailed validation and experiment plans are documented here:

```text
docs/validation_strategy.md
docs/experiment_plan.md
```

Main limitations:

- only one MVTec AD category used so far
- baseline is supervised, not pure anomaly detection
- model uses manually extracted features
- no anomaly heatmaps yet
- no robustness testing under changing camera or lighting conditions
- no production deployment validation

## Experiment Roadmap

Planned experiment stages:

1. Baseline Random Forest model on manually extracted image features
2. Feature-group comparison: colour vs grayscale vs edge features
3. Classical model comparison: Logistic Regression, Random Forest, SVM, and gradient boosting
4. MVTec-style anomaly detection: train only on normal images
5. PatchCore or PaDiM anomaly-detection model
6. Defect-type error analysis
7. Anomaly heatmap generation and comparison with ground-truth masks

The focus is not only on accuracy, but on making the validation more realistic and industrially meaningful.

## Model Card

A model card is provided in:

```text
docs/model_card.md
```

## Limitations

This project is for applied learning and portfolio demonstration. It is not intended for real industrial inspection decisions without:

- more extensive validation
- testing on unseen production data
- robustness checks
- expert review
- deployment monitoring

## Future Improvements

Planned extensions:

- add PatchCore or PaDiM using Anomalib
- train only on normal images
- generate anomaly localization heatmaps
- evaluate image-level and pixel-level anomaly detection
- add more MVTec AD categories
- add a simple inference demo
- connect validation utilities from the ML testing and validation toolkit
