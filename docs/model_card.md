# Model Card

## Model Name

Baseline Random Forest Image Defect Classifier

## Problem

Classify industrial product images as normal or defective.

## Dataset

MVTec AD industrial anomaly detection dataset.

Current category:

- bottle

## Intended Use

This model is intended as an educational and portfolio project for industrial visual inspection and image-based defect detection.

It demonstrates:

- image loading
- image preprocessing
- feature extraction
- baseline machine-learning classification
- model evaluation
- prediction scripting
- basic testing and validation

## Not Intended For

This model is not intended for real industrial quality-control decisions.

It has not been validated on:

- real factory camera data
- changing lighting conditions
- unseen product batches
- different camera angles
- production-line noise
- deployment-time data drift

## Model Type

Random Forest classifier trained on manually extracted image features.

## Input

One industrial product image.

The image is:

- converted to RGB
- resized to 128 × 128
- normalized to [0, 1]
- converted to grayscale for selected features

## Output

The model outputs:

- predicted class: normal or defective
- class probabilities, when available
- confidence score
- expected label inferred from folder structure, when available

## Evaluation Metrics

Current evaluation includes:

- accuracy
- precision
- recall
- F1-score
- confusion matrix
- feature importance

## Main Results

The first baseline produces a working normal-vs-defective classification pipeline for the MVTec AD bottle category.

The result should be interpreted as a development baseline, not as final industrial anomaly-detection performance.

## Known Limitations

Main limitations:

- only one MVTec AD category used so far
- supervised baseline uses both normal and defective images
- not yet trained in a pure anomaly-detection setting
- no PatchCore or PaDiM model yet
- no anomaly heatmaps yet
- no robustness testing under lighting/camera changes
- no production deployment validation

## Possible Failure Cases

The model may fail when:

- defects are visually small
- lighting changes significantly
- camera position changes
- product appearance differs from MVTec AD examples
- defect type was not represented in the baseline training data
- background conditions differ from the dataset

## Future Improvements

- train anomaly-detection model using only normal images
- add PatchCore or PaDiM
- generate anomaly localization heatmaps
- compare defect-type performance
- add more MVTec AD categories
- add threshold analysis
- add model monitoring and data-drift checks
- build a simple inference demo
