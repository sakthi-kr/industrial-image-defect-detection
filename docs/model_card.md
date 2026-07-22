# Model Card

## Project

Industrial Image Defect Detection

## Models

Two models are included.

### Supervised Baseline

A Random Forest classifier trained on manually extracted image features.

Feature groups:

- RGB statistics
- HSV statistics
- grayscale statistics
- grayscale histograms
- Sobel gradient features
- edge density
- Laplacian variance

This model is retained as a simple and interpretable development baseline.

### PatchCore Anomaly Detector

A normal-only anomaly-detection model using pretrained ResNet-18 patch embeddings.

Configuration:

| Parameter | Value |
|---|---|
| Backbone | ResNet-18 |
| Feature layers | `layer2`, `layer3` |
| Image size | 224 × 224 |
| Coreset ratio | 1% |
| Nearest neighbours | 5 |
| Execution | CPU |

PatchCore is the primary industrial anomaly-detection model in this project.

## Problem

Identify whether an industrial bottle image is normal or defective and localize regions that differ from normal training examples.

## Dataset

MVTec AD `bottle` category.

Training data:

- 209 normal images

Test data:

- 83 normal and defective images
- defect types: broken large, broken small, and contamination
- pixel-level masks for defective samples

## Intended Use

This project is intended for:

- applied machine-learning learning
- industrial anomaly-detection experimentation
- visual-inspection prototyping
- portfolio demonstration
- testing and validation workflow development

## Not Intended For

The models are not intended for direct production quality-control decisions.

They have not been validated on:

- real production camera streams
- unseen factories or machines
- changing lighting
- changing camera positions
- new bottle designs
- production-line motion blur
- long-term data drift

## Input

One RGB bottle image.

PatchCore preprocessing:

- image resizing to 224 × 224
- pretrained ResNet-18 feature extraction
- patch embeddings from `layer2` and `layer3`

## Output

PatchCore produces:

- image-level anomaly score
- predicted normal/defective label
- pixel-level anomaly map
- thresholded predicted mask
- heatmap overlay

## Main Results

PatchCore evaluation on 83 test images:

| Metric | Result |
|---|---:|
| Image AUROC | 1.000 |
| Image F1-score | 0.992 |
| Pixel AUROC | 0.976 |
| Pixel F1-score | 0.654 |
| Image-level accuracy | 0.976 |
| Correct images | 81 / 83 |

Observed errors:

- one normal image predicted as defective
- one contamination image predicted as normal

## Interpretation

Image-level defect ranking and classification are strong.

Pixel-level AUROC is also strong, but pixel F1 is lower. This indicates that the anomaly maps generally identify relevant defective regions while the exact thresholded mask boundaries do not always match the ground-truth masks precisely.

The perfect image AUROC does not mean every image is correctly classified at the selected threshold. AUROC evaluates ranking across thresholds, while accuracy and F1 depend on a specific threshold.

## Known Limitations

- only one MVTec category has been evaluated
- benchmark data is more controlled than production data
- the classification threshold was not independently tuned
- one subtle contamination defect was missed
- one normal appearance variation caused a false positive
- exact pixel-level localization requires improvement
- no robustness testing under controlled image corruptions
- no production monitoring or drift detection

## Possible Failure Cases

The model may fail when:

- normal appearance varies beyond the training distribution
- defects are small or low contrast
- illumination changes significantly
- image blur obscures local texture
- camera alignment changes
- a new bottle design is introduced
- the background differs from the benchmark
- the defect resembles normal visual variation

## Ethical and Operational Considerations

A production inspection system should not automatically reject or approve safety-critical products without:

- representative validation
- expert review
- calibrated operating thresholds
- uncertainty handling
- monitoring
- fallback procedures
- audit logging

## Future Improvements

- evaluate more MVTec categories
- compare PatchCore with PaDiM
- improve pixel-level thresholding
- introduce a separate validation set
- evaluate by defect type
- test lighting, blur, noise, and geometric transformations
- add monitoring and drift checks
- build a simple inference interface
