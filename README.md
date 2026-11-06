# Struvite | 磷酸镁铵

Single-frame ConvNeXt-Tiny model weights from the AiRICS study of
struvite-related visual features in ureteroscopic images and videos.

## Release contents

The model consists of **five trained ConvNeXt-Tiny folds**. Inference averages
the positive-class softmax probabilities of all five folds. These are the
original state dictionaries from the `convnext_tiny_patient5fold_20260804` run,
checked against the archived model hashes. No retraining or weight conversion
has been performed for this release.

- `infer.py`: minimal single-image inference example.
- `model_config.json`: architecture, preprocessing, classes, and aggregation.
- `weights_manifest.json`: checkpoint sizes and SHA-256 checksums.
- Release asset `Struvite_ConvNeXtTiny_5fold_v1.0.0.zip`: five checkpoints in a
  `weights/` directory. Download it from this repository's Releases page and
  extract it into the repository directory.

The release covers the **single-frame classifier**. Its input is an analysable
endoscopic stone surface or cross-section image after quality control and
viewport cropping. QC models, temporal event processing, and patient-level
aggregation are separate parts of the AiRICS workflow and are not included here.

## Installation and inference

The loading example was tested with Python 3.8, PyTorch 2.4.1 and torchvision
0.19.1. Use the matching PyTorch/torchvision builds for your platform.

```bash
python -m pip install -r requirements.txt
python infer.py --image /path/to/stone_frame.png --weights-dir weights --device cpu
```

Use `--device cuda` with a compatible CUDA-enabled PyTorch installation.
Output is JSON containing the mean positive probability, the predicted class,
and the five individual fold probabilities. Checkpoint checksums are verified
before loading, and loading uses `weights_only=True` and strict key matching.

## Model definition

- Backbone: torchvision `convnext_tiny`.
- Classification head: existing LayerNorm and Flatten followed by
  `Dropout(0.4)` and `Linear(768, 2)`.
- Input: RGB; resize the shorter edge to 236 pixels using bilinear interpolation;
  centre crop to 224 x 224; convert to a tensor in [0, 1].
- ImageNet normalisation: mean `[0.485, 0.456, 0.406]`,
  standard deviation `[0.229, 0.224, 0.225]`.
- Output index 0: struvite-negative; index 1: struvite-positive.
- Ensemble: arithmetic mean of the five positive softmax probabilities.
- Decision rule: positive when the ensemble probability is **at least 0.5**.

Image labels were assigned from postoperative stone composition reports:
all stone images from a patient were labelled positive when the report identified
struvite, and negative otherwise. A positive prediction is a compositional
classification; it does not by itself establish active infection. The model
does not estimate constituent percentages.

## Data availability

The data supporting the findings of this study are available from the
corresponding authors upon reasonable request. Patient images, videos, identifiers,
and individual-level predictions are not included in this repository or release.

## Intended use and provenance

These weights are provided for research and reproducibility. The single-frame
model is not a standalone clinical diagnostic system. Runtime checks on synthetic
inputs verify loading and execution; they are not a new clinical performance
evaluation.

The source experiment used ImageNet-pretrained ConvNeXt-Tiny weights. For the
architecture, see Liu et al., *A ConvNet for the 2020s*, CVPR 2022, pp. 11976–11986:
https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html

For questions about this release, use the repository's Issues page.
