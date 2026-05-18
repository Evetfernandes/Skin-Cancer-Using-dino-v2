# Enhanced Skin Cancer Detection with DINOv2 + Supervised Contrastive Learning


A deep learning pipeline for 7-class skin lesion classification using a DINOv2 ViT-B/14 backbone trained with Supervised Contrastive Learning (SupCon), evaluated on HAM10000, ISIC 2019, and ISIC 2020 datasets.

---

## Overview

This project implements a two-stage training strategy:

1. **Stage 1 — Contrastive Pretraining**: The DINOv2 backbone is fine-tuned using Supervised Contrastive Loss to learn discriminative feature embeddings across skin lesion classes.
2. **Stage 2 — Classifier Fine-tuning**: A classification head is trained on top of the frozen (or partially unfrozen) backbone using Focal Loss with label smoothing to handle class imbalance.

Baseline architectures (ResNet-152, EfficientNetV2) are also supported for ablation comparisons.

---

## Features

- DINOv2 ViT-B/14 backbone with selective layer unfreezing
- Supervised Contrastive Learning (SupCon) pretraining
- Focal Loss + label smoothing for class imbalance
- Test-Time Augmentation (TTA) with entropy-weighted averaging
- Specialist referral flag when model confidence < 70%
- Cross-dataset evaluation (HAM10000 → ISIC 2019 / ISIC 2020)
- Ablation study framework
- Visualizations: confusion matrix, ROC curves, t-SNE embeddings, DINO attention maps
- Single-image inference script with result figure output

---

## Project Structure

```
.
├── main_2.py            # Full training & evaluation pipeline
├── predict_single.py    # Single-image inference with visualization
├── test_model.py        # Standalone test/evaluation script
├── data/                # Dataset directory (see Data Setup below)
├── checkpoints/         # Saved model weights
├── figures/             # Training visualizations
└── results/             # Test evaluation outputs
```

---

## Data Setup

Download the datasets from the ISIC Archive:

| Dataset | Source |
|---|---|
| HAM10000 | [Kaggle](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) |
| ISIC 2019 | [ISIC Archive](https://challenge.isic-archive.com/data/#2019) |
| ISIC 2020 | [Kaggle](https://www.kaggle.com/c/siim-isic-melanoma-classification) |

Arrange files in the following layout:

```
data/
├── HAM10000_metadata.csv
├── HAM10000_images/            ← merge part_1 and part_2 into one folder
├── ISIC_2019_Training_GroundTruth.csv
├── ISIC_2019_Training_Input/
├── ISIC_2020_Training_GroundTruth.csv
└── ISIC_2020_Training_Input/
```

---

## Installation

```bash
pip install torch torchvision scikit-learn pandas pillow matplotlib seaborn tqdm einops
```

> **Note:** The DINOv2 backbone is loaded automatically via `torch.hub` from `facebookresearch/dinov2` — an internet connection is required on the first run.

---

## Usage

### Full Pipeline

```bash
python main_2.py                        # run all stages
python main_2.py --stage data           # verify data only
python main_2.py --stage train          # train only
python main_2.py --stage eval           # evaluate only
python main_2.py --stage visualize      # generate figures only
python main_2.py --stage ablation       # run ablation study
python main_2.py --arch resnet152       # run CNN baseline instead
python main_2.py --resume checkpoints/stage2_best.pt   # resume from checkpoint
```

Supported architectures: `dinov2` (default), `resnet152`, `efficientnetv2`

### Evaluate a Trained Model

```bash
python test_model.py                                   # auto-detects checkpoint and data
python test_model.py --checkpoint checkpoints/stage2_best.pt
python test_model.py --data_dir /path/to/HAM10000_images --csv /path/to/metadata.csv
python test_model.py --arch resnet152
python test_model.py --no_tta                          # skip TTA (faster)
python test_model.py --no_tsne                         # skip t-SNE (slow)
```

Outputs are saved to `results/`.

### Single Image Inference

```bash
python predict_single.py --image path/to/lesion.jpg
python predict_single.py --image photo.jpg --checkpoint checkpoints/stage2_best.pt
python predict_single.py --image photo.jpg --arch resnet152
python predict_single.py --image photo.jpg --no_tta    # faster, single-pass inference
```

A result figure (`result_<image>.png`) is saved with the predicted class, confidence score, risk level, and class probability chart.

---

## Classes (HAM10000)

| ID | Code | Full Name | Risk |
|---|---|---|---|
| 0 | MEL | Melanoma | HIGH |
| 1 | NV | Melanocytic Nevus | LOW |
| 2 | BCC | Basal Cell Carcinoma | MEDIUM |
| 3 | AK | Actinic Keratosis | MEDIUM |
| 4 | BKL | Benign Keratosis-like Lesion | LOW |
| 5 | DF | Dermatofibroma | LOW |
| 6 | VASC | Vascular Lesion | LOW |

---

## Configuration

Key hyperparameters are set in the `CONFIG` dict at the top of `main_2.py`:

| Parameter | Default | Description |
|---|---|---|
| `arch` | `dinov2` | Backbone architecture |
| `img_size` | `518` | Input resolution (518 for DINOv2, 224 for CNNs) |
| `s1_epochs` | `30` | Stage 1 (SupCon) epochs |
| `s1_batch` | `4` | Stage 1 batch size (×128 grad accumulation → effective 512) |
| `s2_epochs` | `20` | Stage 2 (classifier) epochs |
| `temperature` | `0.07` | SupCon temperature τ |
| `focal_gamma` | `2.0` | Focal loss γ |
| `tta_views` | `10` | Number of TTA augmentation views |
| `referral_thr` | `0.70` | Confidence threshold below which referral is flagged |
| `unfreeze_last_n` | `4` | Number of ViT blocks to unfreeze during fine-tuning |

---

## Output Files

After running `test_model.py`, the `results/` directory contains:

```
results/
├── 01_summary_dashboard.png
├── 02_confusion_matrix_standard.png
├── 02_confusion_matrix_tta.png
├── 03_roc_curves_standard.png
├── 03_roc_curves_tta.png
├── 04_per_class_f1.png
├── 05_confidence_distribution.png
├── 06_tsne_embeddings.png
├── 07_attention_maps.png
└── test_metrics.csv
```

---

## Disclaimer

> ⚠️ **This project is for research purposes only and is not intended for clinical diagnosis or medical decision-making.**
