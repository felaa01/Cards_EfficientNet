# Playing Cards Classification with EfficientNet-B2

Image classification on a 53-class playing cards dataset using **EfficientNet-B2** with transfer learning and two-stage fine-tuning. Developed as a Master's degree lab assignment.

## 🎯 Objective

Outperform three baseline models (MobileNetV2, ViT-B16, Custom CNN) provided as reference.

## ✅ Results

| Model                      | Test Accuracy |
|----------------------------|---------------|
| Custom CNN (baseline)      | 75.09%        |
| MobileNetV2 (baseline)     | 90.19%        |
| ViT-B16 (baseline)         | 90.94%        |
| **EfficientNet-B2 (ours)** | **96.23%**    |

## Dataset

[Cards Image Dataset — Kaggle](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification) - 53 classes, 8,154 images (224×224 JPG) - Split: 7,624 train / 265 validation / 265 test

## Approach

**Model:** EfficientNet-B2 pretrained on ImageNet, with a custom classifier head (Dropout + Linear → 53 classes).

**Training:** - Stage 1 (20 epochs): backbone frozen, head-only training with AdamW (lr=1e-3) - Stage 2 (up to 42 epochs, early stopping): full fine-tuning with AdamW (lr=1e-4) + Cosine Annealing

**Regularization:** MixUp (α=0.2), data augmentation (random crop, flip, rotation, color jitter), weight decay (1e-4).

## ⚠️ Reproducibility

This notebook was developed on **Google Colab Pro** due to GPU memory requirements (\~11 GB). It is shared for reference — all training outputs and results are preserved in the notebook cells.

## Requirements

```         
torch torchvision scikit-learn seaborn matplotlib tqdm
```
