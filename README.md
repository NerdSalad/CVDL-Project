# Explainable Potato Leaf Disease Detection using Vision Transformer (ViT) & XAI

An explainable deep learning framework for detecting potato leaf diseases using a fine-tuned Vision Transformer (ViT) with Grad-CAM and Attention Rollout for model interpretability.

## Overview

This project classifies potato leaf images into three categories — **Early Blight**, **Late Blight**, and **Healthy** — using a ViT model fine-tuned on the PlantVillage dataset. Explainable AI (XAI) techniques are applied to visualize which regions of the leaf the model focuses on when making predictions.

## Dataset

- **Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) (downloaded via `kagglehub`)
- **Classes:** Potato Early Blight, Potato Late Blight, Potato Healthy
- **Split:** 70% Training / 20% Validation / 10% Testing
- **Preprocessing:** Resize to 224x224, normalization, and aggressive data augmentation (rotation, blur, noise, color jitter, grid shuffle, channel shuffle, coarse dropout)

## Model Architecture

- **Base Model:** `google/vit-base-patch16-224` (ImageNet pretrained)
- **Fine-tuning Strategy:** Patch embeddings and first 6 encoder layers frozen; last 6 layers + classifier trainable
- **Regularization:** Dropout (0.4), weight decay (0.05), label smoothing (0.15), cosine LR scheduler, early stopping
- **Class Imbalance:** Handled via weighted cross-entropy loss

## Explainability (XAI)

### Grad-CAM
Generates heatmaps highlighting disease-affected regions by computing gradients at the final LayerNorm of the ViT encoder.

### Attention Rollout
Aggregates attention weights across all transformer layers using top-k head selection (by variance) and percentile thresholding to produce focused attention maps.

## Comparative Study

Performance comparison between:
- **ViT** (Fine-tuned Vision Transformer)
- **ResNet50** (CNN baseline)
- **MobileNetV2** (Lightweight CNN baseline)

## Robustness Evaluation

The ViT model is tested under simulated real-world conditions:
- Gaussian Blur
- Sensor Noise
- Low/High Brightness
- Motion Blur
- Combined Stress

## Output Visualizations

| Output | Description |
|--------|-------------|
| `class_distribution.png` | Dataset class distribution |
| `vit_training_curves.png` | Training loss, validation loss, accuracy & F1 curves |
| `vit_confusion_matrix.png` | Confusion matrix (counts & normalized) |
| `vit_gradcam_heatmaps.png` | Grad-CAM heatmaps per class |
| `vit_attention_rollout.png` | Attention rollout maps per class |
| `model_comparison.png` | ViT vs CNN performance comparison |
| `all_models_confusion_matrices.png` | Side-by-side confusion matrices for all models |
| `vit_robustness_evaluation.png` | Robustness under real-world conditions |
| `sample_predictions.png` | Sample predictions with confidence scores |

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/NerdSalad/CVDL-Project.git
cd CVDL-Project

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies (installs CUDA-enabled PyTorch)
pip install -r requirements.txt

# Launch notebook
jupyter notebook
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- ~6 GB VRAM minimum

## Project Structure

```
.
├── plant-village-disease-classifiy-using-hugging-face.ipynb   # Main notebook
├── requirements.txt                                           # Python dependencies
├── README.md
├── .gitignore
├── class_distribution.png
├── vit_training_curves.png
├── vit_confusion_matrix.png
├── vit_gradcam_heatmaps.png
├── vit_attention_rollout.png
├── model_comparison.png
├── all_models_confusion_matrices.png
├── vit_robustness_evaluation.png
└── sample_predictions.png
```

## Tech Stack

- **PyTorch** + **HuggingFace Transformers** — Model training & inference
- **Albumentations** — Data augmentation
- **pytorch-grad-cam** — Grad-CAM visualization
- **scikit-learn** — Metrics & evaluation
- **Matplotlib / Seaborn** — Plotting
