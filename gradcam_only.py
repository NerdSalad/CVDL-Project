import os
import random
import warnings
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from safetensors.torch import load_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

REPO_ROOT = Path(__file__).resolve().parent
MODEL_NAME = "google/vit-base-patch16-224"
POTATO_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]
LABEL_NAMES = {
    "Potato___Early_blight": "Early Blight",
    "Potato___Late_blight": "Late Blight",
    "Potato___healthy": "Healthy",
}


def find_valid_dataset_root(base_path: Path) -> Path | None:
    candidates = [
        base_path / "PlantVillage",
        base_path / "plantvillage dataset" / "color",
        base_path / "color",
        base_path,
    ]
    for path in candidates:
        if path.is_dir():
            contents = os.listdir(path)
            if any(class_name in contents for class_name in POTATO_CLASSES):
                return path
    return None


def resolve_dataset_path() -> Path:
    local_candidates = [
        REPO_ROOT / "PlantVillage",
        REPO_ROOT,
        REPO_ROOT.parent / "PlantVillage",
    ]
    for local_path in local_candidates:
        resolved = find_valid_dataset_root(local_path)
        if resolved is not None:
            return resolved
    raise FileNotFoundError("PlantVillage dataset not found locally.")


def load_potato_data(dataset_path: Path) -> pd.DataFrame:
    images, labels = [], []
    for subfolder in tqdm(sorted(os.listdir(dataset_path)), desc="Scanning folders"):
        if subfolder not in POTATO_CLASSES:
            continue
        subfolder_path = dataset_path / subfolder
        if not subfolder_path.is_dir():
            continue
        for image_filename in sorted(os.listdir(subfolder_path)):
            if image_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(str(subfolder_path / image_filename))
                labels.append(LABEL_NAMES[subfolder])
    return pd.DataFrame({"image": images, "label": labels})


def prepare_data():
    dataset_path = resolve_dataset_path()
    df = load_potato_data(dataset_path)

    label_encoder = LabelEncoder()
    df["label_enc"] = label_encoder.fit_transform(df["label"])

    _, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label_enc"],
    )
    _, test_df = train_test_split(
        temp_df,
        test_size=0.333,
        random_state=SEED,
        stratify=temp_df["label_enc"],
    )

    test_df = test_df.reset_index(drop=True)
    return test_df, label_encoder


def get_val_transforms():
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    return A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(
                mean=image_processor.image_mean,
                std=image_processor.image_std,
            ),
            ToTensorV2(),
        ]
    )


def resolve_checkpoint_dir() -> Path:
    best_dir = REPO_ROOT / "vit-potato-disease-best"
    if best_dir.is_dir():
        return best_dir

    checkpoint_root = REPO_ROOT / "vit-potato-disease"
    checkpoints = sorted(
        [d for d in checkpoint_root.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda path: int(path.name.split("-")[1]),
    )
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError("No trained ViT checkpoint directory found.")


def load_trained_model(label_encoder: LabelEncoder) -> ViTForImageClassification:
    num_classes = len(label_encoder.classes_)
    id2label = {i: cls for i, cls in enumerate(label_encoder.classes_)}
    label2id = {cls: i for i, cls in enumerate(label_encoder.classes_)}

    vit_model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    original_classifier = vit_model.classifier
    vit_model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        original_classifier,
    )

    checkpoint_dir = resolve_checkpoint_dir()
    state_dict = load_file(str(checkpoint_dir / "model.safetensors"))
    missing_keys, unexpected_keys = vit_model.load_state_dict(state_dict, strict=False)

    print(f"Loaded model from: {checkpoint_dir}")
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    vit_model.to(DEVICE)
    vit_model.eval()
    return vit_model


class HFModelWrapper(nn.Module):
    def __init__(self, hf_model: ViTForImageClassification):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x):
        return self.hf_model(pixel_values=x).logits


def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def predict_test_set(vit_model, test_df, val_transforms):
    preds, labels = [], []
    for _, row in test_df.iterrows():
        image = np.array(Image.open(row["image"]).convert("RGB").resize((224, 224)))
        tensor = val_transforms(image=image)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = vit_model(pixel_values=tensor).logits.argmax(1).item()
        preds.append(pred)
        labels.append(int(row["label_enc"]))
    return np.array(preds), np.array(labels)


def pick_representative_sample(test_df, true_labels, vit_preds, class_idx):
    matching_indices = np.where((true_labels == class_idx) & (vit_preds == class_idx))[0]
    if len(matching_indices) == 0:
        matching_indices = np.where(true_labels == class_idx)[0]
    selected_idx = int(matching_indices[0])
    return test_df.iloc[selected_idx]["image"]


def compute_gradcam_heatmap(model, raw_image_path, label_idx, transform):
    target_layers = [model.hf_model.vit.encoder.layer[-2].layernorm_before]
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=vit_reshape_transform,
    )

    original_image = np.array(Image.open(raw_image_path).convert("RGB").resize((224, 224)))
    tensor_image = transform(image=original_image)["image"].unsqueeze(0).to(DEVICE)

    heatmap = cam(
        input_tensor=tensor_image,
        targets=[ClassifierOutputTarget(label_idx)],
        aug_smooth=True,
        eigen_smooth=True,
    )[0]

    heatmap = np.maximum(heatmap, 0)
    p_low, p_high = np.percentile(heatmap, [55, 99.5])
    heatmap = np.clip(heatmap, p_low, p_high)
    heatmap = (heatmap - p_low) / (p_high - p_low + 1e-8)
    return original_image, heatmap


def create_gradcam_overlay(raw_image_path, heatmap, alpha=0.6):
    original_image = Image.open(raw_image_path).convert("RGB").resize((224, 224))
    img = np.array(original_image, dtype=np.float32)

    heatmap_img = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255))
    heatmap_img = heatmap_img.resize((img.shape[1], img.shape[0]), resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img, dtype=np.float32) / 255.0

    jet = plt.get_cmap("jet")
    jet_heatmap = jet(heatmap_resized)[..., :3]
    jet_heatmap = (jet_heatmap * 255).astype(np.float32)

    activation_mask = (heatmap_resized > 0.35).astype(np.float32)[..., None]
    overlay = np.clip(
        img * (1.0 - alpha * activation_mask) + jet_heatmap * (alpha * activation_mask),
        0,
        255,
    ).astype(np.uint8)

    return img.astype(np.uint8), heatmap_resized, overlay


def show_gradcam_results():
    print(f"Using device: {DEVICE}")
    test_df, label_encoder = prepare_data()
    val_transforms = get_val_transforms()
    vit_model = load_trained_model(label_encoder)
    vit_preds, true_labels = predict_test_set(vit_model, test_df, val_transforms)

    wrapped_model = HFModelWrapper(vit_model)
    wrapped_model.eval()

    num_classes = len(label_encoder.classes_)
    fig, axes = plt.subplots(num_classes, 3, figsize=(15, 5 * num_classes))
    if num_classes == 1:
        axes = np.array([axes])

    for cls_idx, cls_name in enumerate(label_encoder.classes_):
        sample_path = pick_representative_sample(test_df, true_labels, vit_preds, cls_idx)
        _, heatmap = compute_gradcam_heatmap(
            wrapped_model,
            sample_path,
            cls_idx,
            val_transforms,
        )
        original, heatmap_vis, overlay = create_gradcam_overlay(sample_path, heatmap)

        axes[cls_idx, 0].imshow(original)
        axes[cls_idx, 0].set_title(f"Original - {cls_name}", fontsize=11, fontweight="bold")
        axes[cls_idx, 0].axis("off")

        axes[cls_idx, 1].imshow(heatmap_vis, cmap="jet")
        axes[cls_idx, 1].set_title(f"Grad-CAM Map - {cls_name}", fontsize=11, fontweight="bold")
        axes[cls_idx, 1].axis("off")

        axes[cls_idx, 2].imshow(overlay)
        axes[cls_idx, 2].set_title(f"Overlay - {cls_name}", fontsize=11, fontweight="bold")
        axes[cls_idx, 2].axis("off")

    plt.suptitle("ViT Grad-CAM: Disease-Affected Regions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    output_path = REPO_ROOT / "gradcam_output.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved Grad-CAM figure to: {output_path}")
    plt.show()


if __name__ == "__main__":
    show_gradcam_results()
