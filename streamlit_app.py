import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

IMG_SIZE = 320
MODEL_PATH = Path("model/best_densenet_nih.pth")


class DenseNetMultiLabel(nn.Module):
    def __init__(self, num_classes: int = 14):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.features = base.features
        in_features = base.classifier.in_features
        self.classifier = nn.Linear(in_features, num_classes)
        for module in self.features.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad):
        self.gradients = grad.detach()

    def generate(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        out = self.model(x)
        score = out[:, class_idx].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM failed to capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNetMultiLabel(num_classes=len(LABELS)).to(device)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    grad_cam = GradCAM(model, model.features.denseblock4)
    return model, grad_cam, device


def preprocess_image(file_bytes: bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image)
    rgb = np.stack([arr, arr, arr], axis=-1).astype(np.uint8)
    tensor = val_transform(Image.fromarray(rgb)).unsqueeze(0)
    return rgb, tensor


def overlay_cam(rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 0.6, heatmap, 0.4, 0)


def extract_bbox_from_cam(cam: np.ndarray, threshold: float = 0.3):
    cam_uint8 = np.uint8(cam * 255)
    _, thresh = cv2.threshold(cam_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, x + w, y + h


st.set_page_config(page_title="Annotation-Free Bengali Vision-Language Model for Pathology Localization", layout="wide")
st.title("Annotation-Free Bengali Vision-Language Model for Pathology Localization APP")
st.write("Upload a chest X-ray image. The app shows top predictions and Grad-CAM.")
st.caption("Best results for frontal chest X-rays (PA/AP view). Non-chest images may produce unreliable predictions.")

ood_threshold = st.sidebar.slider(
    "Low-confidence warning threshold",
    min_value=0.30,
    max_value=0.90,
    value=0.55,
    step=0.01,
)
bbox_threshold = st.sidebar.slider(
    "BBox threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.01,
)

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        model, grad_cam, device = load_model()
        rgb, tensor = preprocess_image(uploaded.read())
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        top_idx = np.argsort(probs)[::-1][:5]
        best_idx = int(top_idx[0])
        best_prob = float(probs[best_idx])
        cam = grad_cam.generate(tensor.clone().detach().requires_grad_(True), best_idx)
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        pred_with_box = overlay_cam(rgb, cam)
        bbox = extract_bbox_from_cam(cam, threshold=bbox_threshold)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(pred_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if best_prob < ood_threshold:
            st.warning(
                "Low confidence prediction. This image may be out-of-distribution "
                "(e.g., not a frontal chest X-ray), so results may be unreliable."
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(rgb, caption="Input image", use_container_width=True)
        with c2:
            st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)
        with c3:
            st.image(
                pred_with_box,
                caption=f"Pred: {LABELS[best_idx]} ({best_prob:.3f})",
                use_container_width=True,
            )

        # st.subheader("Top 5 Predictions")
        # for i in top_idx:
        #     st.write(f"- {LABELS[int(i)]}: {float(probs[int(i)]):.4f}")
    except Exception as exc:
        st.error(f"Failed to run inference: {exc}")
