# Annotation-Free Bengali Vision-Language Model for Pathology Localization

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/SABBiR1107/Annotation-Free-Bengali-Vision-Language-Model-for-Pathology-Localization/main/streamlit_app.py)

A PyTorch and Streamlit-based web application for chest X-ray pathology classification and localization. This application leverages a DenseNet-121 multi-label model combined with Grad-CAM (Gradient-weighted Class Activation Mapping) to automatically localize areas of interest (pathologies) without requiring bounding box annotations during training.

---

## 🌟 Key Features

- **Multi-Label Classification**: Predicts the likelihood of 14 chest pathologies (including *Atelectasis*, *Cardiomegaly*, *Effusion*, *Infiltration*, *Mass*, *Nodule*, *Pneumonia*, *Pneumothorax*, and more).
- **Grad-CAM Heatmap Visualization**: Highlights the regions of the chest X-ray that contributed most to the model's top prediction.
- **Automatic Bounding Box Localization**: Dynamically draws a bounding box around the localized pathology using contour detection on the Grad-CAM heatmaps.
- **Interactive Thresholding**:
  - **Low-Confidence Warning**: Adjustable warning threshold to detect out-of-distribution or low-confidence images.
  - **BBox Threshold**: Custom thresholding to control the size and sensitivity of the generated bounding boxes.

---

## 🛠️ Project Structure

- `streamlit_app.py` - Core Streamlit application containing the UI, model inference, Grad-CAM generation, and bounding box localization logic.
- `model/` - Contains the model checkpoints (specifically `best_densenet_nih.pth`).
- `pyproject.toml` & `uv.lock` - Dependency specifications managed by `uv`.
- `main.py` - Minimal greeting entrypoint script.
- `NIH.ipynb` - Jupyter notebook containing exploratory analysis / training pipelines.

---

## 🚀 How to Run

### Option 1: Using `uv` (Recommended)
If you have the `uv` package manager installed, you can start the application directly in one command:
```bash
uv run streamlit run streamlit_app.py
```

### Option 2: Using the Local Virtual Environment (`.venv`)
If your virtual environment is already set up, activate it and run streamlit:

**Mac / Linux:**
```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

**Windows:**
```cmd
.venv\Scripts\activate
streamlit run streamlit_app.py
```

---

## 📦 Requirements & Dependencies
Managed via `pyproject.toml`. Main dependencies include:
- `Python >= 3.11`
- `torch` & `torchvision`
- `streamlit`
- `opencv-python`
- `numpy`, `pandas`, `matplotlib`, `pillow`
