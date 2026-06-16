<div align="center">

# 🧠 SmartScan: AI-Driven Brain Tumor Detection

### *Harnessing Transfer Learning & Explainable AI for Intelligent MRI Classification*

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/MobileNetV2-Transfer%20Learning-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/GradCAM-Explainable%20AI-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b?style=for-the-badge" />
</p>

*A deep learning-powered medical imaging application that classifies brain MRI scans into four categories while providing interpretable visual explanations through GradCAM.*

</div>

---

# 📸 Project Preview

> **Replace the image below with a screenshot of your application's home page.**

```text
assets/homepage.png
```

<p align="center">
  <img src="assets/homepage.png" width="90%">
</p>

---

# 🚀 Live Demo

> Replace the link below with your deployed Hugging Face Space.

**🔗 Try SmartScan Live:**  
`https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE`

---

# 📖 Overview

Brain tumor diagnosis from MRI scans is a challenging task that requires expertise and careful analysis. **SmartScan** leverages **Transfer Learning** and **Explainable AI** to assist in this process by automatically classifying MRI images into multiple tumor categories while visually explaining the reasoning behind each prediction.

Unlike conventional image classifiers that act as black boxes, SmartScan integrates **GradCAM (Gradient-weighted Class Activation Mapping)** to highlight the regions of the MRI scan that most influenced the model's prediction, making the system more transparent and interpretable.

The application provides an intuitive interface where users can upload MRI images and instantly receive:

- AI-powered tumor classification
- Confidence scores for every class
- Visual heatmap explanations
- Downloadable GradCAM overlays

---

# ✨ Key Features

## 🧠 Intelligent Brain Tumor Classification

Classifies MRI scans into four categories:

- 🔴 Glioma
- 🟠 Meningioma
- 🟢 No Tumor
- 🟡 Pituitary Tumor

---

## 🚀 Transfer Learning with MobileNetV2

Instead of training a convolutional neural network from scratch, SmartScan utilizes **MobileNetV2** pretrained on ImageNet and fine-tunes it for brain MRI classification.

Benefits include:

- Faster convergence
- Better feature extraction
- Improved performance on limited medical datasets
- Lower computational cost
- Reduced overfitting

Transfer learning enables the model to reuse rich visual representations learned from millions of images while adapting specifically to medical imaging.

---

## 🔥 Explainable AI with GradCAM

SmartScan doesn't just make predictions—it explains them.

GradCAM generates heatmaps that highlight the regions of the MRI image responsible for the final classification, improving transparency and helping users understand the model's decision-making process.

---

## 📊 Confidence Visualization

Instead of showing only the predicted class, SmartScan displays probability scores for every category through intuitive confidence bars.

---

## 📥 Downloadable Heatmaps

Users can export GradCAM visualizations for documentation, analysis, or educational purposes.

---

## 🖼️ Multiple Input Formats

Supports:

- JPG
- JPEG
- PNG
- NumPy (`.npy`) arrays

---

# 🏗️ System Architecture

```text
                  MRI Scan
                      │
                      ▼
             Image Preprocessing
          Resize → Normalize → RGB
                      │
                      ▼
        MobileNetV2 (Transfer Learning)
                      │
                      ▼
          Fine-Tuned Classification Head
                      │
                      ▼
            Softmax Probability Output
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
  Tumor Prediction            GradCAM Generator
        │                           │
        └─────────────┬─────────────┘
                      ▼
        Interactive Web Visualization
```

---

# 📂 Project Structure

```text
SmartScan/
│
├── app.py
├── best_brain_tumor_model.keras
├── requirements.txt
├── README.md
│
├── assets/
│   ├── homepage.png
│   ├── prediction.png
│   ├── gradcam.png
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   └── demo.gif
│
└── notebooks/
```

---

# 🧪 Classification Categories

| Class | Description |
|----------|------------------------------------------------|
| 🔴 Glioma | Tumor originating from glial cells |
| 🟠 Meningioma | Tumor arising from the meninges |
| 🟢 No Tumor | MRI scan without detectable tumor |
| 🟡 Pituitary | Tumor affecting the pituitary gland |

---

# 🖼️ User Interface

## Home Screen

> Replace with:

```text
assets/homepage.png
```

<p align="center">
<img src="assets/homepage.png" width="90%">
</p>

---

## Prediction Output

> Replace with:

```text
assets/prediction.png
```

<p align="center">
<img src="assets/prediction.png" width="90%">
</p>

---

## GradCAM Visualization

> Replace with:

```text
assets/gradcam.png
```

<p align="center">
<img src="assets/gradcam.png" width="90%">
</p>

---

# 📈 Model Performance

## Training Accuracy

> Replace with:

```text
assets/accuracy.png
```

<p align="center">
<img src="assets/accuracy.png" width="80%">
</p>

---

## Training Loss

> Replace with:

```text
assets/loss.png
```

<p align="center">
<img src="assets/loss.png" width="80%">
</p>

---

## Confusion Matrix

> Replace with:

```text
assets/confusion_matrix.png
```

<p align="center">
<img src="assets/confusion_matrix.png" width="80%">
</p>

---

## 📊 Performance Summary

> Replace the placeholders below with your actual metrics.

| Metric | Value |
|----------|--------|
| Test Accuracy | XX.XX% |
| Precision | XX.XX% |
| Recall | XX.XX% |
| F1 Score | XX.XX% |

---

# 🔬 Why Transfer Learning?

Training a deep convolutional neural network from scratch requires enormous datasets and computational resources.

SmartScan instead adopts **Transfer Learning**, where a pretrained MobileNetV2 model serves as a feature extractor before being fine-tuned on brain MRI images.

| Training from Scratch | Transfer Learning |
|-----------------------|------------------|
| Huge dataset required | Works well on limited datasets |
| Slow convergence | Faster training |
| High computation | Efficient |
| Higher overfitting risk | Better generalization |
| Learns features from zero | Reuses pretrained knowledge |

This approach significantly improves practical performance while reducing training time.

---

# 🔥 Explainability with GradCAM

Medical AI systems should provide transparency alongside predictions.

GradCAM computes gradients flowing into the final convolutional layer and projects them back onto the original image, producing a localization heatmap that reveals where the network focused while making its prediction.

Benefits include:

- Increased trust in predictions
- Better interpretability
- Visual verification
- Research and educational value

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/SmartScan.git
```

Move into the project:

```bash
cd SmartScan
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# 💻 Usage

1. Launch the application.
2. Upload an MRI scan.
3. The model preprocesses the image.
4. MobileNetV2 performs inference.
5. The predicted class and confidence scores are displayed.
6. GradCAM generates an explainability heatmap.
7. Download the visualization if desired.

---

# 🛠️ Tech Stack

### Languages

- Python

### Deep Learning

- TensorFlow
- Keras
- MobileNetV2
- Transfer Learning

### Computer Vision

- OpenCV
- Pillow
- NumPy

### Explainability

- GradCAM

### Visualization

- Matplotlib

### Frontend

- Streamlit

### Deployment

- Hugging Face Spaces

---

# 🎯 Future Improvements

- Support for DICOM images
- Brain tumor segmentation
- 3D MRI volume analysis
- Clinical report generation
- Multi-modal medical imaging
- Confidence calibration
- Attention map comparison
- Ensemble deep learning models

---

# 📚 Learning Outcomes

This project demonstrates practical implementation of:

- Deep Learning
- Medical Image Classification
- Transfer Learning
- Explainable AI
- Computer Vision
- CNN Fine-Tuning
- Interactive AI Applications
- Model Deployment
- Human-Centered AI

---

# ⚠️ Disclaimer

SmartScan is developed **for educational and research purposes only**.

It should **not** be used as a substitute for professional medical diagnosis, clinical evaluation, or healthcare decision-making.

Always consult qualified medical professionals for diagnosis and treatment.

---

<div align="center">

## ⭐ If you found this project helpful, consider giving it a star!

*Built with Deep Learning, Transfer Learning, and Explainable AI.*

</div>
