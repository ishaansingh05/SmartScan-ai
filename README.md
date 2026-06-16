<div align="center">

# 🧠 SmartScan: AI-Driven Brain Tumor Detection

### *Leveraging Transfer Learning & Explainable AI for Intelligent MRI Classification*

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/MobileNetV2-Transfer%20Learning-2E8B57?style=for-the-badge">
  <img src="https://img.shields.io/badge/GradCAM-Explainable%20AI-DC143C?style=for-the-badge">
  <img src="https://img.shields.io/badge/HuggingFace-Deployed-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge">
</p>

### 🚀 An end-to-end deep learning application that combines **Transfer Learning**, **Explainable AI**, and an intuitive web interface to classify brain MRI scans while visually explaining every prediction.

[🌐 Live Demo](https://huggingface.co/spaces/ishaan05/brain-tumor-classifier)

</div>

---

# 📸 Project Preview

<p align="center">
<img src="assets/homepage.png" width="95%">
</p>

> 📷 Replace the image above with the main screenshot of your application.

---

# 🌟 Overview

Early detection of brain tumors plays a critical role in improving treatment planning and patient outcomes. However, interpreting MRI scans requires significant medical expertise and can be time-consuming.

**SmartScan** is an AI-powered brain tumor classification system that assists this process by automatically analyzing MRI images using a fine-tuned **MobileNetV2** architecture based on **Transfer Learning**.

Unlike traditional deep learning models that simply output a prediction, SmartScan integrates **GradCAM (Gradient-weighted Class Activation Mapping)** to provide visual explanations, allowing users to understand *why* the model arrived at its decision.

The project bridges the gap between performance and interpretability by combining state-of-the-art computer vision techniques with an intuitive user interface.

---

# 🎯 Key Highlights

- 🧠 AI-powered classification of brain MRI scans
- 🚀 Transfer Learning using MobileNetV2
- 🔥 Explainable AI with GradCAM visualization
- 🎚️ Adjustable heatmap overlay intensity
- 📥 Downloadable GradCAM outputs
- 🖼️ Support for JPG, PNG, JPEG and NumPy (`.npy`) images
- ⚡ Fast real-time inference
- 🎨 Modern dark-themed responsive interface
- 🌐 Deployable on Hugging Face Spaces
- 🔬 Research-oriented transparency for medical imaging

---

# 💡 Why SmartScan?

Most medical image classification models behave like black boxes—they provide predictions without revealing the reasoning behind them.

SmartScan was designed with a different philosophy.

Rather than only identifying the tumor category, it enables users to **visualize the regions of the MRI scan that influenced the prediction** through GradCAM heatmaps.

Additionally, users can dynamically adjust the heatmap intensity, allowing seamless comparison between the original MRI and the model's activation regions.

This emphasis on explainability makes the system more transparent, trustworthy, and educational.

---

# 🧠 Brain Tumor Categories

The model classifies MRI scans into four categories:

| Category | Description |
|------------|------------------------------------------------|
| 🔴 Glioma | Tumor arising from glial cells in the brain |
| 🟠 Meningioma | Tumor originating from the meninges |
| 🟢 No Tumor | MRI scan without detectable tumor |
| 🟡 Pituitary Tumor | Tumor affecting the pituitary gland |

---

# 🚀 Transfer Learning

Training deep convolutional neural networks entirely from scratch typically requires millions of labeled images and extensive computational resources.

SmartScan instead employs **Transfer Learning**, where a pretrained **MobileNetV2** model serves as the backbone for feature extraction.

The pretrained network has already learned rich visual representations from millions of images on ImageNet.

By fine-tuning these learned representations for medical imaging, the model can efficiently adapt to brain MRI classification while requiring significantly less data and training time.

## Advantages

- Faster convergence
- Improved feature extraction
- Better performance on limited datasets
- Lower computational requirements
- Reduced overfitting
- Stronger generalization on unseen images

---

# 🔥 Explainable AI with GradCAM

Artificial Intelligence should not only make predictions—it should explain them.

SmartScan integrates **Gradient-weighted Class Activation Mapping (GradCAM)** to visualize the regions of the MRI scan that contribute most strongly to the final prediction.

Instead of functioning as a black box, the model highlights clinically relevant areas directly on the original image.

### Interactive Heatmap Control

One of SmartScan's distinguishing features is its **interactive heatmap intensity slider**.

Users can dynamically adjust the GradCAM overlay strength, allowing them to:

- Better inspect highlighted regions
- Compare the original MRI with the activation map
- Improve visualization clarity
- Generate customized explainability outputs

The resulting heatmap can also be downloaded for documentation or further analysis.

---

# 🏗️ System Architecture

```text
                    Brain MRI Scan
                           │
                           ▼
                  Image Preprocessing
            Resize → Normalize → RGB Conversion
                           │
                           ▼
          MobileNetV2 Backbone (Transfer Learning)
                           │
                           ▼
             Fine-Tuned Classification Layers
                           │
                           ▼
                 Brain Tumor Prediction
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      Predicted Category          GradCAM Generation
             │                           │
             └─────────────┬─────────────┘
                           ▼
             Interactive Explainable Output
```

---

# ⚙️ Model Pipeline

```text
MRI Upload
      │
      ▼
Image Preprocessing
      │
      ▼
Transfer Learning
(MobileNetV2)
      │
      ▼
Fine-Tuned Classification
      │
      ▼
Predicted Tumor Category
      │
      ▼
GradCAM Heatmap
      │
      ▼
Adjustable Overlay
      │
      ▼
Download Result
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

# 🖼️ User Interface

## 🏠 Home Screen

<p align="center">
<img src="assets/homepage.png" width="90%">
</p>

---

## 🔍 Prediction Output

<p align="center">
<img src="assets/prediction.png" width="90%">
</p>

---

## 🔥 GradCAM Visualization

<p align="center">
<img src="assets/gradcam.png" width="90%">
</p>

The GradCAM visualization highlights the regions that contributed most strongly to the model's decision while allowing users to customize overlay intensity through an interactive slider.
---

# 📊 Model Performance

SmartScan was trained using a **Transfer Learning** approach built upon **MobileNetV2**, enabling efficient feature extraction and strong generalization even with comparatively limited medical imaging data.

The model's performance should be evaluated using multiple metrics rather than accuracy alone, as balanced evaluation is crucial in medical AI applications.

## 📈 Training Accuracy

<p align="center">
<img src="assets/accuracy.png" width="85%">
</p>

> Replace `accuracy.png` with your training/validation accuracy graph.

---

## 📉 Training Loss

<p align="center">
<img src="assets/loss.png" width="85%">
</p>

> Replace `loss.png` with your training/validation loss graph.

---

## 🎯 Confusion Matrix

<p align="center">
<img src="assets/confusion_matrix.png" width="80%">
</p>

> Replace `confusion_matrix.png` with your actual confusion matrix.

---

## 📋 Performance Metrics

| Metric | Value |
|----------|---------|
| Test Accuracy | **XX.XX%** |
| Precision | **XX.XX%** |
| Recall | **XX.XX%** |
| F1-Score | **XX.XX%** |

> Update the values above with your final evaluation results.

---

# 🗂️ Dataset

The model was trained on a curated collection of **Brain MRI images** containing four diagnostic categories:

- 🔴 Glioma
- 🟠 Meningioma
- 🟢 No Tumor
- 🟡 Pituitary Tumor

The images were resized to **224 × 224 RGB** before being passed through the neural network.

## Preprocessing Pipeline

- Image resizing
- RGB conversion
- Pixel normalization
- Tensor preparation
- Model inference

The preprocessing pipeline ensures consistency across all inputs and optimizes compatibility with the MobileNetV2 architecture.

---

# 🚀 Installation

## Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SmartScan.git
```

## Navigate into the project

```bash
cd SmartScan
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

The application will launch locally in your browser.

---

# 🌐 Live Demo

The project is deployed on **Hugging Face Spaces**, allowing users to interact with the model directly without local installation.

### 🔗 Live Application

https://huggingface.co/spaces/ishaan05/brain-tumor-classifier

---

# 💻 How It Works

### Step 1 — Upload MRI

Upload a brain MRI scan in one of the supported formats:

- JPG
- JPEG
- PNG
- NumPy (`.npy`)

---

### Step 2 — Automatic Preprocessing

The application automatically:

- Resizes the image
- Converts it to RGB
- Normalizes pixel values
- Prepares the tensor for inference

---

### Step 3 — AI Prediction

The fine-tuned MobileNetV2 model analyzes the MRI and predicts one of the four supported classes.

---

### Step 4 — Explainability

GradCAM generates an activation heatmap highlighting the regions that contributed most strongly to the prediction.

Unlike static implementations, SmartScan allows users to **adjust the heatmap intensity in real time**, making it easier to inspect subtle patterns while preserving visibility of the original MRI.

---

### Step 5 — Export Results

The generated GradCAM visualization can be downloaded for documentation, research, or educational purposes.

---

# 🎨 Technologies Used

## Programming Language

- Python

## Deep Learning

- TensorFlow
- Keras
- MobileNetV2
- Transfer Learning

## Computer Vision

- OpenCV
- NumPy
- Pillow

## Explainable AI

- GradCAM

## Visualization

- Matplotlib

## Frontend

- Streamlit

## Deployment

- Hugging Face Spaces

---

# 🏆 Technical Highlights

- Fine-tuned **MobileNetV2** using Transfer Learning
- Implemented **GradCAM** for explainable predictions
- Built a fully interactive medical imaging interface
- Developed customizable heatmap visualization with adjustable intensity
- Added support for multiple image formats including `.npy`
- Integrated downloadable GradCAM outputs
- Optimized for lightweight deployment on Hugging Face Spaces

---

# 🎓 Key Concepts Demonstrated

This project showcases practical implementation of:

- Deep Learning
- Convolutional Neural Networks
- Transfer Learning
- Explainable AI (XAI)
- Medical Image Classification
- Computer Vision
- Model Fine-Tuning
- Interactive AI Applications
- Human-Centered AI
- Deployment of ML Models

---

# 🔬 Why Explainability Matters

High-performing models alone are insufficient in healthcare.

Medical professionals require insight into *why* a prediction was made before trusting automated systems.

By integrating GradCAM, SmartScan transforms neural network outputs into interpretable visualizations, enabling users to inspect the decision-making process rather than accepting predictions blindly.

This improves transparency, accountability, and confidence in AI-assisted diagnosis.

---

# 📈 Future Scope

SmartScan provides a strong foundation that can be extended into more advanced clinical applications.

Potential improvements include:

- 🩺 DICOM image support
- 🧠 Brain tumor segmentation
- 📊 3D MRI volume processing
- 🤖 AI-generated diagnostic summaries
- 🌍 Multi-language support
- ☁️ REST API for remote inference
- 🏥 PACS integration for hospitals
- 🔬 Ensemble deep learning models
- 📋 Automated PDF report generation
- 📱 Mobile application deployment

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve SmartScan, feel free to:

- Fork the repository
- Create a feature branch
- Commit your changes
- Submit a pull request

Constructive suggestions and improvements are always appreciated.

---

# 📜 License

This project is released under the **MIT License**.

You are free to use, modify, and distribute the software in accordance with the license terms.

---

# ⚠️ Medical Disclaimer

SmartScan is intended **solely for educational, research, and demonstration purposes**.

It is **not a certified medical device** and should **not be used for clinical diagnosis or treatment decisions**.

Any medical concerns should always be evaluated by qualified healthcare professionals.

---

# 👨‍💻 About the Project

SmartScan was developed to explore the intersection of **Deep Learning**, **Transfer Learning**, and **Explainable AI** within the field of medical image analysis.

By combining efficient CNN architectures with transparent visualization techniques, the project demonstrates how modern AI systems can move beyond prediction accuracy to provide meaningful and interpretable insights.

The objective is not only to classify MRI scans but also to make those predictions understandable and trustworthy.

---

<div align="center">

# ⭐ Star this repository if you found it useful!

### If SmartScan helped you learn something new or inspired your own projects, consider giving it a ⭐ on GitHub.

---

### 🧠 *Making Medical AI More Transparent with Transfer Learning & Explainable AI.*

**Built with ❤️ using Python, TensorFlow, MobileNetV2, and GradCAM**

</div>
