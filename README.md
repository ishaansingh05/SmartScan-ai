# SmartScan-ai
AI-based brain tumor detection using EfficientNetB2 with Grad-CAM for interpretable MRI classification.
# 🧠 SmartScan: Brain Tumor Detection using EfficientNetB2

## 📌 Overview
SmartScan is a deep learning-based system for detecting brain tumors from MRI scans.  
The model uses **EfficientNetB2 (transfer learning)** for classification and **Grad-CAM** for visual explanation of predictions.


## 🔄 Project Pipeline

- ### 1. Data Collection
The dataset consists of four classes of brain MRI images:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

This is a **multi-class classification problem**, where the model predicts the type of brain condition based on MRI scans.

---

### 2. Data Preprocessing
- Resized all images to **224×224** (required for EfficientNetB2)
- Normalized pixel values (0–255 → 0–1)
- Applied noise reduction techniques (Gaussian Blur)
- Converted images into numerical arrays for model input

---

### 3. Data Augmentation
To improve model generalization and prevent overfitting:
- Rotation
- Zoom
- Width & height shift
- Horizontal flipping  

This artificially increases dataset diversity and improves robustness.

---

### 4. Train-Test Split
- Training Set: 80%
- Testing Set: 20%

---

### 5. Model Architecture
- Used **EfficientNetB2 (pre-trained on ImageNet)**
- Applied **transfer learning**
- Froze base layers to retain learned features
- Added custom classification layers:
  - Global Average Pooling
  - Dense layers
- Output layer uses **Softmax activation** for multi-class classification (4 classes)

---

### 6. Model Training
- Loss Function: Categorical Crossentropy  
- Optimizer: Adam  
- Evaluation Metric: Accuracy  
---

### 7. Prediction
The model predicts one of the following classes:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

---

### 8. Explainable AI (Grad-CAM)
- Implemented **Grad-CAM** to visualize model decisions  
- Highlights important regions in MRI scans influencing predictions  
- Improves trust and interpretability of the model  

---

## 🛠 Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  

---

## 📊 Results
- Accuracy: 
- Model performs reliable classification on test data  

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/smartscan-brain-tumor-detection.git
cd smartscan-brain-tumor-detection
pip install -r requirements.txt
python train.py
python predict.py
