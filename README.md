# Pneumonia Detection in Chest X-Rays Using Feature-Level Ensemble Learning

This project implements an AI-powered diagnostic system for pneumonia detection from chest X-ray images using a feature-level ensemble of convolutional neural networks (CNNs). It integrates multiple pre-trained models to improve diagnostic accuracy and interpretability, supporting clinical decision-making.

## 🔬 Project Overview

Pneumonia remains a global health concern, especially in areas lacking sufficient radiology expertise. This project addresses this challenge by developing a deep learning model that automates pneumonia diagnosis from chest X-rays with high precision.

### 🎯 Objectives
- Build an ensemble CNN model combining **VGG19**, **EfficientNetB0**, and **DenseNet121**
- Improve prediction accuracy via **feature-level fusion**
- Generate **Grad-CAM** heatmaps for interpretability
- Deploy the model as a **user-friendly web application** for healthcare use

## 📊 Key Features
- **Accuracy:** 93.1%
- **Precision:** 93.71%
- **Recall:** 95.38%
- **F1 Score:** 94.54%
- Grad-CAM visualizations for region-specific attention
- Robust performance across different dataset splits

## 🧠 Model Architecture
The system uses a weighted feature fusion strategy:
- Extracts features from VGG19, EfficientNetB0, and DenseNet121
- Assigns higher weights to stronger feature extractors
- Combines features before classification using a softmax layer

## 🗃 Dataset
- **Source:** [Kaggle – Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:** Pneumonia, Normal
- **Preprocessing:** 
  - Resizing to 224×224
  - Normalization
  - Data augmentation (rotation, flipping, zoom)

## ⚙️ Tech Stack
- Python 3.6+
- TensorFlow & Keras
- NumPy, Pandas
- OpenCV, Matplotlib
- Grad-CAM for interpretability
- Google Colab (GPU support)

## 📸 Sample Output

<p align="center">
  <img src="assets/xray_sample.png" width="400"/>
  <img src="assets/gradcam_heatmap.png" width="400"/>
</p>

## 💡 Future Enhancements
- Use multimodal data (X-rays + patient info)
- Explore advanced ensemble methods like stacking or boosting
- Improve transparency with Explainable AI (XAI) techniques

## 👩‍💻 Contributors
- Hruthik Krishna Pavarala - 228W1A12B4  
- Ganesh Sankar Kumar Karanam - 228W1A1295  
- Jai Surya Kanna - 228W1A1294  
- Guide: K. Pranathi (Assistant Professor, Dept. of IT)

## 📜 License
This project is for academic and research purposes only. Contact the authors for extended use.
