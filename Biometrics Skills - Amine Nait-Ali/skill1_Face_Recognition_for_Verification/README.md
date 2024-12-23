<div align="center">
  <h1><strong> 👤Biometrics Skill 1: Face Recognition for Verification🔍 </strong></h1>
  
  https://github.com/user-attachments/assets/eb8b5ad0-29cd-408c-a0ec-1ebf7e64e5e0
</div>

This folder contains the Python-based GUI application developed as part of the Biometrics II course assignment. 

## ✨ Features

- 🖼️ **Image-to-Image Comparison**: Upload and compare any two face images
- 📷 **Real-time Webcam Recognition**: Compare live webcam feed with reference images
- 🔄 **Multiple Model Support**: Choose between different deep learning models:
  - `VGG19`
  - `Xception`
  - `ResNet`
- 🎨 **User-Friendly Interface**: Clean and intuitive PyQt5-based GUI
- 📊 **Detailed Results**: Get similarity scores and match confirmations

## 🚀 Usage

1. Launch the application:
```bash
python main.py
```
2. For Image Comparison:
   - 🖼️ Select your preferred model
   - 📁 Upload the first image using "Upload First Image"
   - 📁 Upload the second image using "Upload Second Image"
   - ✨ Click "Verify Faces" to see results

3. For Webcam Recognition:
   - 📷 Upload a reference image
   - 🎥 Click "Start Webcam" to begin camera feed
   - ✨ Click "Verify Face" to compare with reference image
  
## 🔍 How It Works

The system uses a combination of:
1. 🎯 Haar Cascade Classifiers for face detection
2. 🧠 Deep learning models for feature extraction
3. 📐 Cosine similarity metrics for face matching
4. 🖥️ PyQt5 for the graphical interface

## 🎛️ Technical Details

- 🔍 Face Detection: OpenCV's Haar Cascade Classifier
- 🧮 Feature Extraction: Deep CNN models (VGG19/Xception/ResNet)
- 📏 Similarity Metrics: 
  - Euclidean Distance
  - Cosine Similarity
- 🎚️ Default Threshold: 0.30 (adjustable)

## 📂 **Project Structure**
```
├── models/        # Pre-trained models
├── funciton.py    # Helper functions
├── main.py        # Main application script
├── requirements.txt
└── README.md
```

