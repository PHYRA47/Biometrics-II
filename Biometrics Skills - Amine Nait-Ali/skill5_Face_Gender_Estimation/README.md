<div align="center">
  <h1><strong> 👤Biometrics Skill 5: Face Gender Estimation </strong></h1>

    https://github.com/user-attachments/assets/e36c9998-5bc5-495c-9011-1501b0662045
</div>

This repository contains a Python-based GUI application for real-time and image-based gender estimation, developed as part of the Biometrics II course assignment. It builds on the previous age estimation task, utilizing the same dataset and models but focusing on gender classification.

## ✨ Key Features

- 🧠 **Multiple Gender Estimation Models**: Custom `ResNet50` and `Custom CNN` models, with easy integration of additional models for enhanced flexibility.

- 🖼️ **Multiple Recognition Modes**: Upload and analyze images for gender estimation, use real-time webcam feed for live gender detection, and process multiple faces in a single frame.

- 🎨 **Gender Visualization**: Annotated faces with gender labels and bounding boxes.

- 🖥️ **Modern User Interface**: Clean and intuitive PyQt5-based GUI with tabbed navigation for switching between image upload and webcam modes.

## 🔍 How It Works & 🎛️ Technical Details

- 🎯 **Face Detection**:  
  Utilizes OpenCV's Haar Cascade Classifier for robust and reliable face detection in images and video streams.

- 🧠 **Gender Estimation Models**:  
  Employs deep learning models trained on the [`UTKFace`](https://www.kaggle.com/datasets/jangedoo/utkface-new) dataset for precise gender prediction:  
  - `ResNet50`: A pre-trained model fine-tuned specifically for gender estimation.  
  - `Custom CNN`: A lightweight custom CNN model optimized for binary classification.

- 🎨 **Image Processing & Annotation**:  
  Leverages OpenCV for image processing, face annotation, and visualization of results with bounding boxes and gender labels.

- 🖥️ **Graphical Interface**:  
  Built with PyQt5, providing a clean, intuitive, and user-friendly GUI for seamless interaction and navigation.

## 📝 Requirements

To run this project, ensure you have the following installed:

- **Python 3.10+**  
- **OpenCV**: For image processing and face detection.  
- **TensorFlow==2.18**: For loading and using deep learning models.  
- **PyQt5**: For the graphical user interface.  
- **NumPy**: For numerical operations and array handling.  

Install the required dependencies using pip:

```bash
pip install -r requirements.txt

## 📂 **Project Structure**
```
assets/                            # Folder for static assets (e.g., images, icons)
models/                            # Folder containing pre-trained models
├── gender_estimation/             # Gender estimation models
│   ├── custom_CNN/                
│   │   └── custom_CNN.keras       # Relocate the model here before running
│   └── custom_ResNet50/           
│       └── custom_ResNet50.keras  # Relocate the model here before running
scripts/    
├── evaluate.ipynb                 # Evaluation logic including metrices and confusion matrices                       
├── training.ipynb                 # Script for training models
└── webcam.py                      # Webcam implementation for gender estimation
src/                               # Source code for the application
└──  GenderEstimationTab.py        # Gender estimation tab logi
main.py                            # Entry point for the application
README.md                          # Project documentation
requirements.txt                   # File listing project dependencies
```