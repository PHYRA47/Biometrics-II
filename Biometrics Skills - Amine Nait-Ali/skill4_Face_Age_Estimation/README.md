<div align="center">
  <h1><strong> 👤Biometrics Skill 4: Face Age Estimation </strong></h1>
  
  https://github.com/user-attachments/assets/d5e2dfb3-a539-4d8d-95c3-977207a417cc
</div>

This repository contains a Python-based GUI application for real-time and image-based age estimation, as part of the Biometrics II course assignment.

## ✨ Key Features

- 🧠 **Multiple Age Estimation Models**: Custom `ResNet50` and `Custom CNN` models, with easy integration of additional models for enhanced flexibility.

- 🖼️ **Multiple Recognition Modes**: Upload and analyze images for age estimation, use real-time webcam feed for live age detection, and process multiple faces in a single frame.

- 🎨 **Age Visualization**: Annotated faces with age labels and bounding boxes.

- 🖥️ **Modern User Interface**: Clean and intuitive PyQt5-based GUI with tabbed navigation for switching between image upload and webcam modes.

## 🔍 How It Works & 🎛️ Technical Details

- 🎯 **Face Detection**:  
  Utilizes OpenCV's Haar Cascade Classifier for robust and reliable face detection in images and video streams.

- 🧠 **Age Estimation Models**:  
  Employs deep learning models trained on the [`UTKFace`](https://www.kaggle.com/datasets/jangedoo/utkface-new) dataset for precise age prediction:  
  - `ResNet50`: A pre-trained model fine-tuned specifically for age estimation.  
  - `Custom CNN`: A lightweight custom CNN model.  

- 🎨 **Image Processing & Annotation**:  
  Leverages OpenCV for image processing, face annotation, and visualization of results with bounding boxes and age labels.  

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
```

## 📂 **Project Structure**
```
assets/                            # Folder for static assets (e.g., images, icons)
models/                            # Folder containing pre-trained models
├── age_gender_estimation/         # Age and gender estimation models
│   ├── custom_CNN/                
│   │   └── custom_CNN.keras       # Custom CNN model in Keras format
│   └── custom_ResNet50/           
│       └── custom_ResNet50.keras  # Custom ResNet50 model in Keras format
scripts/                           
├── training.ipynb                 # Script for training models
└── webcam.py                      # Webcam implementation for Custom ResNet50
src/                               # Source code for the application
├── AgeEstimationTab.py            # Age estimation tab logic
├── EmotionRecognitionTab.py       # Emotion recognition tab logic
└── GenderEstimationTab.py         # Gender estimation tab logic     
main.py                            # Main window logic and Entry point for the application
README.md                          # Project documentation
requirements.txt                   # File listing project dependencies
```
