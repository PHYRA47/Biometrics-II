<div align="center">
  <h1><strong> 👤Biometrics Skill 3: Facial Expression Estimation </strong></h1>
  
  https://github.com/user-attachments/assets/52b92302-bcdb-403b-9580-960e7f4ee091
</div>

This repository contains a Python-based GUI application for real-time and image-based face emotion recognition, as part of the Biometrics II course assignment.

## ✨ Key Features

- 🧠 **Multiple Emotion Recognition Models**: custome `MobileNetV2` and  Sequential CNN models, and easy integration of additional models for enhanced flexibility.

- 🖼️ **Multiple Recognition Modes**: Upload and analyze images for emotion detection, use real-time webcam feed for live emotion recognition, and process multiple faces in a single frame.

- 🎨 **Emotion Visualization**: Annotated faces with emotion labels and color-coded bounding boxes.

- 🖥️ **Modern User Interface**: Clean and intuitive PyQt5-based GUI with tabbed navigation for switching between image upload and webcam modes.

## 🔍 How It Works & 🎛️ Technical Details

The system uses a combination of advanced techniques and tools to deliver accurate and efficient emotion recognition:

- 🎯 **Face Detection**:  
  Utilizes OpenCV's Haar Cascade Classifier for robust and reliable face detection in images and video streams.

- 🧠 **Emotion Classification**:  
  Employs deep learning models for precise emotion classification:  
  - `MobileNetV2`: A pre-trained model fine-tuned specifically for emotion recognition.  
  - `Sequential CNN`: A custom CNN model trained on emotion datasets for lightweight and efficient performance.  

- 🎭 **Emotion Categories**:  
  Detects and classifies emotions into seven categories:  
  Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.  

- 🎨 **Image Processing & Annotation**:  
  Leverages OpenCV for image processing, face annotation, and visualization of results with color-coded bounding boxes and labels.  

- 🖥️ **Graphical Interface**:  
  Built with PyQt5, providing a clean, intuitive, and user-friendly GUI for seamless interaction and navigation. 

## 📝 Requirements

To run this project, ensure you have the following installed:

- **Python 3.9+**  
- **OpenCV**: For image processing and face detection.  
- **TensorFlow==2.10**: For loading and using deep learning models.  
- **PyQt5**: For the graphical user interface.  
- **NumPy==1.26**: For numerical operations and array handling.  

Install the required dependencies using pip:

```bash
pip install opencv-python tensorflow==2.10 pyqt5 numpy<2
```
Alternatively:

```bash
pip install -r requirements.txt
```

## 📂 **Project Structure**
```
assets/
models/
├── mobilenetv2/
│ ├── mobilenetv2_history.npz       # Training history for MobileNetV2
│ ├── mobilenetv2.h5                # MobileNetV2 model weights
│ └── mobilenetv2.keras
├── sequentialCNN/
│ └── sequentialCNN.h5
scripts/
├── evaluate_mobilenetv2.ipynb      # Notebook for evaluating MobileNetV2
├── evaluate_sequentialCNN.ipynb    # Notebook for evaluating Sequential CNN
├── preprocess.py                   # Script for preprocessing data
├── training.py                     # Script for training models
├── webcam_mobilenetv2.py           # Webcam implementation for MobileNetV2
└── webcam_sequentialCNN.py         # Webcam implementation for Sequential CNN
src/
├── EmotionRecognizer.py
├── FERTab.py
├── MainWindow.py
├── complete_main.py
└── main.py                         # Entry for the application 
README.md
requirements.txt
```