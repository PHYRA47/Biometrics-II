<div align="center">
  <h1><strong> 👤Biometrics Skill 6: Face Pose Estimation </strong></h1>

  https://github.com/user-attachments/assets/18bd4107-bc9d-4c5c-9664-3f9310d71120
</div>

This repository contains a Python-based GUI application for real-time and image-based face pose estimation, developed as part of Biometrics II course assignment

## ✨ Key Features

- 🧠 **Multiple Pose Estimation Models**: Custom `SVR` and `DenseNN` models, with easy integration of additional models for enhanced flexibility.

- 🖼️ **Multiple Recognition Modes**: Upload and analyze images for pose estimation, use real-time webcam feed for live pose detection, and process multiple faces in a single frame.

- 🎨 **Pose Visualization**: Annotated faces with 3D pose axes and facial landmarks.

- 🖥️ **Modern User Interface**: Clean and intuitive PyQt5-based GUI with tabbed navigation for switching between image upload and webcam modes.

## 🔍 How It Works & 🎛️ Technical Details

The system uses a combination of advanced techniques and tools to deliver accurate and efficient pose estimation:

- 🎯 **Face Detection**:  
  Utilizes MediaPipe FaceMesh for robust and reliable facial landmark detection in images and video streams.

- 🧠 **Pose Estimation Models**:  
  Employs machine learning models for precise pose prediction:  
  - `SVR`: Support Vector Regression models for pitch, yaw, and roll estimation.  
  - `DenseNN`: A dense neural network model for end-to-end pose estimation.  

- 🎨 **Image Processing & Annotation**:  
  Leverages OpenCV for image processing, pose annotation, and visualization of results with 3D axes and landmarks.  

- 🖥️ **Graphical Interface**:  
  Built with PyQt5, providing a clean, intuitive, and user-friendly GUI for seamless interaction and navigation.  

## 📝 Requirements

To run this project, ensure you have the following installed:

- **Python 3.9+**  
- **OpenCV**: For image processing and video capture.  
- **MediaPipe**: For facial landmark detection.  
- **PyQt5**: For the graphical user interface.  
- **NumPy**: For numerical operations and array handling.  
- **scikit-learn**: For SVR model support.  
- **TensorFlow/Keras**: For DenseNN model support.  

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## 📂 **Project Structure**
```bash
assets/                            # Folder for static assets (e.g., test images)
├── test_images/                   # Test images for pose estimation
│   ├── 1.png
│   ├── 2.jpg
│   └── 3.jpg
data/                              # Folder containing datasets and processed data
├── AFLW2000/                      # Where the AFLW2000 dataset is downloaded
├── data.npz                       # Processed data file
├── features.npz                   # Extracted features file
├── img_names.npz                  # Image names file
└── labels.npz                     # Labels file
models/                            # Folder containing pre-trained models
├── model_1_svm_best_params.npz    # Best parameters for SVR model
├── model_1_svm_best.pkl           # Trained SVR model
├── model_4_gbr_best_params.npz    # Best parameters for GBR model
├── model_4_gbr_best.pkl           # Trained GBR model
├── model_5_history.npz            # Training history for DenseNN model
├── model_5.keras                  # DenseNN model in Keras format
└── model_5.pkl                    # Trained DenseNN model
scripts/                           # Folder for evaluation and training scripts
├── evaluate.ipynb                 # Notebook for model evaluation
├── headPoseEstimation.py          # Script for head pose estimation
├── training.ipynb                 # Notebook for model training
└── webcam.py                      # Webcam implementation for pose estimation
src/                               # Source code for the application
├── __pycache__/                   # Python cache files
└── FacePoseEstimationTab.py       # Face pose estimation tab logic
main.py                            # Entry point for the application
README.md                          # Project documentation
requirements.txt                   # File listing project dependencies
```