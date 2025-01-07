<div align="center">
  <h1><strong> 👤Biometrics Skill : Human Body Pose Estimation </strong></h1>

  https://github.com/user-attachments/assets/c3093945-7b3a-4c94-818f-2ae9ab7150d4
</div>

This repository contains a Python-based GUI application for real-time and image-based body pose estimation as part of Biometrics II course assignment. The system estimates body keypoints and their connections using OpenCV DNN and MoveNet models. It supports image upload, video processing, and real-time webcam feed.

## ✨ Key Features

- 🧠 **Multiple Pose Estimation Models**: Supports `OpenCV DNN` and `MoveNet` models for accurate body pose estimation.

- 🖼️ **Multiple Recognition Modes**: 
  - Upload and analyze images for pose estimation.
  - Process video files for pose detection.
  - Use real-time webcam feed for live pose estimation.

- 🎨 **Pose Visualization**: Annotated body keypoints and connections with customizable display options.

- 🖥️ **Modern User Interface**: Clean and intuitive PyQt5-based GUI with tabbed navigation for switching between image, video, and webcam modes.

## 🔍 How It Works & 🎛️ Technical Details

The system uses a combination of advanced techniques and tools to deliver accurate and efficient pose estimation:

- 🎯 **Pose Estimation Models**:  
  - **OpenCV DNN**: A pre-trained deep neural network for body keypoint detection.  
  - **MoveNet**: A lightweight TensorFlow Lite model for real-time pose estimation.  

- 🎨 **Image Processing & Annotation**:  
  Leverages OpenCV for image processing, keypoint annotation, and visualization of results with customizable connections and points.  

- 🖥️ **Graphical Interface**:  
  Built with PyQt5, providing a clean, intuitive, and user-friendly GUI for seamless interaction and navigation.  

## 📝 Requirements

To run this project, ensure you have the following installed:

- **Python 3.9+**  
- **OpenCV**: For image processing and video capture.  
- **TensorFlow**: For MoveNet model support.  
- **PyQt5**: For the graphical user interface.  
- **NumPy**: For numerical operations and array handling.  

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## 📂 **Project Structure**
```bash
assets/                            # Folder for static assets (e.g., test images and videos)
├── 1.jpeg                         # Test image for pose estimation
├── test_video_1_from_mixkit.mp4   # Test video 1
├── test_video_2_from_mixkit.mp4   # Test video 2
├── test_video_3_from_mixkit.mp4   # Test video 3
└── val2017/                       # Keypoints annotations for validation COCO Keypoints dataset
    └── person_keypoints_val2017.json  
models/                            # Folder containing pre-trained models
├── graph_opt.pb                                              # OpenCV DNN model
├── movenet-multiphose-lightning-tflite-float16_1.tflite      # Multi-pose MoveNet model
├── movenet-singlepose-lightning_3.tflite                     # Single-pose MoveNet model
└── movenet-singlepose-lightning.pb                           # MoveNet model in PB format
scripts/                           # Folder for evaluation and training scripts
├── eval.ipynb                     # Notebook for model evaluation
├── test.ipynb                     # Notebook for testing models
├── train.ipynb                    # Notebook for model training
└── webcam_OpenPose.py             # Webcam implementation for OpenPose
src/                               # Source code for the application
├── __pycache__/                   
├── BodyPoseEstimationTab.py       # Body pose complete script imported to main app
├── ImageTab.py                    # Image processing subtab logic
├── StandAlonTab.py                
├── VideoTab.py                    # Video processing subtab logic
└── WebcamTab.py                   # Webcam processing subtab logic
main.py                            # Entry point for the application
README.md                          # Project documentation
requirements.txt                   # File listing project dependencies
```