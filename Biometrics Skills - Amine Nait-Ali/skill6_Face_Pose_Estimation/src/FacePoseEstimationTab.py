import cv2
import numpy as np
import mediapipe as mp
import pickle
from math import sin, cos
from PIL import Image

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QComboBox, QCheckBox, QFileDialog, QHBoxLayout, 
                           QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget)

class FacePoseEstimator:
    """Handles face pose estimation logic"""
    def __init__(self):
        self.face_module = mp.solutions.face_mesh
        self.current_model = None
        self.model_type = None
        self.svr_models = None
        self.dense_model = None
        
    def load_models(self):
        """Load both SVR and Dense models"""
        with open('models/model_1_svm_best.pkl', 'rb') as file:
            self.svr_models = pickle.load(file)
        with open('models/model_5.pkl', 'rb') as file:
            self.dense_model = pickle.load(file)
    
    def detect_landmarks(self, img):
        """Detects facial landmarks using MediaPipe FaceMesh"""
        faces_data = []
        with self.face_module.FaceMesh(static_image_mode=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    X = []
                    y = []
                    shape = img.shape
                    for landmark in face_landmarks.landmark:
                        relative_x = int(landmark.x * shape[1])
                        relative_y = int(landmark.y * shape[0])
                        X.append(relative_x)
                        y.append(relative_y)
                    
                    X = np.array([X])
                    y = np.array([y])
                    
                    # Normalize features
                    Nose_centered_X = X - X[:, 1].reshape(-1, 1)
                    Nose_centered_y = y - y[:, 1].reshape(-1, 1)
                    
                    X_171, X_10 = X[:, 171], X[:, 10]
                    y_171, y_10 = y[:, 171], y[:, 10]
                    distance = np.linalg.norm(np.array((X_10, y_10)) - np.array((X_171, y_171)), axis=0).reshape(-1, 1)
                    
                    Norm_X = Nose_centered_X / distance
                    Norm_Y = Nose_centered_y / distance
                    features = np.hstack([Norm_X, Norm_Y])
                    
                    faces_data.append({
                        'features': features,
                        'nose_x': X[:, 1].item(),
                        'nose_y': y[:, 1].item(),
                        'landmarks': face_landmarks
                    })
        
        return faces_data
    
    def draw_axis(self, img, pitch, yaw, roll, tdx, tdy, size=100):
        """Draws 3D axis on the image based on pose angles"""
        yaw = -yaw
        pitch = pitch.item()
        yaw = yaw.item()
        roll = roll.item()
        
        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy
        
        cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)
        
        return img
    
    def process_frame(self, frame, model_type="SVR", show_landmarks=True, show_pose=True):
        """Process a frame and return the annotated result"""
        if not self.svr_models or not self.dense_model:
            self.load_models()
            
        processed_frame = frame.copy()
        faces_data = self.detect_landmarks(processed_frame)
        
        for face_data in faces_data:
            features = face_data['features']
            
            # Get predictions based on model type
            if model_type == "SVR":
                pitch = self.svr_models[0].predict(features)
                yaw = self.svr_models[1].predict(features)
                roll = self.svr_models[2].predict(features)
            else:  # DenseNN
                predictions = self.dense_model.predict(features)
                pitch, yaw, roll = predictions[:, 0], predictions[:, 1], predictions[:, 2]
            
            # Draw pose axes if enabled
            if show_pose:
                processed_frame = self.draw_axis(processed_frame, pitch, yaw, roll, 
                                              face_data['nose_x'], face_data['nose_y'])
            
            # Draw landmarks if enabled
            if show_landmarks:
                for landmark in face_data['landmarks'].landmark:
                    x = int(landmark.x * processed_frame.shape[1])
                    y = int(landmark.y * processed_frame.shape[0])
                    cv2.circle(processed_frame, (x, y), radius=1, 
                             color=(200, 200, 200, 50), thickness=1)
        
        return processed_frame

class PoseEstimationImageTab(QWidget):
    """Tab for uploading and processing images for pose estimation"""
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator
        self.current_image = None
        self.processed = False
        self.max_display_width = 800
        self.max_display_height = 600
        self.init_ui()
          
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Model selector
        self.model_selector = QComboBox()
        self.model_selector.addItems(["SVR", "DenseNN"])
        controls_layout.addWidget(QLabel("Select Model:"))
        controls_layout.addWidget(self.model_selector)
        
        # Checkboxes for toggles
        self.show_landmarks = QCheckBox("Show Landmarks")
        self.show_pose = QCheckBox("Show Pose")
        self.show_landmarks.setChecked(True)
        self.show_pose.setChecked(True)
        
        controls_layout.addWidget(self.show_landmarks)
        controls_layout.addWidget(self.show_pose)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Update checkbox connections
        self.show_landmarks.stateChanged.connect(self.on_toggle_change)
        self.show_pose.stateChanged.connect(self.on_toggle_change)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.upload_button = QPushButton("Upload Image")
        self.process_button = QPushButton("Process Image")
        self.process_button.setEnabled(False)
        
        self.upload_button.clicked.connect(self.upload_image)
        self.process_button.clicked.connect(self.process_image)
        
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.process_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", 
                                                 "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.current_image = cv2.imread(file_name)
            self.display_image(self.current_image)
            self.process_button.setEnabled(True)
            self.processed = False

    def on_toggle_change(self):
        if self.processed and self.current_image is not None:
            self.process_image()
    
    def process_image(self):
        if self.current_image is not None:
            processed_image = self.estimator.process_frame(
                self.current_image.copy(),
                model_type=self.model_selector.currentText(),
                show_landmarks=self.show_landmarks.isChecked(),
                show_pose=self.show_pose.isChecked()
            )
            self.display_image(processed_image)
            self.processed = True
    
    def display_image(self, image):
        if image is None:
            return
            
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        new_width = min(width, self.max_display_width)
        new_height = int(new_width / aspect_ratio)
        
        if new_height > self.max_display_height:
            new_height = self.max_display_height
            new_width = int(new_height * aspect_ratio)
            
        resized = cv2.resize(image, (new_width, new_height))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

class PoseEstimationWebcamTab(QWidget):
    """Tab for real-time webcam pose estimation"""
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator
        self.init_ui()
        
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.prev_frame_time = 0
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Model selector
        self.model_selector = QComboBox()
        self.model_selector.addItems(["SVR", "DenseNN"])
        controls_layout.addWidget(QLabel("Select Model:"))
        controls_layout.addWidget(self.model_selector)
        
        # Checkboxes for toggles
        self.show_landmarks = QCheckBox("Show Landmarks")
        self.show_pose = QCheckBox("Show Pose")
        self.show_landmarks.setChecked(True)
        self.show_pose.setChecked(True)
        
        controls_layout.addWidget(self.show_landmarks)
        controls_layout.addWidget(self.show_pose)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Camera feed
        self.camera_feed = QLabel()
        self.camera_feed.setAlignment(Qt.AlignCenter)
        self.camera_feed.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        layout.addWidget(self.camera_feed)
        
        # FPS display
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFixedHeight(20)
        layout.addWidget(self.fps_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.camera_button = QPushButton("Start Camera")
        self.estimation_button = QPushButton("Start Estimation")
        self.estimation_button.setEnabled(False)
        
        self.camera_button.clicked.connect(self.toggle_camera)
        self.estimation_button.clicked.connect(self.toggle_estimation)
        
        button_layout.addWidget(self.camera_button)
        button_layout.addWidget(self.estimation_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.estimation_active = False
    
    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            if self.camera:
                self.camera.release()
                self.camera = None
            self.camera_button.setText("Start Camera")
            self.estimation_button.setEnabled(False)
            self.estimation_button.setText("Start Estimation")
            self.estimation_active = False
            self.camera_feed.clear()
            self.fps_label.setText("FPS: 0")
        else:
            self.camera = cv2.VideoCapture(0)
            self.timer.start(30)
            self.camera_button.setText("Stop Camera")
            self.estimation_button.setEnabled(True)
    
    def toggle_estimation(self):
        self.estimation_active = not self.estimation_active
        self.estimation_button.setText("Stop Estimation" if self.estimation_active else "Start Estimation")
    
    def update_frame(self):
        if self.camera is None:
            return
        
        ret, frame = self.camera.read()
        if ret:
            # Calculate FPS
            new_frame_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (new_frame_time - self.prev_frame_time)
            self.prev_frame_time = new_frame_time
            self.fps_label.setText(f"FPS: {int(fps)}")
            
            if self.estimation_active:
                frame = self.estimator.process_frame(
                    frame,
                    model_type=self.model_selector.currentText(),
                    show_landmarks=self.show_landmarks.isChecked(),
                    show_pose=self.show_pose.isChecked()
                )
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_feed.setPixmap(QPixmap.fromImage(qt_image))

class PoseEstimationVideoTab(QWidget):
    """Tab for processing video files for pose estimation"""
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator
        self.init_ui()
        
        self.video = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.prev_frame_time = 0
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Model selector
        self.model_selector = QComboBox()
        self.model_selector.addItems(["SVR", "DenseNN"])
        controls_layout.addWidget(QLabel("Select Model:"))
        controls_layout.addWidget(self.model_selector)
        
        # Checkboxes for toggles
        self.show_landmarks = QCheckBox("Show Landmarks")
        self.show_pose = QCheckBox("Show Pose")
        self.show_landmarks.setChecked(True)
        self.show_pose.setChecked(True)
        
        controls_layout.addWidget(self.show_landmarks)
        controls_layout.addWidget(self.show_pose)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        # FPS display
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFixedHeight(20)
        layout.addWidget(self.fps_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.upload_button = QPushButton("Upload Video")
        self.process_button = QPushButton("Start Processing")
        self.process_button.setEnabled(False)
        
        self.upload_button.clicked.connect(self.upload_video)
        self.process_button.clicked.connect(self.toggle_processing)
        
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.process_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.processing_active = False
    
    def upload_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "", 
                                                 "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.video = cv2.VideoCapture(file_name)
            self.process_button.setEnabled(True)
            self.update_frame()
    
    def toggle_processing(self):
        self.processing_active = not self.processing_active
        self.process_button.setText("Stop Processing" if self.processing_active else "Start Processing")
        if self.processing_active:
            self.timer.start(30)
        else:
            self.timer.stop()
    
    def update_frame(self):
        if self.video is None:
            return
        
        ret, frame = self.video.read()
        if ret:
            # Calculate FPS
            new_frame_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (new_frame_time - self.prev_frame_time)
            self.prev_frame_time = new_frame_time
            self.fps_label.setText(f"FPS: {int(fps)}")
            
            if self.processing_active:
                frame = self.estimator.process_frame(
                    frame,
                    model_type=self.model_selector.currentText(),
                    show_landmarks=self.show_landmarks.isChecked(),
                    show_pose=self.show_pose.isChecked()
                )
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        else:
            self.timer.stop()
            self.process_button.setEnabled(False)
            self.process_button.setText("Start Processing")
            self.processing_active = False


class FacePoseEstimationTab(QWidget):
    """Face Pose Estimation Tab with sub-tabs"""
    def __init__(self):
        super().__init__()
        self.estimator = FacePoseEstimator()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create sub-tabs
        self.sub_tabs = QTabWidget()
        self.sub_tabs.addTab(PoseEstimationImageTab(self.estimator), "Upload")
        self.sub_tabs.addTab(PoseEstimationWebcamTab(self.estimator), "Webcam")
        # self.sub_tabs.addTab(PoseEstimationVideoTab(self.estimator), "Video")
        
        layout.addWidget(self.sub_tabs)
        self.setLayout(layout)