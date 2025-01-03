import cv2
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget)
from PyQt5.QtCore import QTimer

from PIL import Image
from tensorflow.keras.models import load_model # type: ignore

class GenderEstimator:
    """Handles gender estimation logic for different models"""
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.gender_dict = {0: 'Male', 1: 'Female'}
        self.current_model = None
        self.model_type = None

    def load_model(self, model_type):
        """Load the selected model"""
        self.model_type = model_type
        if model_type == "Custom CNN":
            self.current_model = load_model('models/age_gender_estimation/custom_CNN/custom_CNN.keras')
        else:  # ResNet50
            self.current_model = load_model('models/age_gender_estimation/custom_ResNet50/custom_ResNet50.keras')

    def process_frame(self, frame):
        """Process a frame and return the annotated result"""
        if self.current_model is None:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            
            # Process face for prediction
            processed_face = self.preprocess_face(face_roi)
            
            # Make prediction
            predictions = self.current_model.predict(processed_face, verbose=0)
            pred_gender = self.gender_dict[round(predictions[0][0][0])]  # Using only gender prediction
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 85, 0), 1)
            
            # Draw label background with transparency
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y + h + 1), (x + w, y + h + 25), (255, 85, 0), cv2.FILLED)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            
            # Add text
            cv2.putText(frame, f"Gender: {pred_gender}", (x + 6, y + h + 20), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def preprocess_face(self, face_roi, target_size=(128, 128)):
        """Preprocess face ROI for model input"""
        rgb_frame = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        resized_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(resized_image) / 255.0
        return np.expand_dims(img_array, axis=0)

class GenderEstimationImageTab(QWidget):
    """Tab for uploading and processing images for gender estimation"""
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Model selector
        model_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Custom CNN", "ResNet50"])
        model_layout.addWidget(QLabel("Select Model:"))
        model_layout.addWidget(self.model_selector)
        model_layout.addStretch()
        layout.addLayout(model_layout)

        # Image display
        self.image_label = QLabel()
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # Control buttons
        button_layout = QHBoxLayout()
        self.upload_button = QPushButton("Upload Image")
        self.estimate_button = QPushButton("Estimate Gender")
        self.estimate_button.setEnabled(False)
        
        self.upload_button.clicked.connect(self.upload_image)
        self.estimate_button.clicked.connect(self.process_image)
        
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.estimate_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", 
                                                 "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.display_image(self.image)
            self.estimate_button.setEnabled(True)

    def process_image(self):
        if hasattr(self, 'image'):
            self.estimator.load_model(self.model_selector.currentText())
            processed_image = self.estimator.process_frame(self.image.copy())
            self.display_image(processed_image)

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

class GenderEstimationWebcamTab(QWidget):
    """Tab for real-time webcam gender estimation"""
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator
        self.init_ui()
        
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.estimation_active = False

    def init_ui(self):
        layout = QVBoxLayout()

        # Model selector
        model_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Custom CNN", "ResNet50"])
        model_layout.addWidget(QLabel("Select Model:"))
        model_layout.addWidget(self.model_selector)
        model_layout.addStretch()
        layout.addLayout(model_layout)

        # Camera feed
        self.camera_feed = QLabel()
        self.camera_feed.setAlignment(Qt.AlignCenter)
        self.camera_feed.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        layout.addWidget(self.camera_feed)

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
            self.camera_feed.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        else:
            self.camera = cv2.VideoCapture(0)
            self.timer.start(30)
            self.camera_button.setText("Stop Camera")
            self.estimation_button.setEnabled(True)

    def toggle_estimation(self):
        if self.estimation_active:
            self.estimation_active = False
            self.estimation_button.setText("Start Estimation")
        else:
            self.estimator.load_model(self.model_selector.currentText())
            self.estimation_active = True
            self.estimation_button.setText("Stop Estimation")

    def update_frame(self):
        if self.camera is None:
            return
            
        ret, frame = self.camera.read()
        if ret:
            if self.estimation_active:
                frame = self.estimator.process_frame(frame)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_feed.setPixmap(QPixmap.fromImage(qt_image))

class GenderEstimationTab(QWidget):
    """Gender Estimation Tab with sub-tabs"""
    def __init__(self):
        super().__init__()
        self.estimator = GenderEstimator()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create sub-tabs
        self.sub_tabs = QTabWidget()
        self.sub_tabs.addTab(GenderEstimationImageTab(self.estimator), "Upload")
        self.sub_tabs.addTab(GenderEstimationWebcamTab(self.estimator), "Webcam")
        
        layout.addWidget(self.sub_tabs)
        self.setLayout(layout)
