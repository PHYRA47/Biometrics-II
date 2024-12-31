import cv2

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QComboBox, QTabWidget
)

from src.EmotionRecognizer import EmotionRecognizer

class ImageUploadTab(QWidget):
    """Tab for uploading and processing images"""
    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Model selector
        model_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(["MobileNetV2", "Sequential CNN"])
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
        self.recognize_button = QPushButton("Recognize Emotions")
        self.recognize_button.setEnabled(False)
        
        self.upload_button.clicked.connect(self.upload_image)
        self.recognize_button.clicked.connect(self.process_image)
        
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.recognize_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", 
                                                 "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.display_image(self.image)
            self.recognize_button.setEnabled(True)

    def process_image(self):
        if hasattr(self, 'image'):
            self.recognizer.load_model(self.model_selector.currentText())
            processed_image = self.recognizer.process_frame(self.image.copy())
            self.display_image(processed_image)

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale the image to fit the label while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

class WebcamTab(QWidget):
    """Tab for real-time webcam emotion recognition"""
    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        self.init_ui()
        
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.fer_active = False

    def init_ui(self):
        layout = QVBoxLayout()

        # Model selector
        model_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(["MobileNetV2", "Sequential CNN"])
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
        self.fer_button = QPushButton("Start FER")
        self.fer_button.setEnabled(False)
        
        self.camera_button.clicked.connect(self.toggle_camera)
        self.fer_button.clicked.connect(self.toggle_fer)
        
        button_layout.addWidget(self.camera_button)
        button_layout.addWidget(self.fer_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            if self.camera:
                self.camera.release()
                self.camera = None
            self.camera_button.setText("Start Camera")
            self.fer_button.setEnabled(False)
            self.fer_button.setText("Start FER")
            self.fer_active = False
            self.camera_feed.clear()
            self.camera_feed.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        else:
            self.camera = cv2.VideoCapture(0)
            self.timer.start(30)  # 30ms refresh rate
            self.camera_button.setText("Stop Camera")
            self.fer_button.setEnabled(True)

    def toggle_fer(self):
        if self.fer_active:
            self.fer_active = False
            self.fer_button.setText("Start FER")
        else:
            self.recognizer.load_model(self.model_selector.currentText())
            self.fer_active = True
            self.fer_button.setText("Stop FER")

    def update_frame(self):
        if self.camera is None:
            return
            
        ret, frame = self.camera.read()
        if ret:
            if self.fer_active:
                frame = self.recognizer.process_frame(frame)
            
            # Convert frame to QPixmap and display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_feed.setPixmap(QPixmap.fromImage(qt_image))


class FERTab(QWidget):
    """Face Emotion Recognition Tab with sub-tabs"""
    def __init__(self):
        super().__init__()
        self.recognizer = EmotionRecognizer()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create sub-tabs
        self.sub_tabs = QTabWidget()
        self.sub_tabs.addTab(ImageUploadTab(self.recognizer), "Upload")
        self.sub_tabs.addTab(WebcamTab(self.recognizer), "Webcam")
        
        layout.addWidget(self.sub_tabs)
        self.setLayout(layout)