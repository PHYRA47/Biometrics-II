import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QComboBox, QTabWidget,
                           QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from tensorflow.keras.models import load_model # type: ignore

class EmotionRecognizer:
    """Handles emotion recognition logic for different models"""
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_colors = {
            'Angry': (0, 0, 255, 200),  # Red
            'Disgust': (0, 255, 0, 200),  # Green
            'Fear': (255, 0, 0, 200),  # Blue
            'Happy': (255, 255, 0, 200),  # Cyan
            'Sad': (0, 255, 255),  # Yellow
            'Surprise': (255, 0, 255),  # Magenta
            'Neutral': (255, 255, 255, 200)  # White
        }
        self.current_model = None
        self.model_type = None

    def load_model(self, model_type):
        """Load the selected model"""
        self.model_type = model_type
        if model_type == "MobileNetV2":
            self.current_model = load_model('models/mobilenetv2/mobilenetv2.h5')
        else:  # Sequential CNN
            self.current_model = load_model('models/sequentialCNN/sequentialCNN.h5')

    def process_frame(self, frame):
        """Process a frame and return the annotated result"""
        if self.current_model is None:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            if self.model_type == "MobileNetV2":
                processed_image = cv2.resize(roi_color, (224, 224))
                processed_image = np.expand_dims(processed_image, axis=0)
                processed_image = processed_image / 255.0
            else:  # Sequential CNN
                processed_image = cv2.resize(roi_gray, (48, 48))
                processed_image = processed_image / 255.0
                processed_image = np.expand_dims(processed_image, axis=0)
                processed_image = np.expand_dims(processed_image, axis=-1)

            predictions = self.current_model.predict(processed_image)
            emotion = self.class_names[np.argmax(predictions[0])]
            color = self.emotion_colors.get(emotion, (255, 255, 255))  # Default to white if not found

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, emotion, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame

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

class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Analysis System")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(800, 600)  # Set minimum size for the application window

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(FERTab(), "Face Emotion Recognition")
        
        self.setCentralWidget(self.tabs)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())