import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QComboBox, QTabWidget, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from functions import preprocess_to_lfw, get_image_embeddings
import os


# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Get the directory of the current script
current_script_dir = os.path.dirname(current_script_path)


class FaceRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set the initial size of the window
        self.resize(400, 300)  
        # self.setMinimumSize(800, 600)

        # Initialize model variables
        self.current_model = None
        self.model_paths = {
            'VGG19': f'{current_script_dir}''/model_vgg19/vgg19.keras',
            'Xception': f'{current_script_dir}''/model_xception/xception.keras',
            'ResNet': f'{current_script_dir}''/model_resnet/resnet.keras'
            # Add other models as needed
        }
        
        self.threshold = 0.30  # Similarity threshold
        self.setup_ui()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # Create tabs
        image_tab = QWidget()
        webcam_tab = QWidget()
        tab_widget.addTab(image_tab, "Image Comparison")
        tab_widget.addTab(webcam_tab, "Webcam Comparison")
        
        self.setup_image_tab(image_tab)
        self.setup_webcam_tab(webcam_tab)
    
    def setup_image_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.model_paths.keys())
        self.model_combo.currentTextChanged.connect(self.load_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Images layout
        images_layout = QHBoxLayout()
        
        # First image
        first_image_layout = QVBoxLayout()
        self.image1_label = QLabel()
        self.image1_label.setFixedSize(400, 400)
        self.image1_label.setAlignment(Qt.AlignCenter)
        upload1_btn = QPushButton("Upload First Image")
        upload1_btn.clicked.connect(lambda: self.upload_image(1))
        first_image_layout.addWidget(self.image1_label)
        first_image_layout.addWidget(upload1_btn)
        images_layout.addLayout(first_image_layout)
        
        # Second image
        second_image_layout = QVBoxLayout()
        self.image2_label = QLabel()
        self.image2_label.setFixedSize(400, 400)
        self.image2_label.setAlignment(Qt.AlignCenter)
        upload2_btn = QPushButton("Upload Second Image")
        upload2_btn.clicked.connect(lambda: self.upload_image(2))
        second_image_layout.addWidget(self.image2_label)
        second_image_layout.addWidget(upload2_btn)
        images_layout.addLayout(second_image_layout)
        
        layout.addLayout(images_layout)
        
        # Verify button
        verify_btn = QPushButton("Verify Faces")
        verify_btn.clicked.connect(self.verify_images)
        layout.addWidget(verify_btn)
        
        # Results text box
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFixedHeight(100)
        layout.addWidget(self.results_text)
        
    def setup_webcam_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        self.webcam_model_combo = QComboBox()
        self.webcam_model_combo.addItems(self.model_paths.keys())
        self.webcam_model_combo.currentTextChanged.connect(self.load_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.webcam_model_combo)
        layout.addLayout(model_layout)
        
        # Images layout
        images_layout = QHBoxLayout()
        
        # Reference image
        ref_image_layout = QVBoxLayout()
        self.ref_image_label = QLabel()
        self.ref_image_label.setFixedSize(400, 400)
        self.ref_image_label.setAlignment(Qt.AlignCenter)
        upload_ref_btn = QPushButton("Upload Reference Image")
        upload_ref_btn.clicked.connect(self.upload_reference_image)
        ref_image_layout.addWidget(self.ref_image_label)
        ref_image_layout.addWidget(upload_ref_btn)
        images_layout.addLayout(ref_image_layout)
        
        # Webcam feed
        webcam_layout = QVBoxLayout()
        self.webcam_label = QLabel()
        self.webcam_label.setFixedSize(400, 400)
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_btn = QPushButton("Start Webcam")
        self.webcam_btn.clicked.connect(self.toggle_webcam)
        webcam_layout.addWidget(self.webcam_label)
        webcam_layout.addWidget(self.webcam_btn)
        images_layout.addLayout(webcam_layout)
        
        layout.addLayout(images_layout)
        
        # Verify button
        verify_webcam_btn = QPushButton("Verify Face")
        verify_webcam_btn.clicked.connect(self.verify_webcam)
        layout.addWidget(verify_webcam_btn)
        
        # Results text box
        self.webcam_results_text = QTextEdit()
        self.webcam_results_text.setReadOnly(True)
        self.webcam_results_text.setFixedHeight(100)
        layout.addWidget(self.webcam_results_text)
        
        # Initialize webcam variables
        self.webcam = None
        self.webcam_timer = QTimer()
        self.webcam_timer.timeout.connect(self.update_webcam)
    
    def load_model(self, model_name):
        if model_name in self.model_paths:
            self.current_model = tf.keras.models.load_model(self.model_paths[model_name])
            
            # Clear cached embeddings when the model is changed
            self.cached_image1_path = None
            self.cached_image2_path = None
            self.cached_encoding1 = None
            self.cached_encoding2 = None
        
    def upload_image(self, image_num):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_name:
            # Load the image
            image = cv2.imread(file_name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Load OpenCV's pre-trained face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Sort faces by size (area) and get the largest one
                x, y, w, h = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)[0]
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Convert the image to QPixmap for display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
        
        # Update the appropriate label and path
        if image_num == 1:
            self.image1_path = file_name
            self.image1_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        else:
            self.image2_path = file_name
            self.image2_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
    
    def verify_images(self):
        if hasattr(self, 'image1_path') and hasattr(self, 'image2_path'):
            if self.current_model is None:
                self.load_model(self.model_combo.currentText())
            
            # Load and process images
            image1 = cv2.imread(self.image1_path)
            image2 = cv2.imread(self.image2_path)
            
            # Check if embeddings are already cached
            if not hasattr(self, 'cached_image1_path') or self.cached_image1_path != self.image1_path:
                try:
                    self.cached_encoding1 = get_image_embeddings(self.current_model, image1, False)
                    self.cached_image1_path = self.image1_path
                except ValueError as e:
                    if str(e) == "No faces detected in the image.":
                        self.results_text.setText("No face detected in the first image.")
                        return
                    else:
                        self.results_text.setText("An error occurred during image verification.")
                        return
            
            if not hasattr(self, 'cached_image2_path') or self.cached_image2_path != self.image2_path:
                try:
                    self.cached_encoding2 = get_image_embeddings(self.current_model, image2, False)
                    self.cached_image2_path = self.image2_path
                except ValueError as e:
                    if str(e) == "No faces detected in the image.":
                        self.results_text.setText("No face detected in the second image.")
                        return
                    else:
                        self.results_text.setText("An error occurred during image verification.")
                        return
            
            # Calculate metrics
            distance = tf.norm(self.cached_encoding1 - self.cached_encoding2).numpy()
            cosine_similarity = tf.tensordot(self.cached_encoding1, self.cached_encoding2, axes=1).numpy() / (
                tf.norm(self.cached_encoding1).numpy() * tf.norm(self.cached_encoding2).numpy()
            )
            
            # Update results
            results = f"{'Distance:':<20}{distance:.4f}\n"
            results += f"{'Cosine Similarity:':<20}{cosine_similarity:.4f}\n"
            results += f"{'Match:':<20}{'Yes' if cosine_similarity > self.threshold else 'No'}"
            self.results_text.setText(results)

    def upload_reference_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
                    self, "Open Reference Image", "", "Image Files (*.png *.jpg *.jpeg)"
                )
        if file_name:
            # Load the image
            image = cv2.imread(file_name)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Load OpenCV's pre-trained face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Sort faces by size (area) and get the largest one
                x, y, w, h = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)[0]
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Convert the image to QPixmap for display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            self.ref_image_path = file_name
            self.ref_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def toggle_webcam(self):
        if self.webcam is None:
            self.webcam = cv2.VideoCapture(0)
            self.webcam_btn.setText("Stop Webcam")
            self.webcam_timer.start(30)
        else:
            self.webcam_timer.stop()
            self.webcam.release()
            self.webcam = None
            self.webcam_btn.setText("Start Webcam")
            self.webcam_label.clear()
    
    def update_webcam(self):
        ret, frame = self.webcam.read()
        if ret:
            self.current_frame = frame.copy()  # Copy for processing purposes
            display_frame = frame.copy()  # Copy for display purposes
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            
            # Load OpenCV's pre-trained face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Sort faces by size (area) and get the largest one
                x, y, w, h = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)[0]
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Convert the frame to QPixmap for display
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            height, width, channel = display_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.webcam_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def verify_webcam(self):
        if hasattr(self, 'ref_image_path') and hasattr(self, 'current_frame'):
            if self.current_model is None:
                self.load_model(self.webcam_model_combo.currentText())
            
            # Load and process reference image
            ref_image = cv2.imread(self.ref_image_path)
            
            try:
                # Get embeddings
                ref_encoding = get_image_embeddings(self.current_model, ref_image, False)
                webcam_encoding = get_image_embeddings(self.current_model, self.current_frame, False)
                
                # Calculate metrics
                distance = tf.norm(ref_encoding - webcam_encoding).numpy()
                cosine_similarity = tf.tensordot(ref_encoding, webcam_encoding, axes=1).numpy() / (
                    tf.norm(ref_encoding).numpy() * tf.norm(webcam_encoding).numpy()
                )
                
                # Update results
                results = f"{'Distance:':<20}{distance:.4f}\n"
                results += f"{'Cosine Similarity:':<20}{cosine_similarity:.4f}\n"
                results += f"{'Match:':<20}{'Yes' if cosine_similarity > self.threshold else 'No'}"
                self.webcam_results_text.setText(results)
            except ValueError as e:
                if str(e) == "No faces detected in the image.":
                    self.webcam_results_text.setText("No face detected in the reference or webcam image.")
                else:
                    self.webcam_results_text.setText("An error occurred during image verification.")

    def closeEvent(self, event):
        if self.webcam is not None:
            self.webcam.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionGUI()
    window.show()
    sys.exit(app.exec_())