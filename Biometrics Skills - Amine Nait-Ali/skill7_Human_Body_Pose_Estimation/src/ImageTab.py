import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

class PoseEstimationImageTab(QWidget):
    """Tab for uploading and processing images"""
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Model selector
        self.model_selector = QComboBox()
        self.model_selector.addItems(["OpenCV", "MoveNet"])
        controls_layout.addWidget(QLabel("Select Model:"))
        controls_layout.addWidget(self.model_selector)
        
        # Checkboxes
        self.show_points = QCheckBox("Show Points")
        self.show_edges = QCheckBox("Show Edges")
        self.show_points.setChecked(True)
        self.show_edges.setChecked(True)
        
        # Connect checkbox toggles to reprocess the image
        self.show_points.stateChanged.connect(self.process_image)
        self.show_edges.stateChanged.connect(self.process_image)
        
        controls_layout.addWidget(self.show_points)
        controls_layout.addWidget(self.show_edges)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label, stretch=1)  # Allow the image label to expand
        
        # Buttons
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
        """Upload an image from the file system"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", 
                                                 "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.display_image(self.image)
            self.process_button.setEnabled(True)
            self.process_image()  # Process the image immediately after upload

    def process_image(self):
        """Process the uploaded image based on the selected model and toggles"""
        if hasattr(self, 'image'):
            processed_image = self.estimator.process_frame(
                self.image.copy(),
                self.model_selector.currentText(),
                self.show_points.isChecked(),
                self.show_edges.isChecked()
            )
            self.display_image(processed_image)

    def display_image(self, image):
        """Display the processed image in the QLabel"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
