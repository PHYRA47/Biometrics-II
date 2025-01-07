import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

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
        self.model_selector.addItems(["OpenCV", "MoveNet"])
        controls_layout.addWidget(QLabel("Select Model:"))
        controls_layout.addWidget(self.model_selector)
        
        # Checkboxes
        self.show_points = QCheckBox("Show Points")
        self.show_edges = QCheckBox("Show Edges")
        self.show_points.setChecked(True)
        self.show_edges.setChecked(True)
        
        controls_layout.addWidget(self.show_points)
        controls_layout.addWidget(self.show_edges)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Webcam feed display
        self.webcam_label = QLabel()
        self.webcam_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        self.webcam_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.webcam_label, stretch=1)  # Allow the webcam label to expand
        
        # FPS display
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFixedHeight(20)  # Set fixed height for FPS label
        layout.addWidget(self.fps_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Webcam")
        self.start_button.clicked.connect(self.toggle_webcam)
        
        button_layout.addWidget(self.start_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.is_webcam_running = False

    def toggle_webcam(self):
        """Start or stop the webcam feed"""
        if self.is_webcam_running:
            self.timer.stop()
            if self.camera:
                self.camera.release()
                self.camera = None
            self.start_button.setText("Start Webcam")
            self.is_webcam_running = False
            self.webcam_label.clear()
            self.fps_label.setText("FPS: 0")
        else:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.start_button.setText("Start Webcam")
                self.is_webcam_running = False
                return
            self.timer.start(30)
            self.start_button.setText("Stop Webcam")
            self.is_webcam_running = True

    def update_frame(self):
        """Update the webcam feed with pose estimation"""
        if self.camera is None:
            return
        
        ret, frame = self.camera.read()
        if ret:
            # Calculate FPS
            new_frame_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (new_frame_time - self.prev_frame_time)
            self.prev_frame_time = new_frame_time
            self.fps_label.setText(f"FPS: {int(fps)}")
            
            # Process the frame with the selected model
            processed_frame = self.estimator.process_frame(
                frame,
                self.model_selector.currentText(),
                self.show_points.isChecked(),
                self.show_edges.isChecked()
            )
            
            # Display the processed frame
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.webcam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.webcam_label.setPixmap(scaled_pixmap)
