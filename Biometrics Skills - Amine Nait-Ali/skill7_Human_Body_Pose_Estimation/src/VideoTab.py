import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

class PoseEstimationVideoTab(QWidget):
    """Tab for processing video files"""
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator
        self.init_ui()
        
        self.video = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

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
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(800, 600)  # Set a fixed size for the video label
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)  # Allow the video label to expand
        
        # Buttons
        button_layout = QHBoxLayout()
        self.upload_button = QPushButton("Upload Video")
        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        
        self.upload_button.clicked.connect(self.upload_video)
        self.play_button.clicked.connect(self.toggle_video)
        
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.play_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def upload_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "", 
                                                 "Video Files (*.mp4 *.avi)")
        if file_name:
            self.video = cv2.VideoCapture(file_name)
            self.play_button.setEnabled(True)
            self.play_button.setText("Play")
            self.timer.stop()

    def toggle_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(30)
            self.play_button.setText("Pause")

    def update_frame(self):
        if self.video is None:
            return
            
        ret, frame = self.video.read()
        if ret:
            processed_frame = self.estimator.process_frame(
                frame,
                self.model_selector.currentText(),
                self.show_points.isChecked(),
                self.show_edges.isChecked()
            )
            
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale the frame to fit the fixed size of the video label
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
        else:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
