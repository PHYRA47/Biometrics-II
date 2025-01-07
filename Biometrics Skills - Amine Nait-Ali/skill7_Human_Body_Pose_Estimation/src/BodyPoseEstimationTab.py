import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QComboBox, QCheckBox, QFileDialog, QHBoxLayout, 
                           QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget)

class PoseEstimator:
    """Handles human pose estimation using different models"""
    def __init__(self):
        # OpenCV DNN model
        self.net = cv2.dnn.readNetFromTensorflow('models/graph_opt.pb')
        self.inWidth = 368
        self.inHeight = 368
        self.thr = 0.2
        
        # MoveNet model
        self.interpreter = tf.lite.Interpreter(model_path='models/movenet-singlepose-lightning_3.tflite')
        self.interpreter.allocate_tensors()
        
        # Body parts and connections for OpenCV model
        self.BODY_PARTS = {
            "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
        }
        
        self.POSE_PAIRS = [
            ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
            ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
            ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
            ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
        ]

        # MoveNet connections
        self.EDGES = {
            (0, 1): 'm',
            (0, 2): 'c',
            (1, 3): 'm',
            (2, 4): 'c',
            (0, 5): 'm',
            (0, 6): 'c',
            (5, 7): 'm',
            (7, 9): 'm',
            (6, 8): 'c',
            (8, 10): 'c',
            (5, 6): 'y',
            (5, 11): 'm',
            (6, 12): 'c',
            (11, 12): 'y',
            (11, 13): 'm',
            (13, 15): 'm',
            (12, 14): 'c',
            (14, 16): 'c'
        }

    def process_frame_opencv(self, frame, show_points=True, show_edges=True):
        """Process frame using OpenCV DNN model"""
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        
        self.net.setInput(cv2.dnn.blobFromImage(
            frame, 1.0, (self.inWidth, self.inHeight),
            (127.5, 127.5, 127.5), swapRB=True, crop=False
        ))
        out = self.net.forward()
        out = out[:, :19, :, :]

        points = []
        for i in range(len(self.BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > self.thr else None)

        # Draw edges if enabled
        if show_edges:
            for pair in self.POSE_PAIRS:
                partFrom = pair[0]
                partTo = pair[1]
                idFrom = self.BODY_PARTS[partFrom]
                idTo = self.BODY_PARTS[partTo]
                if points[idFrom] and points[idTo]:
                    cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 2)

        # Draw points if enabled
        if show_points:
            for point in points:
                if point:
                    cv2.circle(frame, point, 3, (0, 0, 255), cv2.FILLED)

        return frame

    def process_frame_movenet(self, frame, show_points=True, show_edges=True):
        """Process frame using MoveNet model"""
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)
        
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        self.interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(output_details[0]['index'])
        
        if show_points:
            self.draw_keypoints(frame, keypoints_with_scores, 0.4)
        
        if show_edges:
            self.draw_connections(frame, keypoints_with_scores, self.EDGES, 0.4)
        
        return frame

    def draw_keypoints(self, frame, keypoints, confidence_threshold):
        """Draw keypoints on the frame"""
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  

        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 3, (0, 0, 255), cv2.FILLED)

    def draw_connections(self, frame, keypoints, edges, confidence_threshold):
        """Draw connections between keypoints on the frame"""
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))  

        for edge, color in edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]

            if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    def process_frame(self, frame, model_type="OpenCV", show_points=True, show_edges=True):
        """Process frame using selected model"""
        if model_type == "OpenCV":
            return self.process_frame_opencv(frame, show_points, show_edges)
        else:  # MoveNet
            return self.process_frame_movenet(frame, show_points, show_edges)

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
        layout.addWidget(self.video_label, stretch=1)  # Allow the video label to expand
        
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
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
        else:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

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

class BodyPoseEstimationTab(QWidget):
    """Face Pose Estimation Tab with sub-tabs"""
    def __init__(self):
        super().__init__()
        self.estimator = PoseEstimator()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create sub-tabs
        self.sub_tabs = QTabWidget()
        self.sub_tabs.addTab(PoseEstimationImageTab(self.estimator), "Upload")
        self.sub_tabs.addTab(PoseEstimationVideoTab(self.estimator), "Video")
        self.sub_tabs.addTab(PoseEstimationWebcamTab(self.estimator), "Webcam")
        
        layout.addWidget(self.sub_tabs)
        self.setLayout(layout)

class MainWindow(QWidget):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.estimator = PoseEstimator()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(PoseEstimationImageTab(self.estimator), "Image")
        self.tabs.addTab(PoseEstimationVideoTab(self.estimator), "Video")
        self.tabs.addTab(PoseEstimationWebcamTab(self.estimator), "Webcam")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        self.setWindowTitle("Pose Estimation Tool")
        self.resize(800, 600)

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())