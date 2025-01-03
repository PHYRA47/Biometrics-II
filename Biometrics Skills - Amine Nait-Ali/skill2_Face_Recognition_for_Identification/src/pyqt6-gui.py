import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QTabWidget, 
                            QFileDialog, QComboBox, QLineEdit, QMessageBox,
                            QTableWidget, QTableWidgetItem)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
import face_recognition
import mysql.connector
from mysql.connector import Error
import io
from PIL import Image

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 1000, 800)
        
        # Initialize database configuration
        self.db_config = {
            "host": "localhost",
            "user": "my_user",
            "password": "my_password",
            "database": "images_db",
            "port": 3306
        }
        
        # Initialize camera variables
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Set up the main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create tabs
        self.enrollment_tab = QWidget()
        self.image_tab = QWidget()
        self.webcam_tab = QWidget()
        
        self.tabs.addTab(self.enrollment_tab, "Enrollment")
        self.tabs.addTab(self.image_tab, "Image Recognition")
        self.tabs.addTab(self.webcam_tab, "Real-time Recognition")
        
        self.setup_enrollment_tab()
        self.setup_image_tab()
        self.setup_webcam_tab()
    
    def setup_enrollment_tab(self):
        layout = QVBoxLayout(self.enrollment_tab)
        
        # Model selector
        model_layout = QHBoxLayout()
        model_label = QLabel("Select model/backend:")
        self.enrollment_model_selector = QComboBox()
        self.enrollment_model_selector.addItems(["VGGFace Model", "Siamese Network"])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.enrollment_model_selector)
        model_layout.addStretch()
        layout.addLayout(model_layout)
        
        # Image preview and controls
        preview_layout = QHBoxLayout()
        
        # Left side - Image preview
        self.enrollment_image_label = QLabel()
        self.enrollment_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.enrollment_image_label.setMinimumSize(400, 300)
        self.enrollment_image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        preview_layout.addWidget(self.enrollment_image_label)
        
        # Right side - Controls
        controls_layout = QVBoxLayout()
        
        # Name input
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        self.name_input = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        controls_layout.addLayout(name_layout)
        
        # Buttons
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_enrollment_image)
        controls_layout.addWidget(self.upload_btn)
        
        self.enroll_btn = QPushButton("Enroll Face")
        self.enroll_btn.clicked.connect(self.enroll_face)
        controls_layout.addWidget(self.enroll_btn)
        
        preview_layout.addLayout(controls_layout)
        layout.addLayout(preview_layout)
        
        # Database table
        self.enrolled_faces_table = QTableWidget()
        self.enrolled_faces_table.setColumnCount(3)
        self.enrolled_faces_table.setHorizontalHeaderLabels(["ID", "Name", "Actions"])
        layout.addWidget(self.enrolled_faces_table)
        
        # Load enrolled faces
        self.load_enrolled_faces()
    
    def setup_image_tab(self):
        layout = QVBoxLayout(self.image_tab)
        
        # Model selector
        model_layout = QHBoxLayout()
        model_label = QLabel("Select model/backend:")
        self.image_model_selector = QComboBox()
        self.image_model_selector.addItems(["VGGFace Model", "Siamese Network"])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.image_model_selector)
        model_layout.addStretch()
        layout.addLayout(model_layout)
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        layout.addWidget(self.image_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        upload_btn = QPushButton("Upload Image")
        upload_btn.clicked.connect(self.upload_image)
        recognize_btn = QPushButton("Recognize Faces")
        recognize_btn.clicked.connect(self.recognize_faces)
        
        button_layout.addWidget(upload_btn)
        button_layout.addWidget(recognize_btn)
        layout.addLayout(button_layout)
    
    def setup_webcam_tab(self):
        layout = QVBoxLayout(self.webcam_tab)
        
        # Model selector
        model_layout = QHBoxLayout()
        model_label = QLabel("Select model/backend:")
        self.webcam_model_selector = QComboBox()
        self.webcam_model_selector.addItems(["VGGFace Model", "Siamese Network"])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.webcam_model_selector)
        model_layout.addStretch()
        layout.addLayout(model_layout)
        
        # Webcam display
        self.webcam_label = QLabel()
        self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label.setMinimumSize(640, 480)
        self.webcam_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        layout.addWidget(self.webcam_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.clicked.connect(self.toggle_camera)
        controls_layout.addWidget(self.camera_btn)
        layout.addLayout(controls_layout)

    def upload_enrollment_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_name:
            self.current_enrollment_image = file_name
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                400, 300,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.enrollment_image_label.setPixmap(scaled_pixmap)

    def enroll_face(self):
        if not hasattr(self, 'current_enrollment_image') or not self.name_input.text():
            QMessageBox.warning(self, "Warning", "Please provide both image and name.")
            return
            
        try:
            # Load and encode face
            image = face_recognition.load_image_file(self.current_enrollment_image)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                QMessageBox.warning(self, "Warning", "No face detected in the image.")
                return
                
            # Save to database
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Convert image to binary
            with open(self.current_enrollment_image, 'rb') as file:
                binary_data = file.read()
            
            # Insert into database
            cursor.execute("""
                INSERT INTO images_store (image_name, image_column) 
                VALUES (%s, %s)
            """, (self.name_input.text(), binary_data))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.load_enrolled_faces()  # Reload the table
            QMessageBox.information(self, "Success", "Face enrolled successfully!")
            
        except Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {str(e)}")

    def load_enrolled_faces(self):
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, image_name FROM images_store")
            results = cursor.fetchall()
            
            self.enrolled_faces_table.setRowCount(len(results))
            
            for row, (id_, name) in enumerate(results):
                self.enrolled_faces_table.setItem(row, 0, QTableWidgetItem(str(id_)))
                self.enrolled_faces_table.setItem(row, 1, QTableWidgetItem(name))
                
                # Add delete button
                delete_btn = QPushButton("Delete")
                delete_btn.clicked.connect(lambda checked, x=id_: self.delete_face(x))
                self.enrolled_faces_table.setCellWidget(row, 2, delete_btn)
            
            cursor.close()
            conn.close()
            
        except Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {str(e)}")

    def delete_face(self, face_id):
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM images_store WHERE id = %s", (face_id,))
            conn.commit()
            
            cursor.close()
            conn.close()
            
            self.load_enrolled_faces()  # Reload the table
            
        except Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {str(e)}")

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # Convert frame to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Get known faces from database
            known_faces, known_names = self.load_known_faces_from_db()
            
            # Draw boxes and labels for each face
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]
                
                # Draw box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw label
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            # Convert to Qt format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                640, 480,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.webcam_label.setPixmap(scaled_pixmap)

    def load_known_faces_from_db(self):
        known_faces = []
        known_names = []
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("SELECT image_name, image_column FROM images_store")
            results = cursor.fetchall()
            
            for name, image_data in results:
                # Convert image data to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Get face encoding
                face_encodings = face_recognition.face_encodings(rgb_img)
                if face_encodings:
                    known_faces.append(face_encodings[0])
                    known_names.append(name)
            
            cursor.close()
            conn.close()
            
        except Error as e:
            print(f"Error accessing database: {e}")
        
        return known_faces, known_names

    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.camera_btn.setText("Start Camera")
            self.webcam_label.clear()
        else:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.timer.start(30)  # 30ms refresh rate
                self.camera_btn.setText("Stop Camera")

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_name:
            self.current_image_path = file_name  # Store the path for later use
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                640, 480,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def recognize_faces(self):
        if not hasattr(self, 'current_image_path'):
            QMessageBox.warning(self, "Warning", "Please upload an image first.")
            return
            
        try:
            # Load the image
            image = face_recognition.load_image_file(self.current_image_path)
            # Convert to RGB for display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get face locations and encodings
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # Get known faces from database
            known_faces, known_names = self.load_known_faces_from_db()
            
            # Process each face found in the image
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare with known faces
                matches = face_recognition.compare_faces(known_faces, face_encoding)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]
                
                # Draw box around face
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw label with name
                cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            # Convert to Qt format and display
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qt_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                640, 480,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing image: {str(e)}")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()