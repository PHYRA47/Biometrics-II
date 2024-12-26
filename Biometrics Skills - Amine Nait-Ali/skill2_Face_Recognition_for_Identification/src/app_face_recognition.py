import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QTabWidget, 
                            QFileDialog, QLineEdit, QTableWidget, QTableWidgetItem,
                            QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import face_recognition
import mysql.connector
from mysql.connector import Error

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize camera variables
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_identifying = False
        
        # Database configuration
        self.db_config = {
            "host": "localhost",
            "user": "my_user",
            "password": "my_password",
            "database": "images_db",
            "port": 3306
        }
        
        # Set up the main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Create tabs
        self.image_tab = QWidget()
        self.webcam_tab = QWidget()
        self.enrollment_tab = QWidget()
        tabs.addTab(self.image_tab, "Image Upload")
        tabs.addTab(self.webcam_tab, "Webcam")
        tabs.addTab(self.enrollment_tab, "Enrollment")
        
        self.setup_image_tab()
        self.setup_webcam_tab()
        self.setup_enrollment_tab()
    
    def setup_image_tab(self):
        layout = QVBoxLayout(self.image_tab)
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        layout.addWidget(self.image_label)
        
        # Buttons container
        button_layout = QHBoxLayout()
        
        # Upload button
        upload_btn = QPushButton("Upload Image")
        upload_btn.clicked.connect(self.upload_image)
        button_layout.addWidget(upload_btn)
        
        # Identify button
        identify_btn = QPushButton("Identify Face")
        identify_btn.clicked.connect(self.identify_face)
        button_layout.addWidget(identify_btn)
        
        layout.addLayout(button_layout)
        
        # Results label
        self.results_label = QLabel("Results will appear here")
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setFixedHeight(30) 
        layout.addWidget(self.results_label)

        # Set minimum size for the window
        self.setMinimumSize(800, 650)
    
    def setup_webcam_tab(self):
        layout = QVBoxLayout(self.webcam_tab)
        
        # Webcam display area
        self.webcam_label = QLabel()
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setMinimumSize(640, 480)
        self.webcam_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        layout.addWidget(self.webcam_label)
        
        # Button container
        button_layout = QHBoxLayout()
        
        # Start/Stop camera button
        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.clicked.connect(self.toggle_camera)
        button_layout.addWidget(self.camera_btn)
        
        # Start/Stop identification button
        self.identify_btn = QPushButton("Start Identifying")
        self.identify_btn.clicked.connect(self.toggle_identification)
        button_layout.addWidget(self.identify_btn)
        
        layout.addLayout(button_layout)
        
        # Results label for webcam
        self.webcam_results_label = QLabel("Camera status: Stopped")
        self.webcam_results_label.setAlignment(Qt.AlignCenter)
        self.webcam_results_label.setFixedHeight(30) 
        layout.addWidget(self.webcam_results_label)

        # Set minimum size for the window
        self.setMinimumSize(800, 650)
    
    def setup_enrollment_tab(self):
        layout = QVBoxLayout(self.enrollment_tab)
        
        # Image display area
        self.enrollment_image_label = QLabel()
        self.enrollment_image_label.setAlignment(Qt.AlignCenter)
        self.enrollment_image_label.setMinimumSize(400, 300)
        self.enrollment_image_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ddd; }")
        layout.addWidget(self.enrollment_image_label)
        
        # Name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_input = QLineEdit()
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        upload_btn = QPushButton("Upload Image")
        upload_btn.clicked.connect(self.upload_enrollment_image)
        enroll_btn = QPushButton("Enroll Face")
        enroll_btn.clicked.connect(self.enroll_face)
        button_layout.addWidget(upload_btn)
        button_layout.addWidget(enroll_btn)
        layout.addLayout(button_layout)
        
        # Enrolled faces table
        self.enrolled_faces_table = QTableWidget()
        self.enrolled_faces_table.setColumnCount(3)
        self.enrolled_faces_table.setHorizontalHeaderLabels(["ID", "Name", "Action"])
        layout.addWidget(self.enrolled_faces_table)
        
        # Load enrolled faces
        self.load_enrolled_faces()

        # Set minimum size for the window
        self.setMinimumSize(800, 650)
    
    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_name:
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                640, 480,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.current_image_path = file_name
    
    def identify_face(self):
        if not hasattr(self, 'current_image_path'):
            QMessageBox.warning(self, "Warning", "Please upload an image first.")
            return
        
        self.results_label.setText("Identifying faces...")
        QApplication.processEvents()
        
        try:
            # Load known face encodings
            known_face_encodings, known_face_names = self.load_known_faces_from_db()
            
            # Load and process the image
            image = cv2.cvtColor(cv2.imread(self.current_image_path), cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            # Process each face
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                
                # Calculate scale factor based on image width
                scale_factor = image.shape[1] / 640.0
                thickness = int(1 * scale_factor)
                font_scale = 0.5 * scale_factor

                # Draw rectangle and name
                cv2.rectangle(image, (left, top), (right, bottom), (0, 85, 255), thickness)
                cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 85, 255), cv2.FILLED)
                cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness)
            # Convert to Qt format and display
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            qt_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            self.results_label.setText("Identification complete")
            
        except Exception as e:
            self.results_label.setText(f"Error during identification: {str(e)}")
    
    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.camera_btn.setText("Start Camera")
            self.webcam_label.clear()
            self.webcam_results_label.setText("Camera status: Stopped")
            self.is_identifying = False
            self.identify_btn.setText("Start Identifying")
        else:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.timer.start(30)
                self.camera_btn.setText("Stop Camera")
                self.webcam_results_label.setText("Camera status: Running")
    
    def toggle_identification(self):
        if not self.camera or not self.camera.isOpened():
            QMessageBox.warning(self, "Warning", "Please start the camera first.")
            return
        
        self.is_identifying = not self.is_identifying
        self.identify_btn.setText("Stop Identifying" if self.is_identifying else "Start Identifying")
        self.webcam_results_label.setText(
            "Status: Identifying faces..." if self.is_identifying else "Camera status: Running"
        )
    
    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            if self.is_identifying:
                # Load known faces
                known_face_encodings, known_face_names = self.load_known_faces_from_db()
                
                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # Process each face
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                    
                    # Draw rectangle and name
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 85, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 85, 255), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Convert to Qt format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                640, 480,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.webcam_label.setPixmap(scaled_pixmap)
    
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
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
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
            
            self.load_enrolled_faces()
            QMessageBox.information(self, "Success", "Face enrolled successfully!")
            
        except Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {str(e)}")
    
    def load_known_faces_from_db(self):
        known_face_encodings = []
        known_face_names = []
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("SELECT image_name, image_column FROM images_store")
            results = cursor.fetchall()
            
            for name, image_data in results:
                # Convert image data to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Get face encoding
                face_encoding = face_recognition.face_encodings(img_rgb)[0]
                
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
            
            cursor.close()
            conn.close()
        
        except Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {str(e)}")
        
        return known_face_encodings, known_face_names

    def load_enrolled_faces(self):
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, image_name FROM images_store")
            results = cursor.fetchall()
            
            self.enrolled_faces_table.setRowCount(0)
            
            for row_number, (id, name) in enumerate(results):
                self.enrolled_faces_table.insertRow(row_number)
                self.enrolled_faces_table.setItem(row_number, 0, QTableWidgetItem(str(id)))
                self.enrolled_faces_table.setItem(row_number, 1, QTableWidgetItem(name))
                
                # Add delete button
                delete_btn = QPushButton("Delete")
                delete_btn.clicked.connect(lambda _, id=id: self.delete_enrolled_face(id))
                self.enrolled_faces_table.setCellWidget(row_number, 2, delete_btn)
            
            cursor.close()
            conn.close()
        
        except Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {str(e)}")

    def delete_enrolled_face(self, face_id):
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM images_store WHERE id = %s", (face_id,))
            conn.commit()
            
            cursor.close()
            conn.close()
            
            self.load_enrolled_faces()
            QMessageBox.information(self, "Success", "Face deleted successfully!")
        
        except Error as e:
            QMessageBox.critical(self, "Error", f"Database error: {str(e)}")

# def main():
#     app = QApplication(sys.argv)
#     app.setStyle('Fusion')  # Modern style
#     window = FaceRecognitionApp()
#     window.show()
#     sys.exit(app.exec())

# if __name__ == '__main__':
#     main()
