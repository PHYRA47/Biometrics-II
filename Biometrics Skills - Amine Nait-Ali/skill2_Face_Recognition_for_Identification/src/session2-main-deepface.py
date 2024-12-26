import cv2
import numpy as np
import os
import io

import face_recognition

from PIL import Image
from deepface import DeepFace


from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from GUI_3 import Ui_MainWindow

import mysql.connector
from mysql.connector import Error

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Tab 1
        self.t1_org_img_path = 'images/t1/1.jpg'
        self.t1_up_img_path = None

        self.ui.t1_org_img.setPixmap(QPixmap(self.t1_org_img_path))
        self.ui.t1_pb_load.clicked.connect(self.load_image_tab1)
        self.ui.t1_pb_check.clicked.connect(self.display_message)

        # Tab 2
        self.t2_up_img_path = None
        self.ui.t2_pb_load.clicked.connect(self.load_image_tab2)
        self.ui.t2_pb_match.clicked.connect(self.recognize_faces)

        self.models = [
            "VGG-Face", 
            "Facenet", 
            "Facenet512", 
            "OpenFace", 
            "DeepFace", 
            "DeepID", 
            "ArcFace", 
            "Dlib", 
            "SFace",
            "GhostFaceNet",
        ]


    def load_image_tab1(self):
        image_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', 'images/t1/', "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if image_path:
            self.t1_up_img_path = image_path
            self.ui.t1_up_img.setPixmap(QPixmap(image_path))

    def load_image_tab2(self):
        image_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', 'images/t2/', "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if image_path:
            self.t2_up_img_path = image_path
            self.ui.t2_up_img.setPixmap(QPixmap(image_path))

    def get_face_embedding(self, image_path):
        try:
            embedding_objs = DeepFace.represent(
                img_path=image_path,
                model_name="VGG-Face",
                enforce_detection=True
            )
            if embedding_objs and len(embedding_objs) > 0:
                return embedding_objs[0]['embedding'], embedding_objs[0]['facial_area']
            return None, None
        except Exception as e:
            print(f"Error getting face embedding: {e}")
            return None, None

    def verify_faces(self, img1_path, img2_path):
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name="VGG-Face",
                enforce_detection=True
            )
            return result['verified']
        except Exception as e:
            print(f"Error verifying faces: {e}")
            return False

    def display_message(self):
        if self.t1_up_img_path:
            result = self.verify_faces(self.t1_org_img_path, self.t1_up_img_path)
            message = "Match ✅" if result else "Not a match ❌"
            self.ui.t1_msg.setText(message)
        else:
            QMessageBox.warning(self, "Warning", "Please load an image first 😐.")

    def recognize_faces(self):
        if not self.t2_up_img_path:
            QMessageBox.warning(self, "Warning", "Please load an image first 😐.")
            return

        # Load known face embeddings from the database
        known_face_embeddings, known_face_names = self.load_known_faces_from_db()
        
        # Get embeddings for the uploaded image
        try:
            # Load and process the image
            image = cv2.cvtColor(cv2.imread(self.t2_up_img_path), cv2.COLOR_BGR2RGB)
            
            # Get face embeddings from the image
            face_objs = DeepFace.represent(
                img_path=image,
                model_name=self.models[7], # Dlib
            )
            """
            face_locations = face_recognition.face_locations(image); # print(f'face_locations: {face_locations}\n')
            face_embeddings = face_recognition.face_encodings(image, face_locations); # print(f'face_encodings: {face_encodings}\n')
            """
            # Process each face found in the image
            face_embeddings = []
            face_locations = []
            for face_obj in face_objs:
                face_embedding = np.array(face_obj['embedding'])
                face_embeddings.append(face_embedding)

                facial_area = face_obj['facial_area']
                
                # Extract coordinates from facial_area
                left = facial_area['x']
                top = facial_area['y']
                right = left + facial_area['w']
                bottom = top + facial_area['h']

                face_locations.append((top, right, bottom, left))
            

            # Loop through each face found in the unknown image
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_embeddings):
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_embeddings, face_encoding)
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_embeddings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                # Draw a box around the face
                cv2.rectangle(image, (left, top), (right, bottom), (0, 85, 255, 220), 2)

                # Draw a label with a name below the face
                cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 85, 255, 220), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                
            # Convert the image to Qt format
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            
            # Display the result
            self.ui.t2_up_img.setPixmap(QPixmap.fromImage(qImg))

        except Exception as e:
            print(f"Error in face recognition: {e}")
            QMessageBox.warning(self, "Error", "Failed to process the image. Please try again.")

    def load_known_faces_from_db(self):
        known_face_embeddings = []
        known_face_names = []

        try:
            conn = mysql.connector.connect(
                host="localhost",
                user="my_user",
                password="my_password",
                database="images_db"
            )
            cursor = conn.cursor()

            cursor.execute("SELECT image_name, image_column FROM images_store")
            results = cursor.fetchall()

            for name, image_data in results:
                # Convert image data to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # DeepFace - Save temporary image file (DeepFace requires a file path)
                temp_path = "temp_image.jpg"
                cv2.imwrite(temp_path, img)
                
                # Get face embedding
                embedding_objs = DeepFace.represent(img_path=temp_path, model_name=self.models[7]) # Dlib

                known_face_embeddings.append(np.array(embedding_objs[0]['embedding']))
                known_face_names.append(os.path.splitext(name)[0])

                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
             
            cursor.close()
            conn.close()

        except Error as e:
            print(f"Error accessing database: {e}")

        print(f"Known face names: {known_face_names}")
        print(f"Known face embeddings: {known_face_embeddings}")
        print('---------------------------------')
        print(f'lenght of embeddings: {len(known_face_embeddings)}')
        print(f'lenght of one embedding: {len(known_face_embeddings[0])}')

        print('---------------------------------')
        return known_face_embeddings, known_face_names

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())