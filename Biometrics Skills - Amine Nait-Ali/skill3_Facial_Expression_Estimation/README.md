<div align="center">
  <h1><strong> 👤Biometrics Skill 3: Facial Expression Estimation </strong></h1>
  
  https://github.com/user-attachments/assets/52b92302-bcdb-403b-9580-960e7f4ee091
</div>

## ✨ Key Features

- 🔄 **Multiple Recognition Models**:
  - Face Recognition
  - Dlib
  - Facenet
  - VGG-Face
  - Easy to extend with more models!

- 🖥️ **Multiple Recognition Modes**:
  - 📸 Image-to-Image Comparison
  - 📹 Capture and Recognize
  - 🎥 Real-time Webcam Recognition
  - 👥 Face Enrollment System

- 💾 **Database Integration**:
  - MySQL backend for storing face data
  - Secure image storage and retrieval
  - Easy enrollment management

- 🎨 **Modern User Interface**:
  - Clean and intuitive PyQt5-based GUI
  - Tab-based navigation

## 🛠️ Prerequisites

- Python 3.6+
- MySQL Server
- Required Python packages:
  ```
  opencv-python
  numpy
  tensorflow
  face_recognition
  deepface
  PyQt5
  mysql-connector-python
  ```

## 📦 Installation

1. Clone the repository and install required packages:
```bash
pip install -r requirements.txt
```

2. Set Docker Image
```docker run --name mysql-container -e MYSQL_ROOT_PASSWORD=my-secret-pw -e MYSQL_DATABASE=images_db -e MYSQL_USER=my_user -e MYSQL_PASSWORD=my_password -p 3306:3306 -d mysql:latest```


3. Set up MySQL database:
```sql
CREATE DATABASE images_db;
CREATE TABLE images_store (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_name VARCHAR(255),
    image_column LONGBLOB
);
```

4. Configure database connection in `config.py`:
```python
db_config = {
    "host": "localhost",
    "user": "your_username",
    "password": "your_password",
    "database": "images_db",
    "port": 3306
}
```

## 🚀 Usage

1. Start the application:
```bash
python main.py
```

2. **Image Recognition Mode** 📸:
   - Select your preferred model
   - Upload an image
   - Click "Recognize Faces"
   - View results and similarity scores

3. **Capture and Recognize Mode** 📷:
   - Start the camera
   - Capture an image
   - View real-time recognition results
   - Check CMC curve for accuracy analysis

4. **Real-time Recognition Mode** 🎥:
   - Start the webcam
   - View real-time face recognition results
   - Switch between different models

5. **Face Enrollment** 👤:
   - Upload a face image
   - Enter the person's name
   - Manage enrolled faces through the interface

## 🔍 Technical Architecture

- 🎯 **Face Detection**: OpenCV Haar Cascade Classifier
- 🧠 **Feature Extraction**: Multiple deep learning models
- 📊 **Similarity Metrics**:
  - Euclidean Distance
  - Cosine Similarity
- 💾 **Storage**: MySQL database for face data
- 🖼️ **Image Processing**: OpenCV and NumPy
- 🎨 **GUI Framework**: PyQt5

