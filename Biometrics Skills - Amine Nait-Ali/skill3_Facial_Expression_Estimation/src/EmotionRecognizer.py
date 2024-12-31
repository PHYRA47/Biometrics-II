import cv2
import numpy as np

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
