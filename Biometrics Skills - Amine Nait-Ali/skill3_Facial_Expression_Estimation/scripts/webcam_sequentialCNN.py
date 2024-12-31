import cv2
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt


# Source of model:  https://github.com/vicksam/fer-model/tree/main/model
# Source of script: https://www.kaggle.com/code/abhisheksingh016/emotion-detection-jupyter-notebook

# Function to recognize emotion

def recognize_emotion(image, 
                      model=load_model('models/sequentialCNN/sequentialCNN.h5'), 
                      face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'), 
                      class_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                      ):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        # Resize the image
        processed_image = cv2.resize(roi_gray, (48, 48))
        
        # Normalize the image
        processed_image = processed_image / 255.0
        
        # Expand the dimensions
        processed_image = np.expand_dims(processed_image, axis=0)
        processed_image = np.expand_dims(processed_image, axis=-1)
        
        # Predict the emotion
        predictions = model.predict(processed_image)
        emotion_label = class_names[np.argmax(predictions)]
        
        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 85, 0, 220), 2)
        
        # Put the emotion label
        cv2.putText(image, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 85, 0, 220), 2)
    
    return image

# Example usage:
# image = cv2.imread('dataset/test_samples/niggas.jpg')
# result_image = recognize_emotion(image, model, face_cascade, class_names)

# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)); plt.show()

# Real-time recognition

# Load the model
model = load_model('models/sequentialCNN/sequentialCNN.h5')
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Resize the image
        processed_image = cv2.resize(roi_gray, (48, 48))
        
        # Normalize the image
        processed_image = processed_image / 255.0
        
        # Expand the dimensions
        processed_image = np.expand_dims(processed_image, axis=0)
        processed_image = np.expand_dims(processed_image, axis=-1)
        
        # Predict the emotion
        predictions = model.predict(processed_image)
        emotion_label = class_names[np.argmax(predictions)]
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 85, 0, 220), 2)
        
        # Put the emotion label
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 85, 0, 220), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()


