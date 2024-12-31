import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('models/mobilenetv2/mobilenetv2.h5')

# Get the class names
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the face_cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
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
        processed_image = cv2.resize(roi_color, (224, 224))

        # Expand the dimensions
        processed_image = np.expand_dims(processed_image, axis=0)  # Add a batch dimension (1, 224, 224, 3)

        # Normalize the image
        processed_image = processed_image / 255.0

        # Predict the emotion
        predictions = model.predict(processed_image)
        max_index = np.argmax(predictions[0])
        emotion = class_names[max_index]

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 85, 0, 220), 2)

        # Put the emotion text above the rectangle
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 85, 0, 220), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()