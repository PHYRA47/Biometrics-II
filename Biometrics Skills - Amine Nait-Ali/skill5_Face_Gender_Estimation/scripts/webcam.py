import cv2
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('./models/custom_CNN/custom_CNN.keras')

# Gender dictionary
gender_dict = {0: 'Male', 1: 'Female'}

def process_frame(face_roi, target_size=(128, 128)):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Resize
    resized_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    img_array = np.array(resized_image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Initialize face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Process face
        processed_face = process_frame(face_roi)
        
        # Make prediction
        predictions = model.predict(processed_face, verbose=0)
        pred_gender = gender_dict[round(predictions[0][0][0])]
        pred_age = round(predictions[1][0][0])
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 85, 0), 1)
        
        # Draw a label with gender and age below the face with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y + h + 1), (x + w, y + h + 35), (255, 85, 0), cv2.FILLED)
        alpha = 0.5  # Transparency factor
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Add text
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"Gender: {pred_gender}", (x + 6, y + h + 15), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Age: {pred_age}", (x + 6, y + h + 30), font, 0.5, (255, 255, 255), 1)
    # Display frame
    cv2.imshow('Age and Gender Estimation', frame)
    
    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()