import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

def preprocess_to_lfw(image, output_size=(250, 250)):
    """
    Preprocess an image to resemble images in the LFW dataset.

    Args:
        image (np.ndarray): Input image array.
        output_size (tuple): Desired output image size (width, height).

    Returns:
        np.ndarray: Preprocessed image array.
    """
    # Convert the image to grayscale (optional, for LFW similarity)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load a face detector (Haar cascade or DNN-based detector)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")

    # Extract the first detected face (LFW contains one face per image)
    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]

    # Resize to the standard LFW size
    resized_face = cv2.resize(face, output_size)

    # Normalize pixel intensity to 0-1 range
    normalized_face = resized_face # / 255.0

    # Convert from BGR to RGB
    normalized_face = cv2.cvtColor(normalized_face.astype(np.float32), cv2.COLOR_BGR2RGB)

    return normalized_face

def get_image_embeddings(model, image, show_image=False):
    """
    Get embeddings for an input image using the provided model.

    Args:
        model (tf.keras.Model): Pre-trained model.
        image (np.ndarray): Input image array.

    Returns:
        np.ndarray: Embeddings for the input image.
    """
    # Get the output of the 'face_embedding' layer
    embedding_out = model.get_layer('face_embedding').output

    # Create a new model that takes the same input as the original model but outputs the embeddings
    embedding_model = tf.keras.models.Model(inputs=model.input,
                                            outputs=embedding_out,
                                            name=f'{model.name}_embedding')

    # Preprocess the image
    img = preprocess_to_lfw(image)

    if show_image == True:
        # Display the preprocessed image
        plt.imshow(img/255.0)
        plt.show()
    else:
        pass

    # Convert to TensorFlow tensor
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    # Resize the image to (128, 128) using tf.image.resize
    resized_image = tf.image.resize(img, [128, 128])

    # Add a batch dimension using tf.expand_dims
    img = tf.expand_dims(resized_image, axis=0)  # Add batch dimension

    # Get the embeddings for the input image
    embeddings = embedding_model.predict(img)

    # Ensure the output is a numpy array of size (128,)
    embeddings = np.squeeze(embeddings)

    return embeddings