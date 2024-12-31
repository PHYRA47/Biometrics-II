import tensorflow as tf
import cv2
import os
import numpy as np  

from sklearn.utils import shuffle

# Class names
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
train_dir = 'dataset/train'; test_dir = 'dataset/test'

IMG_SIZE = 224 # ImageNet model's image size

# Read image files and change them to array

def create_training_data(folder_path):
    # Pre-allocate memory for the arrays
    num_images = sum([len(files) for r, d, files in os.walk(folder_path)])
    images = np.zeros((num_images, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)  # Pre-allocate
    labels = np.zeros(num_images, dtype=np.int32)  # Pre-allocate

    index = 0
    for idx, class_name in enumerate(class_names):
        path = os.path.join(folder_path, class_name)
        for image_file in os.listdir(path):
            image = cv2.imread(os.path.join(path, image_file))
            image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            # Normalize the image
            image_resized = image_resized / 255.0 
            
            # Assign to pre-allocated arrays
            images[index] = image_resized 
            labels[index] = idx
            index += 1

    return images, labels

X_train, y_train = create_training_data(train_dir)

X_train, y_train = shuffle(X_train, y_train, random_state=0) 

# Save the data
# np.savez_compressed('data/X_train_shuffled', X_train)
# np.savez_compressed('data/y_train_shuffled', y_train)

# Load the data suffled 
X_train = np.load('data/X_train_shuffled.npy.npz')['arr_0']
y_train = np.load('data/y_train_shuffled.npy.npz')['arr_0']

# Create a sub-dataset with 10,000 observations
sub_X_train = X_train[:10000]
sub_y_train = y_train[:10000]

print('sub_X_train:', sub_X_train.shape)
print('sub_y_train:', sub_y_train.shape)

# Save the sub-dataset
# np.savez_compressed('data/subdataset/sub_X_train.npy', sub_X_train)
# np.savez_compressed('data/subdataset/sub_y_train.npy', sub_y_train)

# Load the sub-dataset
X_train = np.load('data/subdataset/sub_X_train.npy.npz')['arr_0']
y_train = np.load('data/subdataset/sub_y_train.npy.npz')['arr_0']

print('sub_X_train:', sub_X_train.shape)
print('sub_y_train:', sub_y_train.shape)