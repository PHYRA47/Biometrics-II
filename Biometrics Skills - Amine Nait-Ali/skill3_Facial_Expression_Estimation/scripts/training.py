import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Ensure TensorFlow uses GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the dataset
X_train = np.load('data/subdataset/sub_X_train.npy.npz')['arr_0']
y_train = np.load('data/subdataset/sub_y_train.npy.npz')['arr_0']


"""
# Base model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)

# Input layer
inputs = base_model.input

# Output layer
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu', name='face_embedding')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(7, activation='softmax')(x)

# Define the model
model = Model(inputs=inputs, outputs=outputs, name='custom_mobilenetv2')

optimizer = Adam(learning_rate=0.0003)
scc_loss = SparseCategoricalCrossentropy()

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
"""

# Base model
base_model = tf.keras.applications.MobileNetV2()

# Input layer
base_input = base_model.input

# Output layer
base_output = base_model.layers[-2].output

# Custom model
x = Dense(128)(base_output)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
final_output = Dense(7, activation='softmax')(x)

# Define the model
model = Model(inputs=base_input, outputs=final_output, name='custom_mobilenetv2')

# model.compile(optimizer=optimizer, loss=scc_loss, metrics=['accuracy'])
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])

"""
history = model.fit(
    X_train, y_train, 
    epochs=25, 
    batch_size=32,  
    validation_split=0.2,
    shuffle=True,
    callbacks=[
        reduce_lr,
        early_stopping
    ]
)
"""

history = model.fit(
    X_train, y_train, 
    epochs=25,
)

# Save the history
np.savez('models/custom_mobilenetv2_history.npz', history.history)

# Save the model
model.save('models/custom_mobilenetv2.h5')
model.save('models/custom_mobilenetv2.keras')
