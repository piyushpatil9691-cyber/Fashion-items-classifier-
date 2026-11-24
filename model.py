"""
CNN Architecture for Fashion-MNIST Classification.

This script designs our model's "brain" structure. We're using a 3-block
CNN that looks at small patterns (like edges and textures) to identify clothing.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Sequential

def build_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Constructs our deep learning model.
    We're using three 'convolutional blocks' to learn increasingly complex features.
    """
    model = Sequential([
        # Block 1: Looking for simple edges and shapes
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(), # Helps the model learn faster and more reliably
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25), # Prevents the model from just 'memorizing' the training photos

        # Block 2: Identifying more complex textures
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 3: Recognizing full parts of clothing (like sleeves or soles)
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Final Classifier: Putting all the pieces together
        layers.Flatten(), # Turns our 2D data into a 1D list
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # The output layer with 10 'neurons'—one for each clothing category
        layers.Dense(num_classes, activation='softmax')
    ])

    # Telling the model how to learn and how to measure its own success
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def get_data_augmentation():
    """
    Creates a 'data factory' that makes small, random tweaks to our training images.
    This helps the model recognize a shirt even if it's slightly rotated or zoomed.
    """
    return Sequential([
        layers.RandomRotation(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.RandomFlip("horizontal"),  # Added horizontal flip for robustness
    ])
