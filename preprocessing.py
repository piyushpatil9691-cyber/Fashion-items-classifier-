"""
Preprocessing pipeline for Fashion-MNIST dataset and real-world photos.
"""

import numpy as np
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.datasets import fashion_mnist

# Clothing categories mapping
FASHION_LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

def load_and_preprocess_data():
    """
    Loads the Fashion-MNIST dataset from TensorFlow, splits it into
    training/validation/test sets, and prepares it for our model.
    """
    # Fetch the dataset (downloads automatically if not cached)
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    # Normalize pixels: 0-255 becomes 0.0-1.0 (easier for the neural network)
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape to include a channel dimension: (28, 28) -> (28, 28, 1)
    # This tells the model "this is a grayscale image"
    X_train_full = np.expand_dims(X_train_full, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Split training data: 50k for training, 10k for validation
    validation_size = 10000
    X_val = X_train_full[:validation_size]
    y_val = y_train_full[:validation_size]
    X_train = X_train_full[validation_size:]
    y_train = y_train_full[validation_size:]

    print(f"Dataset loaded and ready:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def preprocess_single_image(image):
    """
    Preprocesses real-world photos to look like Fashion-MNIST images.
    If the image is already a clean Fashion-MNIST sample, we should avoid
    over-processing it (e.g., double inversion).
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # 1. Convert to grayscale
    image = ImageOps.grayscale(image)

    # 2. Resize to 28x28
    image = image.resize((28, 28))

    # 3. Handle background inversion carefully
    # Fashion-MNIST standard is white object on black background (pixel 0 = black)
    # Real world photos often are black object on white background (pixel 255 = white)
    img_array = np.array(image)

    # Check corners to see if background is mostly light (white) -> needs inversion
    # Only invert if it looks like an inverted real-world photo
    corner_mean = np.mean([
        img_array[0,0], img_array[0, -1],
        img_array[-1, 0], img_array[-1, -1]
    ])

    # If corners are bright (>180), it's likely a real-world photo with white background -> invert
    if corner_mean > 180:
        image = ImageOps.invert(image)
        img_array = np.array(image)

    # 4. Optional denoising (light)
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

    # 5. Normalize and reshape
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))

    return img_array