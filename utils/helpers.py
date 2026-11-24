"""
Utility functions for the Fashion-MNIST application.
"""

import numpy as np
from PIL import Image, ImageOps

# Clothing categories
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

def preprocess_image_for_prediction(image):
    """
    Preprocess a PIL image for model prediction.
    Handles resizing, grayscale conversion, normalization, and inversion.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to array
    img_array = np.array(image)

    # Invert if needed (Fashion-MNIST is white on black)
    if img_array[0,0] > 127:
        image = ImageOps.invert(image)
        img_array = np.array(image)

    # Normalize
    img_array = img_array.astype('float32') / 255.0

    # Add batch and channel dimensions: (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=(0, -1))

    return img_array, np.array(image)

def predict_image(model, processed_image):
    """
    Make prediction on preprocessed image.
    Returns: predicted_class, confidence, all_probabilities
    """
    predictions = model.predict(processed_image, verbose=0)
    probs = predictions[0]
    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return predicted_class, confidence, probs