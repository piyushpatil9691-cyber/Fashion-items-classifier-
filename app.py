"""
Streamlit Web Application for Fashion-MNIST Clothing Classification
"""

import streamlit as st
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from preprocessing import preprocess_single_image, FASHION_LABELS
from tensorflow.keras.datasets import fashion_mnist

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/fashion_mnist_model.keras')

model = load_model()

st.title("Fashion-MNIST Clothing Classifier")

menu = ["Home", "Classification", "Performance"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.write("### Project Description")
    st.write("This CNN-based classifier recognizes clothing items from Fashion-MNIST.")
    st.write("Classes:", FASHION_LABELS)

elif choice == "Classification":
    st.write("### Classify Clothing")
    method = st.radio("Choose input method", ["Upload Image", "Use Sample Data"])

    img = None
    if method == "Upload Image":
        uploaded = st.file_uploader("Upload clothing item", type=['png', 'jpg'])
        if uploaded:
            img = Image.open(uploaded)
    else:
        _, (X_test, y_test) = fashion_mnist.load_data()
        idx = st.slider("Select sample index", 0, len(X_test)-1)
        img = Image.fromarray(X_test[idx])
        st.write(f"Sample Label: {FASHION_LABELS[y_test[idx]]}")

    if img:
        st.image(img, width=200, caption="Original Image")
        if st.button("Predict"):
            # Get the processed image
            processed = preprocess_single_image(img)

            # Show what the model sees
            st.write("Model input (grayscale/resized):")
            st.image(processed.reshape(28, 28), width=100)

            pred = model.predict(processed)
            confidence = np.max(pred)
            idx = np.argmax(pred)

            # Relaxed threshold (0.3 instead of 0.5)
            if confidence < 0.3:
                st.warning("I'm not confident about this one. Could you try a clearer image?")
                st.write(f"Highest confidence: {confidence*100:.1f}%")
            else:
                st.write(f"Prediction: **{FASHION_LABELS[idx]}**")
                st.write(f"Confidence: {confidence*100:.1f}%")
                st.bar_chart(pred[0])

elif choice == "Performance":
    st.write("### Model Evaluation")
    if os.path.exists('results/confusion_matrix.png'):
        st.image('results/confusion_matrix.png')
    if os.path.exists('results/performance_graphs.png'):
        st.image('results/performance_graphs.png')
