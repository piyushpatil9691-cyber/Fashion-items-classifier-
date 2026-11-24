# Fashion-MNIST Clothing Classifier

A friendly, deep-learning-powered tool that recognizes clothing items from images. It uses a Convolutional Neural Network (CNN) trained on the popular Fashion-MNIST dataset.

## Project Overview
This project helps you identify clothing pieces—like shirts, sneakers, or bags—using your own photos. It's built with TensorFlow/Keras for the "brain" (the CNN) and Streamlit for a simple, easy-to-use web interface.

## How to Get It Running

### 1. Set up your computer
First, make sure you're in the project folder, then set up a virtual environment so everything stays clean:
```powershell
# Create the environment
python -m venv venv

# Activate it (on Windows)
.\venv\Scripts\Activate
```

### 2. Install everything needed
Install the required tools listed in our requirements file:
```powershell
pip install -r requirements.txt
```

### 3. Train the model
If you want to train the AI yourself, just run:
```powershell
python train.py
```
*Note: This will automatically download the Fashion-MNIST dataset and train for 30 epochs (or until the model stops improving).*

### 4. See how well it did
To check the final stats, confusion matrix, and performance graphs, run:
```powershell
python evaluate.py
```

### 5. Launch the Web App
Finally, start the web interface to begin classifying:
```powershell
streamlit run app.py
```
You can then open the link provided in your browser (usually `http://localhost:8502`).

## Features
- **Smart Classification**: Upload a photo, and the model will tell you what clothing item it thinks it is.
- **Sample Testing**: Don't have a photo handy? Use the built-in test images to see how the model performs.
- **Performance Dashboard**: View how accurate the model is with training graphs and a confusion matrix.

## The 10 Categories it knows:
T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

---
*Built with love using TensorFlow, Keras, and Streamlit.*
