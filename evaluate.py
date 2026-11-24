"""
Fashion-MNIST Evaluation Script.
Generates metrics and plots for the trained model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.models import load_model
from preprocessing import load_and_preprocess_data, FASHION_LABELS

def evaluate():
    os.makedirs('results', exist_ok=True)

    # Load data
    _, _, (X_test, y_test) = load_and_preprocess_data()

    # Load model
    model = load_model('models/fashion_mnist_model.keras')

    # Predict
    print("Evaluating model on test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics
    report = classification_report(y_test, y_pred, target_names=list(FASHION_LABELS.values()))
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)
    print("Classification report saved to results/classification_report.txt")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=FASHION_LABELS.values(), yticklabels=FASHION_LABELS.values())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to results/confusion_matrix.png")

    # Accuracy/Loss History Plot
    if os.path.exists('results/history.npy'):
        history = np.load('results/history.npy', allow_pickle=True).item()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train')
        plt.plot(history['val_accuracy'], label='Val')
        plt.title('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title('Loss')
        plt.legend()

        plt.savefig('results/performance_graphs.png')
        plt.close()
        print("Performance graphs saved to results/performance_graphs.png")

if __name__ == "__main__":
    evaluate()
