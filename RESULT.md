# Fashion-MNIST Model Training Results

## Summary
The model was trained on the Fashion-MNIST dataset using a 3-block CNN architecture.

## Performance Metrics
| Metric | Value |
|---|---|
| Test Accuracy | 88% |
| Macro Avg F1-Score | 0.88 |
| Weighted Avg F1-Score | 0.88 |

## Confusion Matrix
[Confusion Matrix](results/confusion_matrix.png)

## Training History
[Performance Graphs](results/performance_graphs.png)

## Detailed Classification Report
```text
              precision    recall  f1-score   support

 T-shirt/top       0.81      0.86      0.83      1000
     Trouser       0.99      0.98      0.98      1000
    Pullover       0.85      0.82      0.83      1000
       Dress       0.88      0.88      0.88      1000
        Coat       0.80      0.78      0.79      1000
      Sandal       0.97      0.98      0.98      1000
       Shirt       0.65      0.65      0.65      1000
     Sneaker       0.91      0.98      0.94      1000
         Bag       0.97      0.97      0.97      1000
  Ankle boot       0.99      0.92      0.95      1000

    accuracy                           0.88     10000
   macro avg       0.88      0.88      0.88     10000
weighted avg       0.88      0.88      0.88     10000
```
