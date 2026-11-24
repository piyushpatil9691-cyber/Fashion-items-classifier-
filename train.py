"""
Training script for our clothing classifier.

This script takes the Fashion-MNIST data, feeds it to our CNN,
and saves the best version of our trained "brain" to the models/ folder.
"""

import os
import numpy as np
import tensorflow as tf
from preprocessing import load_and_preprocess_data
from model import build_model, get_data_augmentation

def train():
    # Step 1: Get our data ready
    print("Getting clothing data ready...")
    (X_train, y_train), (X_val, y_val), _ = load_and_preprocess_data()

    # Step 2: Build our CNN model
    print("Setting up the AI structure...")
    model = build_model()
    model.summary()

    # Step 3: Setup tools to save our progress and stop if it's not improving
    os.makedirs('models', exist_ok=True)
    checkpoint_path = 'models/fashion_mnist_model.keras'

    callbacks = [
        # Stop training if the model stops getting better after 5 tries
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
            start_from_epoch=10
        ),
        # Save the best version automatically
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Step 4: Add data variety to help the model learn
    data_aug = get_data_augmentation()

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

    train_ds = train_ds.shuffle(10000).batch(64).map(
        lambda x, y: (data_aug(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    # Step 5: Let the learning begin!
    print("Training... time to make some coffee!")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks
    )

    # Save everything for later use
    model.save('models/fashion_mnist_model_final.keras')
    os.makedirs('results', exist_ok=True)
    np.save('results/history.npy', history.history)
    print(f"Great! Training is complete. The best model is saved at {checkpoint_path}")

if __name__ == "__main__":
    train()
