import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os


def train_model(
    model, train_dataset, val_dataset, epochs=50, checkpoint_dir="artifacts/models"
):
    """
    Train the model with early stopping and model checkpointing.

    Args:
        model (tf.keras.Model): The model to train
        train_dataset (tf.data.Dataset): Training dataset
        val_dataset (tf.data.Dataset): Validation dataset
        epochs (int): Number of epochs to train
        checkpoint_dir (str): Directory to save model checkpoints

    Returns:
        history: Training history
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Callbacks
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.h5")
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
    ]

    # Train the model
    history = model.fit(
        train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks
    )

    return history


def plot_training_history(history, save_path="artifacts/plots/training_history.png"):
    """
    Plot training history showing accuracy and loss curves.

    Args:
        history: Training history from model.fit()
        save_path (str): Path to save the plot
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(history.history["accuracy"], label="Training")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Loss plot
    ax2.plot(history.history["loss"], label="Training")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
