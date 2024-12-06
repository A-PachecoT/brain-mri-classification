import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os


def setup_multi_gpu():
    """
    Setup multi-GPU strategy if available.

    Returns:
        tf.distribute.Strategy: Distribution strategy
    """
    try:
        # Check for GPUs
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Training using {len(gpus)} GPUs")
        else:
            strategy = tf.distribute.get_strategy()  # Default strategy
            print("Training using default strategy")
        return strategy
    except:
        return tf.distribute.get_strategy()  # Default strategy


def train_model(
    model, train_dataset, val_dataset, epochs=50, checkpoint_dir="artifacts/models"
):
    """
    Train the model with parallel processing support.
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Enable mixed precision training for faster computation
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Optimize dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(AUTOTUNE)
    val_dataset = val_dataset.prefetch(AUTOTUNE)

    # Callbacks with parallel processing
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
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6
        ),
    ]

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        workers=os.cpu_count(),  # Parallel data loading
        use_multiprocessing=True,
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
