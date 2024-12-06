import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor


def parallel_predict(model, images_batch):
    """
    Make predictions in parallel for a batch of images.
    """
    return model.predict(images_batch)


def evaluate_model(model, test_dataset):
    """
    Evaluate the model on test data with parallel processing.
    """
    AUTOTUNE = tf.data.AUTOTUNE
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    # Get predictions using parallel processing
    y_pred = []
    y_true = []

    # Create a thread pool for parallel prediction
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []

        for images, labels in test_dataset:
            futures.append(executor.submit(parallel_predict, model, images))
            y_true.extend(labels.numpy())

        # Collect results
        for future in futures:
            predictions = future.result()
            y_pred.extend(predictions.flatten() > 0.5)

    # Convert to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Tumor", "Tumor"]))

    return {"y_true": y_true, "y_pred": y_pred}


def plot_confusion_matrix(
    y_true, y_pred, save_path="artifacts/plots/confusion_matrix.png"
):
    """
    Plot confusion matrix.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        save_path (str): Path to save the plot
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Tumor", "Tumor"],
        yticklabels=["No Tumor", "Tumor"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(save_path)
    plt.close()
