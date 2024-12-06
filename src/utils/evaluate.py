import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os


def evaluate_model(model, test_dataset):
    """
    Evaluate the model on test data and print metrics.

    Args:
        model (tf.keras.Model): Trained model
        test_dataset (tf.data.Dataset): Test dataset

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred = []
    y_true = []

    for images, labels in test_dataset:
        predictions = model.predict(images)
        y_pred.extend(predictions.flatten() > 0.5)
        y_true.extend(labels.numpy())

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
