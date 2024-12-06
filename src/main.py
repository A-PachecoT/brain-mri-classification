import os
import tensorflow as tf
from data.data_loader import load_data, create_dataset
from models.model import create_model
from utils.train import train_model, plot_training_history
from utils.evaluate import evaluate_model, plot_confusion_matrix


def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)

    # Load and preprocess data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(data_dir="data/raw")

    # Create datasets
    train_dataset = create_dataset(X_train, y_train)
    test_dataset = create_dataset(X_test, y_test)

    # Create and compile model
    print("Creating model...")
    model = create_model()
    model.summary()

    # Train the model
    print("\nTraining model...")
    history = train_model(
        model, train_dataset, test_dataset, checkpoint_dir="artifacts/models"
    )

    # Plot training history
    print("Plotting training history...")
    plot_training_history(history, save_path="artifacts/plots/training_history.png")

    # Evaluate the model
    print("\nEvaluating model...")
    results = evaluate_model(model, test_dataset)

    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(
        results["y_true"],
        results["y_pred"],
        save_path="artifacts/plots/confusion_matrix.png",
    )

    print("\nTraining and evaluation completed!")
    print("Check 'artifacts/plots' directory for visualizations.")


if __name__ == "__main__":
    main()
