import os
import tensorflow as tf
from data.data_loader import load_data, create_dataset
from models.model import create_model
from utils.train import train_model, plot_training_history
from utils.evaluate import evaluate_model, plot_confusion_matrix
from utils.ensemble import ParallelEnsemble

# ========================
# Configuración Principal
# ========================


def main():
    # Establecer semillas aleatorias para reproducibilidad
    tf.random.set_seed(42)

    # Habilitar crecimiento de memoria para GPUs
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # ========================
    # Carga de Datos
    # ========================

    # Cargar y preprocesar datos con procesamiento paralelo
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(data_dir="data/raw")

    # Crear datasets optimizados con procesamiento paralelo
    print("Creating datasets...")
    train_dataset = create_dataset(X_train, y_train, is_training=True)
    test_dataset = create_dataset(X_test, y_test, is_training=False)

    # ========================
    # Modelo y Entrenamiento
    # ========================

    # Crear y compilar modelo con soporte multi-GPU
    print("Creating model...")
    model = create_model()
    model.summary()

    # Entrenar el modelo
    print("\nTraining model...")
    history = train_model(
        model, train_dataset, test_dataset, checkpoint_dir="artifacts/models"
    )

    # ========================
    # Visualización y Evaluación
    # ========================

    # Graficar historial de entrenamiento
    print("Plotting training history...")
    plot_training_history(history, save_path="artifacts/plots/training_history.png")

    # Evaluar el modelo
    print("\nEvaluating model...")
    results = evaluate_model(model, test_dataset)

    # Graficar matriz de confusión
    print("Plotting confusion matrix...")
    plot_confusion_matrix(
        results["y_true"],
        results["y_pred"],
        save_path="artifacts/plots/confusion_matrix.png",
    )

    # Crear y entrenar ensemble
    print("\nTraining ensemble models...")
    ensemble = ParallelEnsemble(n_models=3)
    ensemble.train(train_dataset, test_dataset)

    # Evaluar ensemble
    print("\nEvaluating ensemble...")
    y_pred = ensemble.predict(X_test)

    # Graficar matriz de confusión
    print("Plotting confusion matrix...")
    plot_confusion_matrix(
        y_test,
        y_pred,
        save_path="artifacts/plots/ensemble_confusion_matrix.png",
    )

    # Imprimir métricas individuales de cada modelo
    print("\nIndividual model weights (based on validation accuracy):")
    for i, weight in enumerate(ensemble.weights):
        print(f"Model {i+1}: {weight:.3f}")

    print("\nTraining and evaluation completed!")
    print("Check 'artifacts/plots' directory for visualizations.")


if __name__ == "__main__":
    main()
