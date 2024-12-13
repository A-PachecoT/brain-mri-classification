import os
import tensorflow as tf
from data.data_loader import load_data, create_dataset
from models.model import create_model
from utils.train import train_model, plot_training_history
from utils.evaluate import evaluate_model, plot_confusion_matrix
from models.ensemble import ParallelEnsemble

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
    print("Cargando datos...")
    X_train, X_test, y_train, y_test = load_data(data_dir="data/raw")

    # Crear datasets optimizados con procesamiento paralelo
    print("Creando datasets...")
    train_dataset = create_dataset(X_train, y_train, is_training=True)
    test_dataset = create_dataset(X_test, y_test, is_training=False)

    # ========================
    # Modelo y Entrenamiento
    # ========================

    # Crear y compilar modelo con soporte multi-GPU
    print("Creando modelo...")
    model = create_model()
    model.summary()

    # Entrenar el modelo
    print("\nEntrenando modelo...")
    history = train_model(
        model, train_dataset, test_dataset, checkpoint_dir="artifacts/models"
    )

    # ========================
    # Visualización y Evaluación
    # ========================

    # Graficar historial de entrenamiento
    print("Graficando historial de entrenamiento...")
    plot_training_history(history, save_path="artifacts/plots/training_history.png")

    # Evaluar el modelo
    print("\nEvaluando modelo...")
    results = evaluate_model(model, test_dataset)

    # Graficar matriz de confusión
    print("Graficando matriz de confusión...")
    plot_confusion_matrix(
        results["y_true"],
        results["y_pred"],
        save_path="artifacts/plots/confusion_matrix.png",
    )

    # Crear y entrenar ensemble
    print("\nEntrenando modelos del ensemble...")
    ensemble = ParallelEnsemble(n_models=3)
    ensemble.train(train_dataset, test_dataset)

    # Evaluar ensemble
    print("\nEvaluando ensemble...")
    y_pred = ensemble.predict(X_test)

    # Graficar matriz de confusión
    print("Graficando matriz de confusión...")
    plot_confusion_matrix(
        y_test,
        y_pred,
        save_path="artifacts/plots/ensemble_confusion_matrix.png",
    )

    # Imprimir métricas individuales de cada modelo
    print("\nPesos de los modelos individuales (basados en precisión de validación):")
    for i, weight in enumerate(ensemble.weights):
        print(f"Modelo {i+1}: {weight:.3f}")

    print("\nEntrenamiento y evaluación completados!")
    print("Revisa el directorio 'artifacts/plots' para visualizaciones.")


if __name__ == "__main__":
    main()
