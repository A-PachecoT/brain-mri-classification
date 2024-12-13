import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger("MRI-Classification")

# ========================
# Funciones de Evaluación
# ========================


def parallel_predict(model, images_batch):
    """
    Realizar predicciones en paralelo para un lote de imágenes.
    """
    return model.predict(images_batch)


def evaluate_model(model, test_dataset):
    """
    Evaluar el modelo en datos de prueba con procesamiento paralelo.
    """
    AUTOTUNE = tf.data.AUTOTUNE
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    # Obtener predicciones usando procesamiento paralelo
    y_pred = []
    y_true = []

    # Crear un pool de hilos para predicción paralela
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []

        for images, labels in test_dataset:
            y_true.extend(labels.numpy())
            pred = model.predict(images, verbose=0)
            y_pred.extend(pred.flatten() > 0.5)

    # Convertir a arrays numpy
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Calcular métricas
    accuracy = np.mean(y_pred == y_true)

    # Imprimir reporte de clasificación
    logger.info("\nReporte de Clasificación:")
    logger.info(
        classification_report(y_true, y_pred, target_names=["Sin Tumor", "Con Tumor"])
    )

    return {"y_true": y_true, "y_pred": y_pred, "accuracy": accuracy}


# ========================
# Visualización
# ========================


def plot_confusion_matrix(
    y_true, y_pred, save_path="artifacts/plots/confusion_matrix.png"
):
    """
    Graficar matriz de confusión.

    Args:
        y_true (np.ndarray): Etiquetas verdaderas
        y_pred (np.ndarray): Etiquetas predichas
        save_path (str): Ruta para guardar el gráfico
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    # Crear gráfico
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Sin Tumor", "Con Tumor"],
        yticklabels=["Sin Tumor", "Con Tumor"],
    )
    plt.title("Matriz de Confusión")
    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")

    # Guardar y cerrar
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Matriz de confusión guardada en {save_path}")
