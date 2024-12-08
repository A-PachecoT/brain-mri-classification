import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

# ========================
# Configuración GPU
# ========================


def setup_multi_gpu():
    """
    Configurar estrategia multi-GPU si está disponible.

    Returns:
        tf.distribute.Strategy: Estrategia de distribución
    """
    try:
        # Verificar GPUs disponibles
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Entrenando usando {len(gpus)} GPUs")
        else:
            strategy = tf.distribute.get_strategy()  # Estrategia por defecto
            print("Entrenando usando estrategia por defecto")
        return strategy
    except:
        return tf.distribute.get_strategy()  # Estrategia por defecto


# ========================
# Entrenamiento
# ========================


def train_model(
    model, train_dataset, val_dataset, epochs=50, checkpoint_dir="artifacts/models"
):
    """
    Entrenar el modelo con soporte de procesamiento paralelo.
    """
    # Crear directorio de checkpoints si no existe
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Habilitar entrenamiento con precisión mixta para cálculos más rápidos
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Optimizar rendimiento del dataset
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(AUTOTUNE)
    val_dataset = val_dataset.prefetch(AUTOTUNE)

    # Callbacks con procesamiento paralelo
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

    # Entrenar el modelo
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        workers=os.cpu_count(),  # Carga de datos en paralelo
        use_multiprocessing=True,
    )

    return history


# ========================
# Visualización
# ========================


def plot_training_history(history, save_path="artifacts/plots/training_history.png"):
    """
    Graficar historial de entrenamiento mostrando curvas de precisión y pérdida.

    Args:
        history: Historial de entrenamiento de model.fit()
        save_path (str): Ruta para guardar el gráfico
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Gráfico de precisión
    ax1.plot(history.history["accuracy"], label="Entrenamiento")
    ax1.plot(history.history["val_accuracy"], label="Validación")
    ax1.set_title("Precisión del Modelo")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Precisión")
    ax1.legend()

    # Gráfico de pérdida
    ax2.plot(history.history["loss"], label="Entrenamiento")
    ax2.plot(history.history["val_loss"], label="Validación")
    ax2.set_title("Pérdida del Modelo")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Pérdida")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
