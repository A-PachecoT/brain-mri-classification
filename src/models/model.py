import tensorflow as tf
from tensorflow.keras import layers, models
import logging

logger = logging.getLogger("MRI-Classification")

# ========================
# Definición del Modelo
# ========================


def create_model(input_shape=(128, 128, 3)):
    """
    Crear un modelo CNN con soporte multi-GPU.
    """
    # Configurar estrategia de distribución
    strategy = (
        tf.distribute.MirroredStrategy()
        if len(tf.config.list_physical_devices("GPU")) > 1
        else tf.distribute.get_strategy()
    )

    with strategy.scope():
        logger.info("Construyendo arquitectura CNN...")
        # ========================
        # Arquitectura CNN
        # ========================
        model = models.Sequential(
            [
                # Capas convolucionales y de pooling
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                # Capas densas y dropout
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        # Usar precisión mixta para cálculos más rápidos
        optimizer = tf.keras.optimizers.Adam()
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        logger.info("Compilando modelo...")
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

    return model
