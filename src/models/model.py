import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(input_shape=(128, 128, 3)):
    """
    Create a CNN model with multi-GPU support.
    """
    # Setup distribution strategy
    strategy = (
        tf.distribute.MirroredStrategy()
        if len(tf.config.list_physical_devices("GPU")) > 1
        else tf.distribute.get_strategy()
    )

    with strategy.scope():
        model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        # Use mixed precision for faster computation
        optimizer = tf.keras.optimizers.Adam()
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )

    return model
