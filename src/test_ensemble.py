import numpy as np
import tensorflow as tf
from models.ensemble import ParallelEnsemble


def test_ensemble():
    # Crear datos sintéticos con las dimensiones correctas
    X_test_small = np.random.random((20, 224, 224, 3))
    y_test_small = np.random.randint(0, 2, 20)

    # Crear un ensemble más pequeño para prueba
    test_ensemble = ParallelEnsemble(n_models=2)

    # Crear modelos simples para prueba
    for _ in range(2):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(8, 3, activation="relu"),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        test_ensemble.models.append(model)

    # Asignar pesos iguales para prueba
    test_ensemble.weights = np.ones(2) / 2

    # Intentar predicción
    try:
        predictions, metrics = test_ensemble.predict(X_test_small, y_test_small)
        print("✓ Test exitoso!")
        print(f"Predicciones shape: {predictions.shape}")
        print(f"Métricas: {metrics}")
        return True
    except Exception as e:
        print("✗ Test fallido!")
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    test_ensemble()
