import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .model import create_model


class ParallelEnsemble:
    def __init__(self, n_models=3):
        self.n_models = n_models
        self.models = []

        # Detectar GPUs disponibles
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            self.strategy = tf.distribute.MirroredStrategy()
            print(f"Usando {len(gpus)} GPU(s)")
        else:
            self.strategy = tf.distribute.get_strategy()
            print("GPU no disponible, usando CPU")

        # Configurar crecimiento de memoria
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass

    def _train_model(self, train_dataset, val_dataset):
        """Entrena un modelo individual del ensemble y retorna su accuracy"""
        with self.strategy.scope():
            model = create_model()
            # Usar bootstrap sampling para bagging
            bootstrap_dataset = train_dataset.shuffle(1000).take(
                train_dataset.cardinality()
            )
            history = model.fit(bootstrap_dataset, validation_data=val_dataset)
            # Obtener accuracy final de validación
            val_accuracy = history.history["val_accuracy"][-1]
            return model, val_accuracy

    def train(self, train_dataset, val_dataset):
        """Entrena múltiples modelos en paralelo y guarda sus pesos"""
        with ThreadPoolExecutor(max_workers=self.n_models) as executor:
            futures = []
            for _ in range(self.n_models):
                future = executor.submit(self._train_model, train_dataset, val_dataset)
                futures.append(future)

            # Guardar modelos y sus accuracies
            self.models = []
            self.weights = []
            for future in futures:
                model, accuracy = future.result()
                self.models.append(model)
                self.weights.append(accuracy)

            # Normalizar pesos
            self.weights = np.array(self.weights)
            self.weights = self.weights / np.sum(self.weights)

    def predict(self, X):
        """Predicción por votación ponderada según accuracy"""
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)

        # Votación ponderada
        ensemble_pred = np.sum(predictions, axis=0) > 0.5
        return ensemble_pred
