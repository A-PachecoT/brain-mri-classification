import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .model import create_model
import multiprocessing


class ParallelEnsemble:
    def __init__(self, n_models=3, batch_size=32):
        self.n_models = n_models
        self.models = []
        self.batch_size = batch_size

        # Detectar GPUs y CPUs disponibles
        self.gpus = tf.config.list_physical_devices("GPU")
        self.cpu_count = multiprocessing.cpu_count()

        if self.gpus:
            self.strategy = tf.distribute.MirroredStrategy()
            print(f"Usando {len(self.gpus)} GPU(s)")
            # Configurar crecimiento de memoria
            for gpu in self.gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except:
                    pass
        else:
            self.strategy = tf.distribute.get_strategy()
            print(f"GPU no disponible, usando {self.cpu_count} CPU cores")

    def _predict_batch(self, model, batch):
        """Predicción paralela por lotes"""
        return model.predict(batch, batch_size=self.batch_size)

    def _train_model(self):
        """Entrena un modelo individual del ensemble usando CPU cores"""
        with self.strategy.scope():
            model = create_model()
            return model

    def train(self, train_dataset, val_dataset):
        """Entrena múltiples modelos en paralelo usando threads"""
        # Usar ThreadPoolExecutor para paralelismo a nivel de thread
        with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
            # Primero crear los modelos en paralelo
            model_futures = [
                executor.submit(self._train_model) for _ in range(self.n_models)
            ]
            self.models = [future.result() for future in model_futures]

            # Configurar dataset para procesamiento paralelo
            AUTOTUNE = tf.data.AUTOTUNE

            # Entrenar modelos
            self.weights = []
            for model in self.models:
                # Crear copia del dataset para cada modelo
                bootstrap_dataset = (
                    train_dataset.shuffle(1000)
                    .take(train_dataset.cardinality())
                    .prefetch(AUTOTUNE)
                )

                # Optimizar para CPU
                if not self.gpus:
                    bootstrap_dataset = bootstrap_dataset.map(
                        lambda x, y: (x, y), num_parallel_calls=AUTOTUNE
                    ).cache()

                # Entrenar modelo
                history = model.fit(
                    bootstrap_dataset,
                    validation_data=val_dataset,
                    workers=self.cpu_count if not self.gpus else 1,
                    use_multiprocessing=False,  # Evitar problemas de pickle
                )

                val_accuracy = history.history["val_accuracy"][-1]
                self.weights.append(val_accuracy)

            # Normalizar pesos
            self.weights = np.array(self.weights)
            self.weights = self.weights / np.sum(self.weights)

    def predict(self, X):
        """Predicción paralela por lotes con votación ponderada"""
        # Dividir datos en lotes para predicción paralela
        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        predictions = []
        with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
            for model, weight in zip(self.models, self.weights):
                model_preds = []
                futures = []

                # Procesar cada lote en paralelo
                for i in range(n_batches):
                    start_idx = i * self.batch_size
                    end_idx = min((i + 1) * self.batch_size, n_samples)
                    batch = X[start_idx:end_idx]

                    future = executor.submit(self._predict_batch, model, batch)
                    futures.append(future)

                # Recolectar resultados
                for future in futures:
                    batch_pred = future.result()
                    model_preds.append(batch_pred)

                # Combinar predicciones del modelo
                model_preds = np.concatenate(model_preds) * weight
                predictions.append(model_preds)

        # Votación ponderada final
        ensemble_pred = np.sum(predictions, axis=0) > 0.5
        return ensemble_pred
