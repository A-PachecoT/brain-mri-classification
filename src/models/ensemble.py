import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .model import create_model
import multiprocessing
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParallelEnsemble")

# Suprimir warnings de TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR


class ParallelEnsemble:
    def __init__(self, n_models=3, batch_size=32, verbose=1):
        self.n_models = n_models
        self.models = []
        self.batch_size = batch_size
        self.verbose = verbose

        # Detectar GPUs y CPUs disponibles
        self.gpus = tf.config.list_physical_devices("GPU")
        self.cpu_count = multiprocessing.cpu_count()

        if self.gpus:
            self.strategy = tf.distribute.MirroredStrategy()
            logger.info(f"✓ Usando {len(self.gpus)} GPU(s)")
            # Configurar crecimiento de memoria
            for gpu in self.gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except:
                    pass
        else:
            self.strategy = tf.distribute.get_strategy()
            logger.info(f"✓ Usando {self.cpu_count} CPU cores")

    def _predict_batch(self, model, batch):
        """Predicción paralela por lotes"""
        return model.predict(batch, batch_size=self.batch_size, verbose=0)

    def _train_model(self):
        """Entrena un modelo individual del ensemble"""
        with self.strategy.scope():
            model = create_model()
            return model

    @tf.function(reduce_retracing=True)
    def _predict_tf_function(self, model, batch):
        """Función TF optimizada para predicción"""
        return model(batch, training=False)

    def train(self, train_dataset, val_dataset):
        """Entrena múltiples modelos en paralelo"""
        logger.info("Iniciando entrenamiento del ensemble...")

        with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
            # Crear modelos en paralelo
            logger.info(f"Creando {self.n_models} modelos en paralelo...")
            model_futures = [
                executor.submit(self._train_model) for _ in range(self.n_models)
            ]
            self.models = [future.result() for future in model_futures]

            # Configurar dataset
            AUTOTUNE = tf.data.AUTOTUNE
            dataset_size = int(train_dataset.cardinality().numpy())  # Convertir a int

            # Entrenar modelos
            self.weights = []
            for i, model in enumerate(self.models, 1):
                logger.info(f"\nEntrenando modelo {i}/{self.n_models}")

                # Aumentar diversidad con diferentes configuraciones
                bootstrap_size = np.random.uniform(0.7, 1.0)
                dropout_rate = np.random.uniform(0.3, 0.5)

                # Calcular tamaño del bootstrap
                n_samples = int(dataset_size * bootstrap_size)

                bootstrap_dataset = (
                    train_dataset.shuffle(1000, seed=i)
                    .take(n_samples)  # Usar el número calculado
                    .prefetch(AUTOTUNE)
                )

                if not self.gpus:
                    bootstrap_dataset = bootstrap_dataset.map(
                        lambda x, y: (x, y), num_parallel_calls=AUTOTUNE
                    ).cache()

                # Configurar callbacks específicos por modelo
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=5,
                        restore_best_weights=True,
                        verbose=0,
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.2, patience=3, verbose=0
                    ),
                ]

                history = model.fit(
                    bootstrap_dataset,
                    validation_data=val_dataset,
                    callbacks=callbacks,
                    workers=self.cpu_count if not self.gpus else 1,
                    use_multiprocessing=False,
                    verbose=self.verbose,
                )

                val_accuracy = history.history["val_accuracy"][-1]
                self.weights.append(val_accuracy)
                logger.info(
                    f"Modelo {i} - Accuracy: {val_accuracy:.4f} "
                    f"(bootstrap_size={bootstrap_size:.2f}, dropout={dropout_rate:.2f})"
                )

            # Normalizar pesos
            self.weights = np.array(self.weights)
            self.weights = self.weights / np.sum(self.weights)

            logger.info("\nPesos finales del ensemble:")
            for i, w in enumerate(self.weights, 1):
                logger.info(f"Modelo {i}: {w:.4f}")

    def predict(self, X):
        """Predicción paralela por lotes"""
        logger.info("Realizando predicciones del ensemble...")

        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        predictions = []
        with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
            for model_idx, (model, weight) in enumerate(
                zip(self.models, self.weights), 1
            ):
                model_preds = []
                futures = []

                for i in range(n_batches):
                    start_idx = i * self.batch_size
                    end_idx = min((i + 1) * self.batch_size, n_samples)
                    batch = X[start_idx:end_idx]
                    future = executor.submit(self._predict_batch, model, batch)
                    futures.append(future)

                for future in futures:
                    batch_pred = future.result()
                    model_preds.append(batch_pred)

                model_preds = np.concatenate(model_preds) * weight
                predictions.append(model_preds)

                if self.verbose:
                    logger.info(f"Modelo {model_idx}/{self.n_models} procesado")

        # Votación ponderada final
        ensemble_pred = np.sum(predictions, axis=0) > 0.5
        return ensemble_pred
