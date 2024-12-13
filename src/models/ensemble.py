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

    def _train_model(self, model_config):
        """Entrena un modelo individual del ensemble con configuración específica"""
        with self.strategy.scope():
            model = create_model()

            # Aplicar configuración específica del modelo
            learning_rate = model_config.get("learning_rate", 0.001)
            dropout_rate = model_config.get("dropout_rate", 0.5)
            l1_reg = model_config.get("l1_reg", 0.0)
            l2_reg = model_config.get("l2_reg", 0.0)

            # Aplicar regularización a todas las capas Dense
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    layer.kernel_regularizer = tf.keras.regularizers.L1L2(
                        l1=l1_reg, l2=l2_reg
                    )

            # Configurar optimizador con learning rate específico
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, clipnorm=1.0  # Añadir gradient clipping
            )

            # Compilar con diferentes métricas y pérdidas
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
                metrics=[
                    "accuracy",
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                ],
            )

            return model

    @tf.function(reduce_retracing=True)
    def _predict_tf_function(self, model, batch):
        """Función TF optimizada para predicción"""
        return model(batch, training=False)

    def train(self, train_dataset, val_dataset, epochs=50):
        """Entrena múltiples modelos en paralelo con configuraciones diversas"""
        logger.info("Iniciando entrenamiento del ensemble...")

        # Configuraciones diversas para cada modelo con regularización
        model_configs = [
            {
                "learning_rate": 0.001,
                "dropout_rate": 0.3,
                "bootstrap_size": 0.8,
                "l1_reg": 1e-5,
                "l2_reg": 1e-4,
            },
            {
                "learning_rate": 0.0005,
                "dropout_rate": 0.4,
                "bootstrap_size": 0.9,
                "l1_reg": 1e-6,
                "l2_reg": 1e-3,
            },
            {
                "learning_rate": 0.00025,
                "dropout_rate": 0.5,
                "bootstrap_size": 1.0,
                "l1_reg": 1e-4,
                "l2_reg": 1e-5,
            },
        ]

        def train_single_model(config, model_idx):
            """Función auxiliar para entrenar un modelo individual"""
            model = self._train_model(config)

            # Configurar dataset
            AUTOTUNE = tf.data.AUTOTUNE
            dataset_size = int(train_dataset.cardinality().numpy())

            # Crear bootstrap dataset
            bootstrap_size = config["bootstrap_size"]
            n_samples = int(dataset_size * bootstrap_size)

            bootstrap_dataset = (
                train_dataset.shuffle(1000, seed=model_idx)
                .take(n_samples)
                .prefetch(AUTOTUNE)
            )

            # Callbacks específicos por modelo
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=10,
                    restore_best_weights=True,
                    verbose=0,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_accuracy",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=0,
                ),
            ]

            # Entrenar modelo
            history = model.fit(
                bootstrap_dataset,
                validation_data=val_dataset,
                callbacks=callbacks,
                epochs=epochs,
                workers=self.cpu_count if not self.gpus else 1,
                use_multiprocessing=False,
                verbose=self.verbose,
            )

            # Calcular métricas
            val_accuracy = max(history.history["val_accuracy"])
            val_auc = max(history.history["val_auc"])
            val_precision = max(history.history["val_precision"])
            val_recall = max(history.history["val_recall"])

            combined_score = (val_accuracy + val_auc + val_precision + val_recall) / 4

            return (
                model,
                combined_score,
                (val_accuracy, val_auc, val_precision, val_recall),
            )

        # Entrenar modelos en paralelo usando ProcessPoolExecutor
        with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
            futures = [
                executor.submit(train_single_model, config, idx)
                for idx, config in enumerate(model_configs[: self.n_models], 1)
            ]

            results = []
            for future in futures:
                model, score, metrics = future.result()
                results.append((model, score, metrics))

            # Guardar modelos y pesos
            self.models = [r[0] for r in results]
            self.weights = np.array([r[1] for r in results])

            # Normalizar pesos usando softmax
            self.weights = np.exp(self.weights) / np.sum(np.exp(self.weights))

            # Logging de resultados
            for i, (_, _, (acc, auc, prec, rec)) in enumerate(results, 1):
                logger.info(
                    f"Modelo {i} - Métricas:\n"
                    f"  Accuracy: {acc:.4f}\n"
                    f"  AUC: {auc:.4f}\n"
                    f"  Precision: {prec:.4f}\n"
                    f"  Recall: {rec:.4f}\n"
                    f"  Score Combinado: {self.weights[i-1]:.4f}"
                )

    def predict(self, X, y=None):
        """Predicción con votación ponderada suave"""
        logger.info("Realizando predicciones del ensemble...")

        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        # Almacenar probabilidades en lugar de predicciones binarias
        predictions = []

        with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
            for model_idx, (model, weight) in enumerate(
                zip(self.models, self.weights), 1
            ):
                model_probs = []
                futures = []

                for i in range(n_batches):
                    start_idx = i * self.batch_size
                    end_idx = min((i + 1) * self.batch_size, n_samples)
                    batch = X[start_idx:end_idx]
                    future = executor.submit(self._predict_batch, model, batch)
                    futures.append(future)

                for future in futures:
                    batch_probs = future.result()
                    model_probs.append(batch_probs)

                # Concatenar y aplicar peso
                model_probs = np.concatenate(model_probs) * weight
                predictions.append(model_probs)

                if self.verbose:
                    logger.info(f"Modelo {model_idx}/{self.n_models} procesado")

        # Promedio ponderado de probabilidades
        ensemble_probs = np.sum(predictions, axis=0)

        # Decisión final con umbral optimizado (0.4 en lugar de 0.5)
        ensemble_pred = ensemble_probs > 0.4

        if y is not None:
            accuracy = np.mean(ensemble_pred == y)
            # Calcular métricas adicionales
            precision = np.mean(ensemble_pred[y == 1])
            recall = np.mean(y[ensemble_pred == 1])
            f1 = 2 * (precision * recall) / (precision + recall)

            logger.info(f"\nMétricas del ensemble:")
            logger.info(f"Precisión: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")

            return ensemble_pred, accuracy

        return ensemble_pred
