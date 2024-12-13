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
        self.model_dir = "artifacts/models/ensemble"
        os.makedirs(self.model_dir, exist_ok=True)

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

    def save_models(self):
        """Guarda los modelos del ensemble y sus pesos"""
        logger.info("Guardando modelos del ensemble...")

        # Guardar pesos del ensemble
        weights_path = os.path.join(self.model_dir, "ensemble_weights.npy")
        np.save(weights_path, self.weights)

        # Guardar cada modelo
        for i, model in enumerate(self.models):
            model_path = os.path.join(self.model_dir, f"model_{i+1}")
            model.save(model_path)

        logger.info(f"✓ {len(self.models)} modelos guardados en {self.model_dir}")

    def load_models(self):
        """Carga los modelos del ensemble y sus pesos"""
        logger.info("Cargando modelos del ensemble...")

        try:
            # Cargar pesos del ensemble
            weights_path = os.path.join(self.model_dir, "ensemble_weights.npy")
            self.weights = np.load(weights_path)

            # Cargar modelos
            self.models = []
            i = 1
            while True:
                model_path = os.path.join(self.model_dir, f"model_{i}")
                if not os.path.exists(model_path):
                    break
                model = tf.keras.models.load_model(model_path)
                self.models.append(model)
                i += 1

            logger.info(f"✓ {len(self.models)} modelos cargados")
            return True

        except Exception as e:
            logger.error(f"Error cargando modelos: {str(e)}")
            return False

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

        # Guardar modelos al finalizar entrenamiento
        self.save_models()

    def predict(self, X, y):
        """
        Realiza predicciones usando todos los modelos del ensemble
        """
        logger.info("Realizando predicciones del ensemble...")

        try:
            # Lista para almacenar predicciones de cada modelo
            predictions = []

            # Realizar predicciones con cada modelo
            for i, model in enumerate(self.models, 1):
                pred = model.predict(X)
                # Debug info
                logger.info(f"Shape de predicción modelo {i}: {pred.shape}")
                pred = pred.squeeze()
                logger.info(f"Shape después de squeeze: {pred.shape}")
                predictions.append(pred)
                logger.info(f"Modelo {i}/{len(self.models)} procesado")

            # Debug info
            predictions = np.array(predictions)
            logger.info(f"Shape final de predictions: {predictions.shape}")
            logger.info(f"Shape de y: {y.shape}")

            weighted_predictions = np.average(predictions, axis=0, weights=self.weights)
            ensemble_pred = (weighted_predictions > 0.5).astype(int)

            # Calcular métricas de forma más segura
            accuracy = np.mean(ensemble_pred == y)

            true_pos = np.sum((ensemble_pred == 1) & (y == 1))
            pred_pos = np.sum(ensemble_pred == 1)
            actual_pos = np.sum(y == 1)

            precision = true_pos / (pred_pos + 1e-10)
            recall = true_pos / (actual_pos + 1e-10)

            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
            }

            return ensemble_pred, metrics

        except Exception as e:
            logger.error(f"Error en predict: {str(e)}")
            logger.error(f"Tipo de error: {type(e)}")
            raise
