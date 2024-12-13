import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .model import create_model


class ParallelEnsemble:
    def __init__(self, n_models=3):
        self.n_models = n_models
        self.models = []
        self.strategy = tf.distribute.MirroredStrategy()

    def train_model_parallel(self, model_id, train_dataset, val_dataset):
        """Entrena un modelo individual del ensemble"""
        with self.strategy.scope():
            model = create_model()
            # Usar bootstrap sampling para bagging
            bootstrap_dataset = train_dataset.shuffle(1000).take(
                train_dataset.cardinality()
            )
            model.fit(bootstrap_dataset, validation_data=val_dataset)
            return model

    def train(self, train_dataset, val_dataset):
        """Entrena múltiples modelos en paralelo"""
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(self.n_models):
                future = executor.submit(
                    self.train_model_parallel, i, train_dataset, val_dataset
                )
                futures.append(future)

            self.models = [future.result() for future in futures]

    def predict(self, X):
        """Predicción por votación mayoritaria"""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Votación mayoritaria
        ensemble_pred = np.mean(predictions, axis=0) > 0.5
        return ensemble_pred
