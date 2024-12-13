import os
import tensorflow as tf
import logging
from data.data_loader import load_data, create_dataset
from models.model import create_model
from utils.train import train_model, plot_training_history
from utils.evaluate import evaluate_model, plot_confusion_matrix
from models.ensemble import ParallelEnsemble

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MRI-Classification")

# Suprimir warnings de TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    # ========================
    # Configuración General
    # ========================
    EPOCHS = 3
    BATCH_SIZE = 32

    logger.info("=" * 50)
    logger.info("INICIANDO CLASIFICACIÓN DE RESONANCIAS MAGNÉTICAS")
    logger.info("=" * 50)

    # Establecer semillas aleatorias
    tf.random.set_seed(42)

    # ========================
    # Configuración de Hardware
    # ========================
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"✓ {len(gpus)} GPU(s) detectada(s)")
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.info("✗ No se detectaron GPUs, usando CPU")

    # ========================
    # Carga de Datos
    # ========================
    logger.info("\n1. PREPARACIÓN DE DATOS")
    logger.info("-" * 30)

    logger.info("Cargando imágenes...")
    X_train, X_test, y_train, y_test = load_data(data_dir="data/raw")

    logger.info("Creando datasets optimizados...")
    train_dataset = create_dataset(X_train, y_train, is_training=True)
    test_dataset = create_dataset(X_test, y_test, is_training=False)
    logger.info("✓ Datos preparados correctamente")

    # ========================
    # Modelo Base
    # ========================
    logger.info("\n2. ENTRENAMIENTO MODELO BASE")
    logger.info("-" * 30)

    logger.info("Inicializando modelo...")
    model = create_model()
    model.summary()

    logger.info("\nEntrenando modelo base...")
    history = train_model(
        model,
        train_dataset,
        test_dataset,
        checkpoint_dir="artifacts/models",
        epochs=EPOCHS,
    )

    logger.info("Generando visualizaciones...")
    plot_training_history(history, save_path="artifacts/plots/training_history.png")

    logger.info("\nEvaluando modelo base...")
    results = evaluate_model(model, test_dataset)
    plot_confusion_matrix(
        results["y_true"],
        results["y_pred"],
        save_path="artifacts/plots/confusion_matrix.png",
    )

    # ========================
    # Ensemble Learning
    # ========================
    logger.info("\n3. ENTRENAMIENTO ENSEMBLE")
    logger.info("-" * 30)

    logger.info("Inicializando ensemble...")
    ensemble = ParallelEnsemble(n_models=3, verbose=1)

    logger.info("Entrenando modelos del ensemble...")
    ensemble.train(train_dataset, test_dataset, epochs=EPOCHS)

    # Evaluar ensemble
    logger.info("\nEvaluando ensemble...")
    y_pred_ensemble, metrics = ensemble.predict(X_test, y_test)
    plot_confusion_matrix(
        y_test,
        y_pred_ensemble,
        save_path="artifacts/plots/ensemble_confusion_matrix.png",
    )

    # ========================
    # Resumen Final
    # ========================
    logger.info("\nRESUMEN DE ENTRENAMIENTO")
    logger.info("=" * 30)
    logger.info(f"Precisión modelo base: {results['accuracy']:.4f}")
    logger.info(f"Precisión ensemble: {metrics['accuracy']:.4f}")
    logger.info(f"Precision ensemble: {metrics['precision']:.4f}")
    logger.info(f"Recall ensemble: {metrics['recall']:.4f}")
    logger.info("\nPesos de los modelos:")
    for i, weight in enumerate(ensemble.weights, 1):
        logger.info(f"  Modelo {i}: {weight:.4f}")

    logger.info("\n✓ Entrenamiento completado!")
    logger.info("  Revisa el directorio 'artifacts/plots' para las visualizaciones")


if __name__ == "__main__":
    main()
