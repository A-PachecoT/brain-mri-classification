import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# ========================
# Funciones de Carga
# ========================


def load_single_image(args):
    """
    Cargar y preprocesar una única imagen.

    Args:
        args (tuple): (img_path, img_size, label)

    Returns:
        tuple: (image_array, label)
    """
    img_path, img_size, label = args
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        return img_array, label
    except Exception as e:
        print(f"Error al cargar la imagen {img_path}: {e}")
        return None, None


def load_data(data_dir="images", img_size=(128, 128)):
    """
    Cargar y preprocesar imágenes de resonancia magnética usando procesamiento paralelo.
    """
    # Preparar rutas de imágenes y etiquetas
    image_data = []

    # Recolectar rutas para imágenes positivas (con tumor)
    yes_path = os.path.join(data_dir, "yes")
    for img_name in os.listdir(yes_path):
        image_data.append((os.path.join(yes_path, img_name), img_size, 1))

    # Recolectar rutas para imágenes negativas (sin tumor)
    no_path = os.path.join(data_dir, "no")
    for img_name in os.listdir(no_path):
        image_data.append((os.path.join(no_path, img_name), img_size, 0))

    # Usar procesamiento paralelo para cargar imágenes
    num_workers = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(load_single_image, image_data))

    # Filtrar valores None y separar imágenes y etiquetas
    valid_results = [(img, label) for img, label in results if img is not None]
    images, labels = zip(*valid_results)

    X = np.array(images)
    y = np.array(labels)

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# ========================
# Creación de Dataset
# ========================


def create_dataset(X, y, batch_size=32, is_training=True):
    """
    Crear un dataset de TensorFlow optimizado con procesamiento paralelo.

    Args:
        X (np.ndarray): Datos de imágenes
        y (np.ndarray): Etiquetas
        batch_size (int): Tamaño del lote
        is_training (bool): Si es un dataset de entrenamiento

    Returns:
        tf.data.Dataset: Dataset de TensorFlow optimizado
    """
    # Calcular el número de llamadas paralelas
    AUTOTUNE = tf.data.AUTOTUNE

    # Crear el dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if is_training:
        # Cachear el dataset en memoria para mejor rendimiento
        dataset = dataset.cache()

        # Mezclar con un buffer lo suficientemente grande para asegurar buena aleatorización
        dataset = dataset.shuffle(buffer_size=1000)

    # Configurar procesamiento por lotes en paralelo
    dataset = dataset.batch(batch_size)

    # Precargar siguiente lote mientras se procesa el actual
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
