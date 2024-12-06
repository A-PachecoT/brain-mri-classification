import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import h5py
import json
from pathlib import Path
import shutil

# ========================
# Repositorio de Almacenamiento en Disco (RAD)
# ========================


class DiskStorageRepository:
    """
    Implementación de un Repositorio de Almacenamiento en Disco (RAD) para gestionar
    grandes volúmenes de datos de imágenes médicas de manera eficiente.
    """

    def __init__(self, base_path="data/processed"):
        self.base_path = Path(base_path)
        self.metadata_file = self.base_path / "metadata.json"
        self.data_file = self.base_path / "image_data.h5"
        self.initialize_storage()

    def initialize_storage(self):
        """Inicializar la estructura del almacenamiento."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        if not self.metadata_file.exists():
            self._save_metadata({"num_samples": 0, "indices": {}})

    def _save_metadata(self, metadata):
        """Guardar metadatos en disco."""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f)

    def _load_metadata(self):
        """Cargar metadatos desde disco."""
        with open(self.metadata_file, "r") as f:
            return json.load(f)

    def store_batch(self, images, labels, batch_indices):
        """
        Almacenar un lote de imágenes y etiquetas en el RAD.

        Args:
            images: Array de imágenes numpy
            labels: Array de etiquetas
            batch_indices: Lista de índices para las imágenes
        """
        metadata = self._load_metadata()
        current_size = 0

        with h5py.File(self.data_file, "a") as f:
            if "images" not in f:
                f.create_dataset(
                    "images", data=images, maxshape=(None, *images.shape[1:])
                )
                f.create_dataset("labels", data=labels, maxshape=(None,))
            else:
                current_size = f["images"].shape[0]
                new_size = current_size + len(images)
                f["images"].resize(new_size, axis=0)
                f["labels"].resize(new_size, axis=0)
                f["images"][current_size:] = images
                f["labels"][current_size:] = labels

            for idx, orig_idx in enumerate(batch_indices):
                metadata["indices"][str(orig_idx)] = current_size + idx

        metadata["num_samples"] = len(metadata["indices"])
        self._save_metadata(metadata)

    def get_batch(self, indices):
        """
        Recuperar un lote de imágenes y etiquetas del RAD.

        Args:
            indices: Lista de índices a recuperar

        Returns:
            tuple: (imágenes, etiquetas)
        """
        metadata = self._load_metadata()
        storage_indices = [metadata["indices"][str(i)] for i in indices]

        with h5py.File(self.data_file, "r") as f:
            images = f["images"][storage_indices]
            labels = f["labels"][storage_indices]

        return images, labels

    def clear_storage(self):
        """Limpiar todo el almacenamiento en disco."""
        if self.data_file.exists():
            self.data_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        self.initialize_storage()


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


def load_data(data_dir="images", img_size=(128, 128), batch_size=32):
    """
    Cargar y preprocesar imágenes de resonancia magnética usando procesamiento paralelo
    y almacenamiento en disco.
    """
    # Inicializar el RAD
    rad = DiskStorageRepository()
    rad.clear_storage()  # Limpiar almacenamiento anterior

    # Preparar rutas de imágenes y etiquetas
    image_data = []

    # Verificar que el directorio existe
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"El directorio {data_dir} no existe")

    # Recolectar rutas para imágenes positivas (con tumor)
    yes_path = os.path.join(data_dir, "yes")
    if os.path.exists(yes_path):
        for img_name in os.listdir(yes_path):
            image_data.append((os.path.join(yes_path, img_name), img_size, 1))

    # Recolectar rutas para imágenes negativas (sin tumor)
    no_path = os.path.join(data_dir, "no")
    if os.path.exists(no_path):
        for img_name in os.listdir(no_path):
            image_data.append((os.path.join(no_path, img_name), img_size, 0))

    if not image_data:
        raise ValueError(f"No se encontraron imágenes en {data_dir}")

    # Procesar y almacenar imágenes en lotes usando RAD
    num_workers = multiprocessing.cpu_count()

    for i in range(0, len(image_data), batch_size):
        batch_data = image_data[i : i + batch_size]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(load_single_image, batch_data))

        # Filtrar resultados válidos
        valid_results = [(img, label) for img, label in results if img is not None]
        if valid_results:
            batch_images, batch_labels = zip(*valid_results)
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            batch_indices = range(i, i + len(valid_results))
            rad.store_batch(batch_images, batch_labels, batch_indices)

    # Cargar todos los datos del RAD
    metadata = rad._load_metadata()
    all_indices = list(range(metadata["num_samples"]))

    if not all_indices:
        raise ValueError("No se pudieron cargar imágenes válidas")

    X, y = rad.get_batch(all_indices)

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
