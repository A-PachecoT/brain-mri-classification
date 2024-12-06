import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


def load_single_image(args):
    """
    Load and preprocess a single image.

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
        print(f"Error loading image {img_path}: {e}")
        return None, None


def load_data(data_dir="images", img_size=(128, 128)):
    """
    Load and preprocess MRI images using parallel processing.
    """
    # Prepare image paths and labels
    image_data = []

    # Collect paths for yes (tumor) images
    yes_path = os.path.join(data_dir, "yes")
    for img_name in os.listdir(yes_path):
        image_data.append((os.path.join(yes_path, img_name), img_size, 1))

    # Collect paths for no (no tumor) images
    no_path = os.path.join(data_dir, "no")
    for img_name in os.listdir(no_path):
        image_data.append((os.path.join(no_path, img_name), img_size, 0))

    # Use parallel processing to load images
    num_workers = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(load_single_image, image_data))

    # Filter out None values and separate images and labels
    valid_results = [(img, label) for img, label in results if img is not None]
    images, labels = zip(*valid_results)

    X = np.array(images)
    y = np.array(labels)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def create_dataset(X, y, batch_size=32, is_training=True):
    """
    Create an optimized TensorFlow dataset with parallel processing.

    Args:
        X (np.ndarray): Image data
        y (np.ndarray): Labels
        batch_size (int): Batch size
        is_training (bool): Whether this is a training dataset

    Returns:
        tf.data.Dataset: Optimized TensorFlow dataset
    """
    # Calculate the number of parallel calls
    AUTOTUNE = tf.data.AUTOTUNE

    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if is_training:
        # Cache the dataset in memory for better performance
        dataset = dataset.cache()

        # Shuffle with a buffer size large enough to ensure good randomization
        dataset = dataset.shuffle(buffer_size=1000)

    # Set up parallel batching
    dataset = dataset.batch(batch_size)

    # Prefetch next batch while current batch is being processed
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
