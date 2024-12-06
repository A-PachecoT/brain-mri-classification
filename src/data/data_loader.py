import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_data(data_dir="images", img_size=(128, 128)):
    """
    Load and preprocess MRI images from the specified directory.

    Args:
        data_dir (str): Directory containing 'yes' and 'no' subdirectories
        img_size (tuple): Target size for the images

    Returns:
        X_train, X_test, y_train, y_test: Train and test splits of data
    """
    images = []
    labels = []

    # Load images with tumor (yes)
    yes_path = os.path.join(data_dir, "yes")
    for img_name in os.listdir(yes_path):
        img_path = os.path.join(yes_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(1)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    # Load images without tumor (no)
    no_path = os.path.join(data_dir, "no")
    for img_name in os.listdir(no_path):
        img_path = os.path.join(no_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(0)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    X = np.array(images)
    y = np.array(labels)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def create_dataset(X, y, batch_size=32):
    """
    Create a TensorFlow dataset with batching and shuffling.

    Args:
        X (np.ndarray): Image data
        y (np.ndarray): Labels
        batch_size (int): Batch size

    Returns:
        tf.data.Dataset: TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(1000).batch(batch_size)
    return dataset
