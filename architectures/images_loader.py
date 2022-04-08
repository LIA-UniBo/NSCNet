# This module manages the loading of the images to store them as numpy arrays or datasets

import tensorflow as tf
import numpy as np

from architectures import matrix_manipulation


def import_image_tf_dataset(path, batch_size, input_shape, shuffle=True):

    # Load all the images in a path and create a tensor dataset

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels=None,
        label_mode=None,
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=input_shape,
        shuffle=shuffle,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=True,
        crop_to_aspect_ratio=False)

    return dataset


def import_image_np_dataset(path, input_shape, normalize):

    # Load all the images into a tensor dataset with only one batch (first dim=1)
    tf_dataset = import_image_tf_dataset(path, 1000000, input_shape, shuffle=False)

    # Transform tf.Dataset into a numpy array
    np_dataset = None
    for batch in tf_dataset:
        np_dataset = batch.numpy()

    # Squeeze to remove first dimension
    np_dataset = np.squeeze(np_dataset)

    # Convert RGB values from [0,255] to [0.0,1.0]
    if normalize:
        np_dataset = matrix_manipulation.rgb_normalize(np_dataset)

    # np_dataset.shape = (N_samples, Height, Width, Channels)

    return np_dataset
