import tensorflow as tf
import numpy as np
import matrix_manipulation

def import_image_tf_dataset(path, batch_size, input_shape, shuffle=True):

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

    tf_dataset = import_image_tf_dataset(path, 1000000, input_shape, shuffle=False)

    np_dataset = None
    for batch in tf_dataset:
        np_dataset = batch.numpy()

    np_dataset = np.squeeze(np_dataset)

    if normalize:
        np_dataset = matrix_manipulation.rgb_normalize(np_dataset)

    return np_dataset
