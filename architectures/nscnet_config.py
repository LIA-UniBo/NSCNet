# This module contains constants

import tensorflow as tf
from architectures import spec_augmentation

POST_PROCESSING_OPTIONS = {
    "normalize": True,
    "pca": 128,
    "whiten": True,
    "l2 normalize": True
}

SPEC_AUGMENTATION_OPTIONS = {
    "apply": True,
    "policy": spec_augmentation.POLICIES["LB"]
}

EARLY_STOPPING_OPTIONS = {
    "apply": True,
    "min_delta": 0.005,
    "patience": 10
}

POOLING = "max"
DIM_REPRESENTATION = 512


LEARNING_RATE = 1e-3
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
BATCH_SIZE = 16
EPOCHS = 1

USE_ARCFACE_LOSS = False

# RGB_NORMALIZATION = True
# N_CLUSTERS = 20  # Test value
# CLUSTERING_METHOD = "kmeans"
# IMAGES_PATH = "data"
# INPUT_SHAPE = (64, 512, 3)