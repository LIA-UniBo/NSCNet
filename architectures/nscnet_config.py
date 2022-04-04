#This module contains constants

import tensorflow as tf
import spec_augmentation

POST_PROCESSING_OPTIONS = {
    "normalize": True,
    "pca": 128,
    "whiten": True,
    "l2 normalize": True
}

RGB_NORMALIZATION = True

SPEC_AUGMENTATION_OPTIONS = {
    "apply": True,
    "policy": spec_augmentation.POLICIES["LB"]
}

EARLY_STOPPING_OPTIONS = {
    "apply": True,
    "min_delta": 0.005,
    "patience": 10
}

IMAGES_PATH = "Samples"

POOLING = "max"
DIM_REPRESENTATION = 512
INPUT_SHAPE = (240, 320, 3)
N_CLUSTERS = 20  # Test value
CLUSTERING_METHOD = "kmeans"

LEARNING_RATE = 1e-3
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
BATCH_SIZE = 16
EPOCHS = 50

USE_ARCFACE_LOSS = False
