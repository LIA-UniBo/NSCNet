# This module contains constants

import tensorflow as tf
import os

from tensorflow_addons.optimizers import extend_with_decoupled_weight_decay

from architectures.common import spec_augmentation

SAVE_WEIGHTS = True
LOAD_WEIGHTS = True
WEIGHTS_PATH = os.path.join("architectures","weights","nscnet")

POST_PROCESSING_OPTIONS = {
    "normalize": True,
    "pca": 128,
    "whiten": True,
    "l2 normalize": True
}

SPEC_AUGMENTATION_OPTIONS = {
    "apply": True,
    "policy": spec_augmentation.POLICIES["Custom"]
}

EARLY_STOPPING_OPTIONS = {
    "apply": True,
    "min_delta": 0.005,
    "patience": 10
}

POOLING = "max"
DIM_REPRESENTATION = 512


LEARNING_RATE = 1e-4
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)
# OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
BATCH_SIZE = 32
EPOCHS = 100
BATCHES_PER_EPOCH = 1

USE_ARCFACE_LOSS = False

# customSGD = extend_with_decoupled_weight_decay(tf.keras.optimizers.SGD)
# OPTIMIZER = customSGD(weight_decay=1e-4, learning_rate=0.05)

ACTIVATION = tf.keras.layers.LeakyReLU(alpha=0.01)
