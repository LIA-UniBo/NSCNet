# This module contains constants

import tensorflow as tf
import os
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


LEARNING_RATE = 2*1e-3
#OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9)
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
BATCH_SIZE = 32
EPOCHS = 100
BATCHES_PER_EPOCH = 1

USE_ARCFACE_LOSS = False
