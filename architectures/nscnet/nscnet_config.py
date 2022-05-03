# This module contains constants

import tensorflow as tf
import os

from architectures.common import spec_augmentation

# Model weights control
SAVE_WEIGHTS = True
LOAD_WEIGHTS = True
WEIGHTS_PATH = os.path.join("architectures", "weights", "nscnet")

# Operations computed before executing the clustering algorithm
POST_PROCESSING_OPTIONS = {
    "normalize": False,
    "pca": 256,
    "whiten": False,
    "l2 normalize": True
}

# Data augmentation
SPEC_AUGMENTATION_OPTIONS = {
    "apply": True,
    "policy": spec_augmentation.POLICIES["Custom"]
}

# Early stopping configuration
EARLY_STOPPING_OPTIONS = {
    "apply": True,
    "min_delta": 0.005,
    "patience": 10
}


POOLING = "max"  # Pooling operation used by the NSCNet's backbone (the EfficientNet-B0)
DIM_REPRESENTATION = 512  # Output dimension after the last NSCNet's backbone (the EfficientNet-B0)

# Model parameters
LEARNING_RATE = 1e-4
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
BATCH_SIZE = 32
EPOCHS = 100
BATCHES_PER_EPOCH = 1
ACTIVATION = tf.keras.layers.LeakyReLU(alpha=0.01)
USE_ARCFACE_LOSS = False

