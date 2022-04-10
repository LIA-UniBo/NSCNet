import os
import tensorflow as tf

SAVE_WEIGHTS = True
LOAD_WEIGHTS = True
WEIGHTS_PATH = os.path.join("architectures", "weights", "vae")

STRIDE = 2
KERNEL_SIZE = 3
PADDING = "same"
STARTING_FILTERS = 16
LATENT_DIM = 128
N_CONV_LAYERS = 5

LEARNING_RATE = 1e-3
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)
BATCH_SIZE = 32
EPOCHS = 1
