import os
import tensorflow as tf

# Model weights control
SAVE_WEIGHTS = False
LOAD_WEIGHTS = True
WEIGHTS_PATH = os.path.join("architectures", "weights", "vade")

# Encoder/Decoder convolution and dense parameters
STRIDE = 2
KERNEL_SIZE = 3
PADDING = "same"
STARTING_FILTERS = 32

DENSE_UNITS = 1024  # Number of the first dense layer applied after the last convolutional one
LATENT_DIM = 64  # The latent space dimension
N_CONV_LAYERS = 5  # Number of convolutional layers that will be applied in both VAE sides (e.g., Encoder and Decoder)

# Model parameters
LEARNING_RATE = 1e-3
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE)
BATCH_SIZE = 32
EPOCHS = 10
PRETRAIN_EPOCHS = 0
