import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, metrics

from images_loader import import_image_np_dataset
import clustering

import vae_config as config


class Encoder(tf.keras.Model):

    def __init__(self, stride, kernel_size, padding, starting_filters, latent_dim, n_conv_layers, activation="relu"):
        super(Encoder, self).__init__()

        self.conv_layers = [layers.Conv2D(starting_filters * 2**(i-1),
                                          kernel_size,
                                          strides=stride,
                                          padding=padding,
                                          data_format="channels_last",
                                          activation=activation,
                                          name=f'encoder_conv_{i}')
                            for i in range(1, n_conv_layers + 1)]

        self.flatten_layer = layers.Flatten()

        self.mean_dense_layer = layers.Dense(latent_dim, activation=activation, name="mean")
        self.std_dense_layer = layers.Dense(latent_dim, activation=activation, name="std")

    def call(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        compressed_shape = x.shape

        x = self.flatten_layer(x)

        mean_x = self.mean_dense_layer(x)
        log_var_x = self.std_dense_layer(x)

        return [mean_x, log_var_x, compressed_shape]


class Decoder(tf.keras.Model):

    def __init__(self, stride, kernel_size, padding, starting_filters, n_conv_layers, activation="relu"):

        super(Decoder, self).__init__()

        self.activation = activation

        self.conv_transpose_layers = [layers.Conv2DTranspose(starting_filters * 2**(i-1),
                                                             kernel_size,
                                                             strides=stride,
                                                             padding=padding,
                                                             data_format="channels_last",
                                                             activation=activation,
                                                             name=f'decoder_deconv_{i}')
                                      for i in range(n_conv_layers, 0, -1)]

        self.output_layer = layers.Conv2DTranspose(3,
                                                   kernel_size,
                                                   padding="same",
                                                   data_format="channels_last",
                                                   activation="sigmoid")  # Why sigmoid?

        self.dense_layer = None
        self.reshape_layer = None

    def call(self, inputs):

        assert len(inputs) == 2

        x = inputs[0]
        compressed_shape = inputs[1]

        if self.dense_layer is None:
            self.dense_layer = layers.Dense(compressed_shape[1] * compressed_shape[2] * compressed_shape[3],
                                            activation=self.activation)
            self.reshape_layer = layers.Reshape((compressed_shape[1], compressed_shape[2], compressed_shape[3]))

        x = self.dense_layer(x)
        x = self.reshape_layer(x)

        for conv_transpose_layer in self.conv_transpose_layers:
            x = conv_transpose_layer(x)

        x = self.output_layer(x)  # Note: this layer is used often in examples, is it necessary?

        return x


class ConvolutionalVAE(tf.keras.Model):

    def __init__(self, stride, kernel_size, padding, starting_filters, latent_dim, n_conv_layers, activation="relu"):
        super(ConvolutionalVAE, self).__init__()

        self.encoder = Encoder(stride, kernel_size, padding, starting_filters, latent_dim, n_conv_layers, activation)
        self.decoder = Decoder(stride, kernel_size, padding, starting_filters, n_conv_layers, activation)

        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    def call(self, x):
        mean_x, log_var_x, compressed_shape = self.encode(x)
        z = self.sample(mean_x, log_var_x)
        decoded_x = self.decode(z, compressed_shape)

        return decoded_x

    def encode(self, x):
        encoded_x = self.encoder(x)

        mean_x = encoded_x[0]
        log_var_x = encoded_x[1]
        compressed_shape = encoded_x[2]

        return mean_x, log_var_x, compressed_shape

    def decode(self, z, compressed_shape):
        decoded_x = self.decoder([z, compressed_shape])
        return decoded_x

    def sample(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(log_var * 0.5) + mean

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, compressed_shape = self.encode(data)
            z = self.sample(z_mean, z_log_var)
            decoded_x = self.decode(z, compressed_shape)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, decoded_x), axis=(1, 2)))

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()}


def build_model(input_shape, stride, kernel_size, padding, starting_filters, latent_dim, n_conv_layers, optimizer):
    input = tf.keras.Input(shape=input_shape)
    vae = ConvolutionalVAE(stride, kernel_size, padding, starting_filters, latent_dim, n_conv_layers)
    vae(input)
    vae.compile(optimizer=optimizer)

    return vae


def train_model(model, data, epochs, batch_size):
    history = model.fit(data,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)
    return history


def clusterize(vae, samples, cluster_method, cluster_args):
    # TODO: predict in batches

    if cluster_method not in clustering.CLUSTERING_METHODS:
        raise Exception("cluster method must be one between " + ",".join(clustering.CLUSTERING_METHODS))

    z_mean, z_log_var, compressed_shape = vae.encode(samples)
    features = vae.sample(z_mean, z_log_var)

    clustering_output = None
    if cluster_method == "kmeans":
        clustering_output = clustering.k_means(features, **cluster_args)
    elif cluster_method == "dbscan":
        clustering_output = clustering.dbscan(features, **cluster_args)

    return clustering_output["labels"]


# -----------------------------------
# Test
inputs = import_image_np_dataset(config.IMAGES_PATH, (config.INPUT_SHAPE[0], config.INPUT_SHAPE[1]),
                                 config.RGB_NORMALIZATION)

cluster_args = {
    "n_clusters": config.N_CLUSTERS
}

model = build_model(config.INPUT_SHAPE,
                    config.STRIDE,
                    config.KERNEL_SIZE,
                    config.PADDING,
                    config.STARTING_FILTERS,
                    config.LATENT_DIM,
                    config.N_CONV_LAYERS,
                    config.OPTIMIZER)

model.summary(expand_nested=True)


print("Training launched...")
history = train_model(model, inputs, config.EPOCHS, config.BATCH_SIZE)
print("Training completed!")

print("Clustering launched...")
clusters = clusterize(model, inputs, config.CLUSTERING_METHOD, cluster_args)
print("Clustering completed!")
