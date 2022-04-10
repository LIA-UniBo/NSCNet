import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, metrics
import os

from architectures.common import clustering
import architectures.vaenet.vaenet_config as config
import matplotlib.pyplot as plt


class Encoder(tf.keras.Model):

    def __init__(self, stride, kernel_size, padding, starting_filters, latent_dim, n_conv_layers, activation="relu"):
        super(Encoder, self).__init__()

        self.conv_layers = [layers.Conv2D(starting_filters * 2 ** (i - 1),
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

    def __init__(self, stride, kernel_size, padding, starting_filters, n_conv_layers, input_shape, activation="relu"):

        super(Decoder, self).__init__()

        self.activation = activation

        self.conv_transpose_layers = [layers.Conv2DTranspose(starting_filters * 2 ** (i - 1),
                                                             kernel_size,
                                                             strides=stride,
                                                             padding=padding,
                                                             data_format="channels_last",
                                                             activation=activation,
                                                             name=f'decoder_deconv_{i}')
                                      for i in range(n_conv_layers, 0, -1)]

        self.output_layer = layers.Conv2DTranspose(input_shape[2],
                                                   kernel_size,
                                                   padding="same",
                                                   data_format="channels_last",
                                                   activation="sigmoid")

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

        x = self.output_layer(x)

        return x


class ConvolutionalVAE(tf.keras.Model):

    def __init__(self,
                 stride,
                 kernel_size,
                 padding,
                 starting_filters,
                 latent_dim,
                 n_conv_layers,
                 input_shape,
                 activation="relu"):
        super(ConvolutionalVAE, self).__init__()

        self.encoder = Encoder(stride, kernel_size, padding, starting_filters, latent_dim, n_conv_layers, activation)
        self.decoder = Decoder(stride, kernel_size, padding, starting_filters, n_conv_layers, input_shape, activation)

        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    def call(self, x):
        mean_x, log_var_x, compressed_shape = self.encode(x)

        # Reparametrization trick
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
            "kl_loss": self.kl_loss_tracker.result()
        }


class VAENet:

    def __init__(self, input_shape, cluster_dic):

        self.model = self.build_model(input_shape)
        self.n_clusters = cluster_dic['n_clusters']
        self.cluster_args = cluster_dic['config']
        self.cluster_method = cluster_dic['method']
        self.config = config

        self.checkpoint_path = os.path.join(config.WEIGHTS_PATH, "checkpoint {} {}.ckpt".format(self.cluster_method, self.n_clusters))

        self.model = self.build_model(input_shape)

        if config.LOAD_WEIGHTS:
            if os.path.exists(self.checkpoint_path + ".index"):
                print("Loading model's weights...")
                self.model.load_weights(self.checkpoint_path)
                print("Model's weights successfully loaded!")

            else:
                print("WARNING: model's weights not found, the model will be executed with initialized random weights.")
                print("Ignore this warning if it is a test.")

        self.model.summary()
        print('NSCNet initialization completed.')

    def build_model(self, input_shape):
        model_input = tf.keras.Input(shape=input_shape)
        vae = ConvolutionalVAE(config.STRIDE,
                               config.KERNEL_SIZE,
                               config.PADDING,
                               config.STARTING_FILTERS,
                               config.LATENT_DIM,
                               config.N_CONV_LAYERS,
                               input_shape)
        vae(model_input)
        vae.compile(optimizer=config.OPTIMIZER, run_eagerly=False)

        return vae

    def train_model(self, data):

        callbacks = []
        if config.SAVE_WEIGHTS:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True, verbose=1))

        history = self.model.fit(data,
                                 epochs=self.config.EPOCHS,
                                 batch_size=self.config.BATCH_SIZE,
                                 callbacks=callbacks,
                                 verbose=1)

        # TODO: to remove, just for test.
        # save_test_images(data, model)

        return history

    def save_test_images(self, data):
        mean_x, log_var_x, compressed_shape = self.model.encode(data)

        z = self.model.sample(mean_x, log_var_x)
        decoded_x = self.model.decode(z, compressed_shape)
        for i, img in enumerate(data[:100]):
            plt.imsave(f'data/original/{i}.png', np.squeeze(img, axis=-1), cmap='gray')
        for i, img in enumerate(decoded_x[:100]):
            plt.imsave(f'data/decoded/{i}.png', np.squeeze(img.numpy(), axis=-1), cmap='gray')

    def compute_clusters(self, samples):
        # TODO: predict in batches

        if self.cluster_method not in clustering.CLUSTERING_METHODS:
            raise Exception("cluster method must be one between " + ",".join(clustering.CLUSTERING_METHODS))

        z_mean, z_log_var, compressed_shape = self.model.encode(samples)
        features = self.model.sample(z_mean, z_log_var)

        clustering_output = None
        if self.cluster_method == "kmeans":
            clustering_output = clustering.k_means(features, **self.cluster_args)
        elif self.cluster_method == "dbscan":
            clustering_output = clustering.dbscan(features, **self.cluster_args)

        return clustering_output, features


# -----------------------------------
# Test
'''
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
clusters = compute_clusters(model, inputs, config.CLUSTERING_METHOD, cluster_args)
print("Clustering completed!")
'''
