import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, metrics
import os

from architectures.common import clustering
import architectures.vaenet.vaenet_config as config
import matplotlib.pyplot as plt


class Encoder(tf.keras.Model):
    """
    VAENet's encoder creation and management
    """

    def __init__(self, stride, kernel_size, padding, starting_filters, latent_dim, n_conv_layers, dense_units,
                 activation="relu"):
        super(Encoder, self).__init__()

        self.conv_layers = [layers.Conv2D(starting_filters * 2 ** (i - 1),
                                          kernel_size,
                                          strides=stride,
                                          padding=padding,
                                          data_format="channels_last",
                                          activation=activation,
                                          name=f'encoder_conv_{i}')
                            for i in range(1, n_conv_layers + 1)]

        self.last_conv = layers.Conv2D(starting_filters * 2 ** (n_conv_layers - 1),
                                       kernel_size,
                                       strides=stride,
                                       padding=padding,
                                       data_format="channels_last",
                                       activation=activation,
                                       name=f'encoder_conv_{n_conv_layers + 1}')

        self.flatten_layer = layers.Flatten()

        self.dense_layer = layers.Dense(dense_units, activation=activation)

        self.mean_dense_layer = layers.Dense(latent_dim, activation=activation, name="mean")
        self.std_dense_layer = layers.Dense(latent_dim, activation=activation, name="std")

    def call(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.last_conv(x)

        compressed_shape = x.shape

        x = self.flatten_layer(x)

        x = self.dense_layer(x)

        mean_x = self.mean_dense_layer(x)
        log_var_x = self.std_dense_layer(x)

        return [mean_x, log_var_x, compressed_shape]


class Decoder(tf.keras.Model):
    """
    VAENet's decoder creation and management
    """

    def __init__(self, stride, kernel_size, padding, starting_filters, n_conv_layers, input_shape, dense_units,
                 activation="relu"):

        super(Decoder, self).__init__()

        self.activation = activation

        self.conv_transpose = layers.Conv2DTranspose(starting_filters * 2 ** (n_conv_layers - 1),
                                                     kernel_size,
                                                     padding=padding,
                                                     data_format="channels_last",
                                                     activation=activation,
                                                     name=f'decoder_deconv_{n_conv_layers}')

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
                                                   strides=stride,
                                                   padding="same",
                                                   data_format="channels_last",
                                                   activation="sigmoid")

        self.dense_layer_1 = layers.Dense(dense_units, activation=activation)

        self.dense_layer_2 = None
        self.reshape_layer = None

    def call(self, inputs):

        assert len(inputs) == 2

        x = inputs[0]
        compressed_shape = inputs[1]

        if self.dense_layer_2 is None:
            self.dense_layer_2 = layers.Dense(compressed_shape[1] * compressed_shape[2] * compressed_shape[3],
                                              activation=self.activation)
            self.reshape_layer = layers.Reshape((compressed_shape[1], compressed_shape[2], compressed_shape[3]))

        x = self.dense_layer_1(x)
        x = self.dense_layer_2(x)

        x = self.reshape_layer(x)

        x = self.conv_transpose(x)

        for conv_transpose_layer in self.conv_transpose_layers:
            x = conv_transpose_layer(x)

        x = self.output_layer(x)

        return x


class ConvolutionalVAE(tf.keras.Model):
    """
    Wrapper class that merges together the Encoder and the Decoder parts of the VAE.
    """
    def __init__(self,
                 stride,
                 kernel_size,
                 padding,
                 starting_filters,
                 latent_dim,
                 n_conv_layers,
                 dense_units,
                 input_shape,
                 activation="relu"):
        super(ConvolutionalVAE, self).__init__()

        self.encoder = Encoder(stride, kernel_size, padding, starting_filters, latent_dim, n_conv_layers, dense_units,
                               activation)
        self.decoder = Decoder(stride, kernel_size, padding, starting_filters, n_conv_layers, input_shape, dense_units,
                               activation)

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
        """
        REPARAMETRIZATION TRICK

        An important concept related to the theory behind VAEs.

        We could sample directly sample from a normal distribution created by considering the mean and std given by the
        encoder output:

            z = N(mean, std)

        The problem is that this expression is not deterministic (e.g., we could obtain any real number as a result),
        and we would not be able to create a backpropagation expression WRT to the mean and the STD, which are
        parameters that we want to learn.

        But why backpropagation would not work?
        The sampling operation HAS some parameters (e.g., mean and std), but it cannot be seen as a smooth and continue
        function, on which gradients could be computed. Rather, it is a function with many discontinuities.
        Therefore, when computing the gradients WRT to the sampling operation parameters, the gradients would be zero.

        The reparametrization trick overcomes this problem, by means of the following expression:

            z = mean + std ⊙ eps

        where eps is sampled from a normal distribution, it is fixed and it can be treated as an input of the model.
        Once the eps sampling operation is done, all the parameters are known, and the result of this operation becomes
        deterministic. This means that we can write a backpropagation expression WRT to mean and std, letting the
        network learn them.
        """
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

            # data and decoded_x values are between 0 and 1, therefore we can conveniently use the binary cross entropy
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, decoded_x), axis=(1, 2)))

            # p, q are Normal distributions, then it is possible to write the kl_loss as follows:
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
    """
    Class responsible of creating the VAENet, and exposing training and inference methods.
    """

    def __init__(self, input_shape, cluster_dic, debug=False):

        self.weights_name = 'VAENet'
        self.cluster_args = cluster_dic['config']
        self.cluster_method = cluster_dic['method']
        self.config = config
        self.debug = debug

        self.checkpoint_path = os.path.join(config.WEIGHTS_PATH, "checkpoint {}.ckpt".format(self.weights_name))

        self.model = self.build_model(input_shape)

        self.model_already_trained = False
        if config.LOAD_WEIGHTS:
            if os.path.exists(self.checkpoint_path + ".index"):
                print("Loading model's weights...")
                self.model.load_weights(self.checkpoint_path)
                print("Model's weights successfully loaded!")
                self.model_already_trained = True

            else:
                print("WARNING: model's weights not found, the model will be executed with initialized random weights.")
                print("Ignore this warning if it is a test.")
                self.model_already_trained = False

        self.model.summary()
        print('VAENet initialization completed.')

    def build_model(self, input_shape):
        model_input = tf.keras.Input(shape=input_shape)
        vae = ConvolutionalVAE(config.STRIDE,
                               config.KERNEL_SIZE,
                               config.PADDING,
                               config.STARTING_FILTERS,
                               config.LATENT_DIM,
                               config.N_CONV_LAYERS,
                               config.DENSE_UNITS,
                               input_shape)
        vae(model_input)
        vae.compile(optimizer=config.OPTIMIZER, run_eagerly=False)
        return vae

    def train_model(self, data):

        callbacks = []
        if config.SAVE_WEIGHTS:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True, verbose=1))

        history = self.model.fit(data,
                                 epochs=self.config.EPOCHS,
                                 batch_size=self.config.BATCH_SIZE,
                                 callbacks=callbacks,
                                 verbose=1)

        if self.debug:
            print("DEBUG: Saving image data...")
            self.save_test_images(data)

        return history

    def save_test_images(self, data):
        mean_x, log_var_x, compressed_shape = self.model.encode(data[:100])

        z = self.model.sample(mean_x, log_var_x)
        decoded_x = self.model.decode(z, compressed_shape)
        for i, img in enumerate(data[:100]):
            plt.imsave(f'data/original/{i}.png', img)
        for i, img in enumerate(decoded_x[:100]):
            plt.imsave(f'data/decoded/{i}.png', img.numpy())

    def compute_clusters(self, samples):
        features = []
        n_samples = samples.shape[0]

        for i in range(0, n_samples, config.BATCH_SIZE):
            batch = samples[i:i + config.BATCH_SIZE]
            z_mean, z_var, _ = self.model.encode(batch)
            features.extend(self.model.sample(z_mean, z_var))

        if self.cluster_method not in clustering.CLUSTERING_METHODS:
            raise Exception("cluster method must be one between " + ",".join(clustering.CLUSTERING_METHODS))

        # z_mean, z_log_var, compressed_shape = self.model.encode(samples)
        # features = self.model.sample(z_mean, z_log_var)

        clustering_output = None
        if self.cluster_method == "kmeans":
            clustering_output = clustering.k_means(features, **self.cluster_args)
        elif self.cluster_method == "dbscan":
            clustering_output = clustering.dbscan(features, **self.cluster_args)

        return clustering_output, features
