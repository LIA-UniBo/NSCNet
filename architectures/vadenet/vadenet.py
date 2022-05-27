import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, metrics
import os
import math

from architectures.common import clustering
import architectures.vadenet.vadenet_config as config
from architectures.vadenet.gmm_variables_initializer import estimate_gmm_variables
import matplotlib.pyplot as plt

#Idea: Zipfs law can be forced (knowledge injection)
#Idea: The number of clusters can be less, if some probabilities go to zero

class Encoder(tf.keras.Model):
    """
    VADENet's encoder creation and management
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

        self.dense_layer_1 = layers.Dense(dense_units, activation=activation)
        self.dense_layer_2 = layers.Dense(dense_units//4, activation=activation)

        self.mean_dense_layer = layers.Dense(latent_dim, name="mean")
        self.std_dense_layer = layers.Dense(latent_dim, name="std")

    def call(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = self.last_conv(x)

        compressed_shape = x.shape

        x = self.flatten_layer(x)

        x = self.dense_layer_1(x)
        x = self.dense_layer_2(x)

        mean_x = self.mean_dense_layer(x)
        log_var_x = self.std_dense_layer(x)

        return [mean_x, log_var_x, compressed_shape]


class Decoder(tf.keras.Model):
    """
    VADENet's decoder creation and management
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

        self.dense_layer_1 = layers.Dense(dense_units//4, activation=activation)
        self.dense_layer_2 = layers.Dense(dense_units, activation=activation)

        self.dense_layer_3 = None
        self.reshape_layer = None

    def call(self, inputs):

        assert len(inputs) == 2

        x = inputs[0]
        compressed_shape = inputs[1]

        if self.dense_layer_3 is None:
            self.dense_layer_3 = layers.Dense(compressed_shape[1] * compressed_shape[2] * compressed_shape[3],
                                              activation=self.activation)
            self.reshape_layer = layers.Reshape((compressed_shape[1], compressed_shape[2], compressed_shape[3]))

        x = self.dense_layer_1(x)
        x = self.dense_layer_2(x)
        x = self.dense_layer_3(x)

        x = self.reshape_layer(x)

        x = self.conv_transpose(x)

        for conv_transpose_layer in self.conv_transpose_layers:
            x = conv_transpose_layer(x)

        x = self.output_layer(x)

        return x

class ConvolutionalVAE(tf.keras.Model):
    """
    Wrapper class that merges together the Encoder and the Decoder parts of the VADE.
    """
    def __init__(self,
                 n_clusters,
                 stride,
                 kernel_size,
                 padding,
                 starting_filters,
                 latent_dim,
                 n_conv_layers,
                 dense_units,
                 input_shape,
                 activation="relu",
                 gmm_vars_init=[None,None,None]):

        super(ConvolutionalVAE, self).__init__()

        self.n_clusters = n_clusters
        self.latent_dim = latent_dim

        self.eps = 1e-10

        self.pretrain = True

        self.encoder = Encoder(stride, kernel_size, padding, starting_filters, latent_dim, n_conv_layers, dense_units,
                               activation)
        self.decoder = Decoder(stride, kernel_size, padding, starting_filters, n_conv_layers, input_shape, dense_units,
                               activation)

        self.encoder_weights_path = os.path.join(config.WEIGHTS_PATH, "encoder_weights.ckpt")
        self.decoder_weights_path = os.path.join(config.WEIGHTS_PATH, "decoder_weights.ckpt")

        self.initialize_gmm_variables(gmm_vars_init)

        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
        self.entropy_loss_tracker = metrics.Mean(name="entropy_loss")

    @property
    def softmax_pi(self):

        k = tf.range(self.n_clusters, dtype=tf.float32) + 1.0
        k_to_s = tf.math.pow(k, tf.math.abs(self.s))
        harmonic_number = tf.reduce_sum(1.0 / k_to_s)

        pi = 1.0 / (k_to_s * harmonic_number)

        return pi

        #return tf.nn.softmax(self.pi)

    def initialize_gmm_variables(self, gmm_vars_init):

        assert len(gmm_vars_init)==3

        pi_init = gmm_vars_init[0]
        mu_init = gmm_vars_init[1]
        sigma_init = gmm_vars_init[2]

        if pi_init is None:
            pi_init = np.ones(self.n_clusters)/self.n_clusters
            #pi_init = np.random.rand(self.n_clusters)

        if mu_init is None:
            #mu_init = np.zeros((self.latent_dim, self.n_clusters))
            mu_init = tf.random.normal(shape=(self.n_clusters, self.latent_dim))

        if sigma_init is None:
            #sigma_init = np.ones((self.latent_dim, self.n_clusters))
            sigma_init = tf.random.normal(shape=(self.n_clusters, self.latent_dim))

        self.s = tf.Variable(initial_value=tf.constant(1.0),
                            trainable=True,
                            name="s",
                            dtype=tf.float32)

        #self.pi = tf.Variable(initial_value=pi_init,
        #                      trainable=True,
        #                      name="pi",
        #                      dtype=tf.float32)

        self.mu = tf.Variable(initial_value=mu_init,
                              trainable=True,
                              name="mean",
                              dtype=tf.float32)

        self.sigma = tf.Variable(initial_value=sigma_init,
                              trainable=True,
                              name="std",
                              dtype=tf.float32)

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

    def normalize(self, distribution, axis=-1):

        if axis==0:
            return distribution / tf.reduce_sum(distribution, axis=axis, keepdims=False)
        else:
            return distribution / tf.reduce_sum(distribution, axis=axis, keepdims=True)

    def get_entropy(self, distribution, axis=-1):

        distribution = self.normalize(distribution, axis)

        entropy =  - tf.reduce_sum(distribution * tf.math.log(distribution + self.eps), axis=axis)
        return entropy

    def get_conditioned_probability(self, z):

        z = tf.expand_dims(z, 1)

        h = z - self.mu
        h = tf.exp(-0.5 * tf.reduce_mean(h*h / tf.exp(self.sigma), axis=2))
        h = h / tf.exp(tf.reduce_mean(0.5 * self.sigma, axis=1))
        p_z_given_c = h / (2 * math.pi) + self.eps

        return p_z_given_c

    def get_gamma(self, p_z_given_c):

        p_c_z = self.normalize(p_z_given_c) * self.softmax_pi

        norm_p_c_z = self.normalize(p_c_z)

        return norm_p_c_z

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker, self.entropy_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, compressed_shape = self.encode(data)
            z = self.sample(z_mean, z_log_var)
            decoded_x = self.decode(z, compressed_shape)

            # data and decoded_x values are between 0 and 1, therefore we can conveniently use the binary cross entropy
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, decoded_x), axis=(1, 2)))

            p_z_given_c = self.get_conditioned_probability(z)
            gamma = self.get_gamma(p_z_given_c)

            h = tf.expand_dims(tf.exp(z_log_var),1) + tf.square((tf.expand_dims(z_mean,1) - self.mu))
            h = tf.reduce_sum(self.sigma + h / tf.exp(self.sigma), axis=2)
            kl_loss = 0.5 * tf.reduce_sum(gamma * h) \
                    - tf.reduce_sum(gamma * tf.math.log(self.softmax_pi + self.eps)) \
                    + tf.reduce_sum(gamma * tf.math.log(gamma + self.eps)) \
                    - 0.5 * tf.reduce_sum(z_log_var + 1)

            entropy_loss = tf.reduce_sum(self.get_entropy(gamma))
            entropy_loss = 100*tf.math.abs(entropy_loss - 110)

            s_regularizer = 100*tf.math.abs(tf.math.abs(self.s) - 0.8)

            if self.pretrain:
                total_loss = reconstruction_loss
            else:
                total_loss = reconstruction_loss + 1e-2*kl_loss + s_regularizer + entropy_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.entropy_loss_tracker.update_state(entropy_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "entropy_loss": self.entropy_loss_tracker.result()
        }

class SaveVAECallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        self.model.encoder.save_weights(self.model.encoder_weights_path)
        self.model.decoder.save_weights(self.model.decoder_weights_path)
        print("Encoder and Decoder weights saved correctly")

class VADENet:
    """
    Class responsible of creating the VADENet, and exposing training and inference methods.
    """

    def __init__(self, input_shape, cluster_dic, debug=False, random_init=True):

        self.weights_name = 'VADENet'
        self.cluster_args = cluster_dic['config']
        self.cluster_method = cluster_dic['method']
        self.config = config
        self.debug = debug
        self.random_init = random_init
        self.in_shape = input_shape

        self.n_clusters = self.cluster_args['n_clusters']

        self.model = self.build_model(input_shape, self.n_clusters)

        self.weights_loaded = False

        if config.LOAD_WEIGHTS:
            if os.path.exists(self.model.encoder_weights_path + ".index"):
                print("Loading model's weights...")
                self.model.encoder.load_weights(self.model.encoder_weights_path)
                self.model.decoder.load_weights(self.model.decoder_weights_path)
                self.weights_loaded = True
                print("Model's weights successfully loaded!")

            else:
                print("WARNING: model's weights not found, the model will be executed with initialized random weights.")
                print("Ignore this warning if it is a test.")

        self.model.summary()
        print('VADENet initialization completed.')

    def build_model(self, input_shape, n_clusters):

        gmm_init = [None,None,None] if self.random_init else estimate_gmm_variables(config.LATENT_DIM, self.n_clusters)

        model_input = tf.keras.Input(shape=input_shape)
        vae = ConvolutionalVAE(n_clusters,
                               config.STRIDE,
                               config.KERNEL_SIZE,
                               config.PADDING,
                               config.STARTING_FILTERS,
                               config.LATENT_DIM,
                               config.N_CONV_LAYERS,
                               config.DENSE_UNITS,
                               input_shape,
                               gmm_vars_init=gmm_init)
        vae(model_input)
        vae.compile(optimizer=config.OPTIMIZER, run_eagerly=False)
        return vae

    def rebuild_model(self):
        model_input = tf.keras.Input(shape=self.in_shape)
        self.model(model_input)
        self.model.compile(optimizer=config.OPTIMIZER, run_eagerly=False)

    def train_model(self, data):

        callbacks = []
        if config.SAVE_WEIGHTS:
            callbacks.append(SaveVAECallback())

        if config.PRETRAIN_EPOCHS>0 and not self.weights_loaded:
            print("Pretraining...")

            history = self.model.fit(data,
                                     epochs=self.config.PRETRAIN_EPOCHS,
                                     batch_size=self.config.BATCH_SIZE,
                                     callbacks=callbacks,
                                     verbose=1)

            print("Pretraining ended")

        self.model.pretrain = False
        self.rebuild_model()

        history = self.model.fit(data,
                                 epochs=self.config.EPOCHS,
                                 batch_size=self.config.BATCH_SIZE,
                                 callbacks=callbacks,
                                 verbose=1)

        if self.debug:
            print("DEBUG: Saving image data...")
            for i, batch in enumerate(data):
                if i==0:
                    self.save_test_images(batch.numpy())

        return history

    def save_test_images(self, data):
        mean_x, log_var_x, compressed_shape = self.model.encode(data)

        z = self.model.sample(mean_x, log_var_x)
        decoded_x = self.model.decode(z, compressed_shape)
        for i, img in enumerate(data):
            plt.imsave(f'data/original/{i}.png', img)
        for i, img in enumerate(decoded_x):
            plt.imsave(f'data/decoded/{i}.png', img.numpy())

    def compute_clusters(self, samples):

        if self.cluster_method not in clustering.CLUSTERING_METHODS:
            raise Exception("cluster method must be one between " + ",".join(clustering.CLUSTERING_METHODS))

        features = []
        gammas = []

        for batch in samples:
            z_mean, z_var, _ = self.model.encode(batch)
            z = self.model.sample(z_mean, z_var)
            features.extend(z)
            p_z_given_c = self.model.get_conditioned_probability(z)
            gammas.extend(self.model.get_gamma(p_z_given_c))

        clustering_output = None

        if self.cluster_method == "gmm":
            if self.cluster_args["auto"]:
                gammas = np.array(gammas)
                cluster_predictions = np.argmax(gammas, axis=1)
                print(cluster_predictions)
                print(self.model.s)
                silhouette_avg, silhouette_sample_scores = clustering.compute_silouhette(features, cluster_predictions)
                clustering_output = {
                    "labels": cluster_predictions,
                    "silhouette": silhouette_avg,
                    "silhouette_sample_scores": silhouette_sample_scores,
                    "aic": None,
                    "bic": None
                }
            else:
                clustering_output = clustering.gaussian_mixture(features, **self.cluster_args)

        elif self.cluster_method == "kmeans":
            clustering_output = clustering.k_means(features, **self.cluster_args)

        #elif self.cluster_method == "dbscan":
            #clustering_output = clustering.dbscan(features, **self.cluster_args)

        return clustering_output, features
