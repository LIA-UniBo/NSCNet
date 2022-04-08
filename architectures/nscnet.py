import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from architectures.arcface_layer import ArcFace
from architectures.images_loader import import_image_np_dataset
from architectures.pseudo_labels_generator import Generator
import architectures.nscnet_config as config


class ConvNet(tf.keras.Model):

    def __init__(self, input_shape, pooling, linear_units):
        super(ConvNet, self).__init__()

        self.efficient_net = tf.keras.applications.EfficientNetB0(include_top=False,
                                                                  weights=None,
                                                                  input_shape=input_shape,
                                                                  pooling=pooling)

        self.linear_layer = layers.Dense(linear_units, activation="relu")

        self.force_stop = False

    def call(self, x):
        x = self.efficient_net(x)
        x = self.linear_layer(x)

        return x


class Classifier(tf.keras.Model):

    def __init__(self, conv_net, n_clusters, use_arcface=False):
        super(Classifier, self).__init__()

        self.conv_net = conv_net

        if use_arcface:
            self.classification_head = ArcFace(n_clusters)
        else:
            self.classification_head = layers.Dense(n_clusters, activation="softmax")

    def call(self, x):
        x = self.conv_net(x)
        x = self.classification_head(x)

        return x


class CustomEarlyStop(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if self.model.conv_net.force_stop:
            print("\nEarly stopping...")
            self.model.stop_training = True


def build_model(input_shape, pooling, linear_units, n_clusters, optimizer, loss_function, use_arcface):
    input = tf.keras.Input(shape=input_shape)

    conv_net = ConvNet(input_shape, pooling, linear_units)

    model = Classifier(conv_net, n_clusters, use_arcface=use_arcface)
    model(input)
    model.compile(optimizer=optimizer, loss=loss_function)

    return model


def train_model(model, inputs, batch_size, epochs, callbacks, post_processing_options, spec_augmentation_options,
                early_stopping_options, cluster_method, cluster_args):

    generator = Generator(inputs,
                          batch_size,
                          model.conv_net,
                          post_processing_options,
                          spec_augmentation_options,
                          early_stopping_options,
                          cluster_method,
                          cluster_args)

    history = model.fit(x=generator,
                        verbose=1,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks)

    return history

# -------------------------------------------------
# Test
inputs = import_image_np_dataset(config.IMAGES_PATH, (config.INPUT_SHAPE[0], config.INPUT_SHAPE[1]), config.RGB_NORMALIZATION)

cluster_args = {
    "n_clusters": config.N_CLUSTERS
}

model = build_model(config.INPUT_SHAPE,
                    config.POOLING,
                    config.DIM_REPRESENTATION,
                    config.N_CLUSTERS,
                    config.OPTIMIZER,
                    config.LOSS,
                    config.USE_ARCFACE_LOSS)
model.summary()

history = train_model(model,
            inputs,
            config.BATCH_SIZE,
            config.EPOCHS,
            [CustomEarlyStop()],
            config.POST_PROCESSING_OPTIONS,
            config.SPEC_AUGMENTATION_OPTIONS,
            config.EARLY_STOPPING_OPTIONS,
            config.CLUSTERING_METHOD,
            cluster_args)
