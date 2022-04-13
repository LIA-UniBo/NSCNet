import tensorflow as tf
from tensorflow.keras import layers
import os

from architectures.nscnet.arcface_layer import ArcFace
import architectures.nscnet.nscnet_config as config
from architectures.nscnet.pseudo_labels_generator import Generator


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


class NSCNet:
    def __init__(self, input_shape, cluster_dic):

        self.weights_name = cluster_dic['name']
        self.cluster_args = cluster_dic['config']
        self.cluster_method = cluster_dic['method']

        self.n_clusters = self.cluster_args['n_clusters']

        self.checkpoint_path = os.path.join(config.WEIGHTS_PATH, "checkpoint {}.ckpt".format(self.weights_name))

        self.model = self.build_model(self.n_clusters, input_shape)

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

    def build_model(self, n_clusters, input_shape):

        model_input = tf.keras.Input(shape=input_shape)

        conv_net = ConvNet(input_shape,
                           config.POOLING,
                           config.DIM_REPRESENTATION)

        model = Classifier(conv_net,
                           n_clusters,
                           use_arcface=config.USE_ARCFACE_LOSS)
        model(model_input)
        model.compile(optimizer=config.OPTIMIZER, loss=config.LOSS)

        return model

    def train_model(self, inputs):
        generator = Generator(inputs,
                              config.BATCH_SIZE,
                              self.model.conv_net,
                              config.POST_PROCESSING_OPTIONS,
                              config.SPEC_AUGMENTATION_OPTIONS,
                              config.EARLY_STOPPING_OPTIONS,
                              self.cluster_method,
                              self.cluster_args,
                              config.BATCHES_PER_EPOCH)

        callbacks = [CustomEarlyStop()]
        if config.SAVE_WEIGHTS:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True, verbose=1))

        history = self.model.fit(x=generator,
                                 verbose=1,
                                 batch_size=config.BATCH_SIZE,
                                 epochs=config.EPOCHS,
                                 callbacks=callbacks)

        return history, generator.nmi_scores

    def compute_clusters(self, inputs):

        generator = Generator(inputs,
                              config.BATCH_SIZE,
                              self.model.conv_net,
                              config.POST_PROCESSING_OPTIONS,
                              config.SPEC_AUGMENTATION_OPTIONS,
                              config.EARLY_STOPPING_OPTIONS,
                              self.cluster_method,
                              self.cluster_args,
                              config.batches_per_epoch,
                              shuffle=False,
                              custom_sampler=False,
                              generate_label_on_init=False)

        return generator.generate_pseudo_labels()
