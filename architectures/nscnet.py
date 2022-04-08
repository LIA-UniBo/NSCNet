import tensorflow as tf
from tensorflow.keras import layers
import os

from architectures.arcface_layer import ArcFace
import architectures.nscnet_config as config
from architectures.pseudo_labels_generator import Generator


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
        self.n_clusters = cluster_dic['n_clusters']
        self.cluster_args = cluster_dic['config']
        self.cluster_method = cluster_dic['method']

        self.checkpoint_path = os.path.join(config.WEIGHTS_PATH, "checkpoint {} {}.ckpt".format(self.cluster_method, self.n_clusters))

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
                              self.cluster_args)

        callbacks = [CustomEarlyStop()]
        if config.SAVE_WEIGHTS:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True, verbose=1))

        history = self.model.fit(x=generator,
                                 verbose=1,
                                 batch_size=config.BATCH_SIZE,
                                 epochs=config.EPOCHS,
                                 callbacks=callbacks)

        return history

    def compute_clusters(self, inputs):
        generator = Generator(inputs,
                              config.BATCH_SIZE,
                              self.model.conv_net,
                              config.POST_PROCESSING_OPTIONS,
                              config.SPEC_AUGMENTATION_OPTIONS,
                              config.EARLY_STOPPING_OPTIONS,
                              self.cluster_method,
                              self.cluster_args,
                              shuffle=False,
                              custom_sampler=False)

        return generator.generate_pseudo_labels()

'''
# -------------------------------------------------
# Test
inputs = import_image_np_dataset(config.IMAGES_PATH, (config.INPUT_SHAPE[0], config.INPUT_SHAPE[1]), config.RGB_NORMALIZATION)



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
'''
