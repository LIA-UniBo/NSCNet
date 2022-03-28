import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from pseudo_labels_generator import Generator
from images_loader import import_image_np_dataset
import spec_augmentation

POST_PROCESSING_OPTIONS = {
"normalize": True,
"pca": 128,
"whiten": True,
"l2 normalize": True
}

RGB_NORMALIZATION = True

SPEC_AUGMENTATION_OPTIONS = {
"apply": True,
"policy": spec_augmentation.POLICIES["LB"]
}

IMAGES_PATH = "Samples"

POOLING = "max"
DIM_REPRESENTATION = 512
INPUT_SHAPE = (480,640,3)
N_CLUSTERS = 20 #Test value
CLUSTERING_METHOD = "kmeans"

LEARNING_RATE = 1e-3
OPTIMIZER = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.8, beta_2=0.999, epsilon=1e-7)
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
BATCH_SIZE = 32
EPOCHS = 50

class ConvNet(tf.keras.Model):

    def __init__(self, input_shape, pooling, linear_units):

        super(ConvNet, self).__init__()

        self.efficient_net = tf.keras.applications.EfficientNetB0(include_top=False,
                                                                weights=None,
                                                                input_shape=input_shape,
                                                                pooling=pooling)

        self.linear_layer = layers.Dense(linear_units, activation="relu")

    def call(self, x):

        x = self.efficient_net(x)
        x = self.linear_layer(x)

        return x

class Classifier(tf.keras.Model):

    def __init__(self, conv_net, n_clusters):

        super(Classifier, self).__init__()

        self.conv_net = conv_net
        self.classification_head = layers.Dense(n_clusters, activation="softmax")

    def call(self, x):

        x = self.conv_net(x)
        x = self.classification_head(x)

        return x

def build_model(input_shape, pooling, linear_units, n_clusters, optimizer, loss_function):

    input = tf.keras.Input(shape=input_shape)

    conv_net = ConvNet(input_shape, pooling, linear_units)

    model = Classifier(conv_net, n_clusters)
    model(input)
    #TODO: add metrics (NMI? Inertia?)
    model.compile(optimizer=optimizer, loss=loss_function)

    return model

def train_model(model, inputs, batch_size, epochs, callbacks, post_processing_options, spec_augmentation_options, cluster_method, cluster_args):

    #TODO: add early stopping for NMI

    generator = Generator(inputs,
                        batch_size,
                        model.conv_net,
                        post_processing_options,
                        spec_augmentation_options,
                        cluster_method,
                        cluster_args)

    history = model.fit(x=generator,
                        verbose=1,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks)

#-------------------------------------------------
#Test
inputs = import_image_np_dataset(IMAGES_PATH, (INPUT_SHAPE[0],INPUT_SHAPE[1]), RGB_NORMALIZATION)

cluster_args = {
"n_clusters":N_CLUSTERS
}

model = build_model(INPUT_SHAPE, POOLING, DIM_REPRESENTATION, N_CLUSTERS, OPTIMIZER, LOSS)
model.summary()
train_model(model, inputs, BATCH_SIZE, EPOCHS, [], POST_PROCESSING_OPTIONS, SPEC_AUGMENTATION_OPTIONS ,CLUSTERING_METHOD, cluster_args)
