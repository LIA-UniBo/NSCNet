import numpy as np
import math
import time
import tensorflow as tf

import matrix_manipulation
import clustering

class Generator(tf.keras.utils.Sequence):

    def __init__(self, x, batch_size, conv_net_model, features_extraction_options, cluster_method, cluster_args, verbose=True):

        #TODO: consider shuffle here or before call

        if cluster_method not in clustering.CLUSTERING_METHODS:
            raise Exception("cluster method must be one between " + ",".join(clustering.CLUSTERING_METHODS))

        self.x = x
        self.batch_size = batch_size
        self.feature_extractor = conv_net_model
        self.features_extraction_options = features_extraction_options
        self.cluster_method = cluster_method
        self.cluster_args = cluster_args
        self.verbose = verbose

        self.y = self.generate_pseudo_labels()

    def __len__(self):

        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):

        #TODO: add Spec augmentation here

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):

        #TODO: add new shuffle with uniform cluster distribution

        self.y = self.generate_pseudo_labels()

    def generate_pseudo_labels(self):

        start_time = time.time()

        if self.verbose:
            print("Started generating pseudo-labels...")

        features = self.feature_extractor.predict(self.x, self.batch_size)

        features = self.run_features_post_processing(features)

        clustering_output = None
        if self.cluster_method=="kmeans":
            clustering_output = clustering.k_means(features, **self.cluster_args)
        elif self.cluster_method=="dbscan":
            clustering_output = clustering.dbscan(features, **self.cluster_args)

        if self.verbose:
            execution_time = time.time() - start_time
            print("Pseudo-labels generation completed in {} seconds".format(round(execution_time,2)))

        return clustering_output["labels"]

    def run_features_post_processing(self, features):

        if self.features_extraction_options["normalize"]:
            features = matrix_manipulation.normalize(features)

        pca_n_components = self.features_extraction_options["pca"]
        apply_whitening = self.features_extraction_options["whiten"]
        if pca_n_components is not None:
            features, lost_variance_information = matrix_manipulation.compute_pca(features, pca_n_components, apply_whitening)

        if self.features_extraction_options["l2 normalize"]:
            features = matrix_manipulation.l2_normalize(features)

        return features
