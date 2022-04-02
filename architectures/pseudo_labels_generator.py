#This module handles all the operations between one epoch and the other in the training phase

import numpy as np
import math
import time
import tensorflow as tf
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

import matrix_manipulation
import clustering
import spec_augmentation
from uniform_cluster_sampler import ClusterSampler

class Generator(tf.keras.utils.Sequence):

    def __init__(self, x, batch_size, conv_net_model, features_extraction_options, spec_augmentation_options, early_stopping_options, cluster_method, cluster_args, shuffle=True, verbose=True, custom_sampler=True):

        """
        Parameters:
        -----------
        x: input of the network to be trained on
        batch_size: batch size used for training
        conv_net_model: layer of the complete network responsible for the extraction of the features
        features_extraction_options:
            "normalize": decide whether normalizing the extracted features
            "pca": dimensions to project the extracted features
            "whiten": decide whether applying whitening during the PCA
            "l2 normalize": decide whether l2-normalizing the extracted features
        spec_augmentation_options:
            "apply": decide if applying spectrograms augmentation or not
            "policy": what policy to use for spec-augmentation
        early_stopping_options:
            "apply": decide to check for early stopping or not
            "min_delta": minimum value of difference between one epoch and the other in NMI to consider it an improvement
            "patience": number of epochs without improvement to look for
        cluster_method: algorithm of clustering
        cluster_args: dictionary of parameters to pass to the clustering algorithm
        shuffle: determine to shuffle to input or not
        verbose: log some info if True
        custom_sampler: decide to use the custom sampler to prepare the batches or not
        """

        if cluster_method not in clustering.CLUSTERING_METHODS:
            raise Exception("cluster method must be one between " + ",".join(clustering.CLUSTERING_METHODS))

        self.x = x
        self.batch_size = batch_size
        self.feature_extractor = conv_net_model
        self.features_extraction_options = features_extraction_options
        self.spec_augmentation_options = spec_augmentation_options
        self.early_stopping_options = early_stopping_options
        self.cluster_method = cluster_method
        self.cluster_args = cluster_args
        self.verbose = verbose
        self.custom_sampler = custom_sampler

        #Shuffle the input data
        if shuffle:
            np.random.shuffle(self.x)

        #Create the first random labels to start the training
        self.y = self.generate_pseudo_labels()
        self.nmi_scores = []

        #Divide the samples in different lists depending on their labels
        if self.custom_sampler:
            self.sampler = ClusterSampler(cluster_args["n_clusters"], batch_size)
            self.sampler.segment_clusters(self.y)

    def __len__(self):

        #Return the number of batches for training

        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):

        """
        Send the next batch of inputs and correspondent labels to the network.
        """

        #Apply uniform sampling to avoid having a batch whose samples belong to the same cluster
        if self.custom_sampler:
            batch_x, batch_y = self.sampler.sample(self.x, self.y)

        #Create batches following data order
        else:
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        #Produce an augmented version of all the images in the batch
        if self.spec_augmentation_options["apply"] is True:
            policy = self.spec_augmentation_options["policy"]
            augment_batch_func = lambda x: spec_augmentation.augment(x, **policy)
            batch_x = list(map(augment_batch_func, batch_x))

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):

        """
        Operations to be done at the end of each epoch
        """

        #Copy the previous labels
        old_y = np.copy(self.y)
        #Generate the new pseudo-labels
        self.y = self.generate_pseudo_labels()

        #Update the metrics involving previous and current labels
        self.update_metrics(old_y, self.y)

        #Divide the samples in different lists depending on their labels
        if self.custom_sampler:
            self.sampler.segment_clusters(self.y)

    def generate_pseudo_labels(self):

        """
        Extract the features and apply a clustering algorithm to get the new pseudo-labels
        """

        start_time = time.time()

        if self.verbose:
            print("Started generating pseudo-labels...")

        #Use the layer responsible for features extraction to extract the features of the entire dataset
        features = self.feature_extractor.predict(self.x, self.batch_size) #(N_samples, 512)

        #Apply all the request operations to the extracted features (PCA, Whitening, Normalization ecc.)
        features = self.run_features_post_processing(features) #(N_samples, PCA_dim)

        #Apply the selected clustering method on the extracted features of the entire dataset
        clustering_output = None
        if self.cluster_method=="kmeans":
            clustering_output = clustering.k_means(features, **self.cluster_args)
        elif self.cluster_method=="dbscan":
            clustering_output = clustering.dbscan(features, **self.cluster_args)

        #Log the required time to produce the new labels
        if self.verbose:
            execution_time = time.time() - start_time
            print("Pseudo-labels generation completed in {} seconds".format(round(execution_time,2)))

        return clustering_output["labels"]

    def run_features_post_processing(self, features):

        """
        Apply all the request operations to the extracted features (PCA, Whitening, Normalization ecc.)
        """

        #Normalize (mean=0 and std=1)
        if self.features_extraction_options["normalize"]:
            features = matrix_manipulation.normalize(features)

        #PCA and Whitening
        pca_n_components = self.features_extraction_options["pca"]
        apply_whitening = self.features_extraction_options["whiten"]
        if pca_n_components is not None:
            features, lost_variance_information = matrix_manipulation.compute_pca(features, pca_n_components, apply_whitening)

        #L2-normalize
        if self.features_extraction_options["l2 normalize"]:
            features = matrix_manipulation.l2_normalize(features)

        return features

    def update_metrics(self, old_y, new_y):

        """
        Update the metrics and check for early stopping
        """

        #Compute the Normalized Mutual Information between old and new pseudo-labels
        nmi_score = nmi(old_y, new_y)
        #Update history of metrics
        self.nmi_scores.append(nmi_score)
        if self.verbose:
            print("NMI score: {}".format(nmi_score))

        #Check for early stopping
        if self.early_stopping_options["apply"]:
            self.check_early_stopping()

    def check_early_stopping(self):

        """
        Stop the training if for N=patience epochs the NMI score does not improve
        """

        min_delta = self.early_stopping_options["min_delta"]
        patience = self.early_stopping_options["patience"]+1

        if len(self.nmi_scores)>=patience:
            last_scores = self.nmi_scores[-patience:] #Take N=patience+1 last scores
            delta_scores = -np.diff(last_scores) #Compute deltas between consecutive scores (epochs)
            no_improvemements = np.all(delta_scores<min_delta) #Check if there is no improvement in all of them

            #Stop the training
            if no_improvemements:
                self.feature_extractor.force_stop=True
