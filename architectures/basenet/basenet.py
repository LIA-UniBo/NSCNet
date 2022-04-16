from sklearn.neighbors import NearestNeighbors

import architectures.basenet.basenet_config as config
from architectures.common import clustering, matrix_manipulation
import matplotlib.pyplot as plt
import numpy as np


class BaseNet:

    def __init__(self, cluster_dic):
        self.cluster_args = cluster_dic['config']
        self.cluster_method = cluster_dic['method']

    def compress_data(self, data, normalize, pca, whiten, l2_normalize):

        n_samples = data.shape[0]
        pixels = data.shape[1]*data.shape[2]*data.shape[3]
        data = np.reshape(data, (n_samples, pixels))

        if normalize:
            data = matrix_manipulation.normalize(data)

        data, lost_variance_information = matrix_manipulation.compute_pca(data, pca, whiten)
        print("Lost variance information after PCA: {}".format(lost_variance_information))

        if l2_normalize:
            data = matrix_manipulation.l2_normalize(data)

        return data

    def compute_clusters(self, data, features=None):

        if self.cluster_method not in clustering.CLUSTERING_METHODS:
            raise Exception("cluster method must be one between " + ",".join(clustering.CLUSTERING_METHODS))

        # Avoid to compute multiple times the same operation during different training
        if features is None:
            features = self.compress_data(data, **config.COMPRESSION_PROCESSING_OPTIONS)

        '''
        # # TODO: this could be useful for finding the EPS
        for i in range(2, 50):
            fig = plt.figure()
            neighbors = NearestNeighbors(n_neighbors=i)
            neighbors_fit = neighbors.fit(features)
            distances, indices = neighbors_fit.kneighbors(features)
            distances = np.sort(distances, axis=0)
            distances = distances[:, 1]
            plt.plot(distances)
            # plt.show(block=True)

            plt.savefig(f'{i}.png', bbox_inches='tight')
            plt.close(fig)
        return
        '''

        clustering_output = None
        if self.cluster_method == "kmeans":
            clustering_output = clustering.k_means(features, **self.cluster_args)
        elif self.cluster_method == "dbscan":
            clustering_output = clustering.dbscan(features, **self.cluster_args)

        return clustering_output, features
