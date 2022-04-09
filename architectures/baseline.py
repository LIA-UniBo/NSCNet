import numpy as np
import architectures.baseline_config as config
from architectures import matrix_manipulation, clustering


class BaseNet:

    def __init__(self, cluster_dic):
        self.n_clusters = cluster_dic['n_clusters']
        self.cluster_args = cluster_dic['config']
        self.cluster_method = cluster_dic['method']

    def compress_data(self, data, normalize, pca, whiten, l2_normalize):

        n_samples = data.shape[0]
        pixels = data.shape[1]*data.shape[2]*data.shape[3]
        data = np.reshape(data, (n_samples, pixels))

        if normalize:
            data = matrix_manipulation.normalize(data)

        data, lost_variance_information = matrix_manipulation.compute_pca(data, pca, whiten)
        print("Lost variance information: {}".format(lost_variance_information))

        if l2_normalize:
            data = matrix_manipulation.l2_normalize(data)

        return data

    def clusterize(self, data):

        if self.cluster_method not in clustering.CLUSTERING_METHODS:
            raise Exception("cluster method must be one between " + ",".join(clustering.CLUSTERING_METHODS))

        features = self.compress_data(data, **config.COMPRESSION_PROCESSING_OPTIONS)

        clustering_output = None
        if self.cluster_method == "kmeans":
            clustering_output = clustering.k_means(features, **self.cluster_args)
        elif self.cluster_method == "dbscan":
            clustering_output = clustering.dbscan(features, **self.cluster_args)

        return clustering_output, features

'''
# ----------------------------------------------
# Test
inputs = import_image_np_dataset(IMAGES_PATH, (INPUT_SHAPE[0], INPUT_SHAPE[1]), RGB_NORMALIZATION)

cluster_args = {
    "n_clusters": N_CLUSTERS
}

features, clusters = clusterize(inputs, CLUSTERING_METHOD, cluster_args, COMPRESSION_PROCESSING_OPTIONS)
visualizer.visualize_clusters(features, clusters)
'''