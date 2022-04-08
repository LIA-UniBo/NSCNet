import numpy as np

from architectures import matrix_manipulation, clustering, visualizer
from architectures.images_loader import import_image_np_dataset

from architectures.baseline_config import *


def compress_data(data, normalize, pca, whiten, l2_normalize):

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


def clusterize(data, cluster_method, cluster_args, compression_options):

    if cluster_method not in clustering.CLUSTERING_METHODS:
        raise Exception("cluster method must be one between " + ",".join(clustering.CLUSTERING_METHODS))

    features = compress_data(data, **compression_options)

    clustering_output = None
    if cluster_method == "kmeans":
        clustering_output = clustering.k_means(features, **cluster_args)
    elif cluster_method == "dbscan":
        clustering_output = clustering.dbscan(features, **cluster_args)

    return features, clustering_output["labels"]


# ----------------------------------------------
# Test
inputs = import_image_np_dataset(IMAGES_PATH, (INPUT_SHAPE[0], INPUT_SHAPE[1]), RGB_NORMALIZATION)

cluster_args = {
    "n_clusters": N_CLUSTERS
}

features, clusters = clusterize(inputs, CLUSTERING_METHOD, cluster_args, COMPRESSION_PROCESSING_OPTIONS)
visualizer.visualize_clusters(features, clusters)
