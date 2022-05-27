import config

from architectures.common import clustering, matrix_manipulation
from architectures.common.images_loader import import_image_np_dataset

import numpy as np

def load_data():

    print("Loading data for GMM initialization...")

    inputs = import_image_np_dataset(config.IMAGES_PATH,
                                     (config.INPUT_SHAPE[0], config.INPUT_SHAPE[1]),
                                     config.RGB_NORMALIZATION)

    print("Data loaded for GMM initialization!")

    return inputs

def compress_data(normalize, pca, whiten, l2_normalize):

    data = load_data()

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

def estimate_gmm_variables(latent_dim, n_clusters, normalize=True, whiten=True, l2_normalize=True):

    features = compress_data(normalize, latent_dim, whiten, l2_normalize)

    pi, mu, sigma = clustering.gaussian_mixture(features, n_clusters, get_params=True)

    return [pi, mu, sigma]
