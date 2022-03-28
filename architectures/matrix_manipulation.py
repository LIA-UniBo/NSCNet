import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

def normalize(x):

    normalized_x = StandardScaler().fit_transform(x)
    return normalized_x

def l2_normalize(x):

    l2_normalized_x = Normalizer(norm='l2').fit_transform(x)
    return l2_normalized_x

def compute_pca(x, n_components, apply_whitening):

    pca = PCA(n_components=n_components, whiten=apply_whitening)
    principal_components = pca.fit_transform(x)

    lost_variance_information = 1-sum(pca.explained_variance_ratio_)

    return principal_components, lost_variance_information

def rgb_normalize(x):

    return x/255
