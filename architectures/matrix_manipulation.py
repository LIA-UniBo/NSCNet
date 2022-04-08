# In this module there are all the functions that are used to do some algebric operations on matrices

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def normalize(x):

    # Normalize a matrix to have mean=0 and std=1

    normalized_x = StandardScaler().fit_transform(x)
    return normalized_x


def l2_normalize(x):

    # Apply the l2-normalization

    l2_normalized_x = Normalizer(norm='l2').fit_transform(x)
    return l2_normalized_x


def compute_pca(x, n_components, apply_whitening):

    """
    Compute the principal components analysis and so a matrix compression.

    Parameters:
    -----------
    x: matrix to be compressed
    n_components: new number of dimensions in which the matrix will be compressed
    apply_whitening: boolean value to decide if the matrix should also be whitened

    Returns:
    --------
    principal_components: compressed matrix
    lost_variance_information: information lost during the compression (between 0 and 1)
    """

    pca = PCA(n_components=n_components, whiten=apply_whitening)
    principal_components = pca.fit_transform(x)

    lost_variance_information = 1-sum(pca.explained_variance_ratio_)

    return principal_components, lost_variance_information

def compute_lda(x, y, n_components):

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    projected_data = lda.fit_transform(x,y)

    lost_variance_information = 1-sum(lda.explained_variance_ratio_)

    return projected_data, lost_variance_information

def rgb_normalize(x):
    # Normalize the intervals of an rgb image from [0,255] to [0.0,1.0]

    return x/255
