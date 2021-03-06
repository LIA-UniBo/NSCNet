# This module works as an interface for the sklearn to apply a clustering algorithm

import numpy as np
from sklearn.cluster import KMeans, OPTICS
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples

CLUSTERING_METHODS = ["kmeans", "dbscan", "gmm"]

def compute_silouhette(x, cluster_predictions):

    silhouette_avg = silhouette_score(x, cluster_predictions)
    silhouette_sample_scores = silhouette_samples(x, cluster_predictions)

    return silhouette_avg, silhouette_sample_scores

def k_means(x, n_clusters, max_iterations=300, batch_size=None, compute_scores=False, **kwargs):
    """
    This method computes the k-means clustering in a classical (iterative) form or with the mini-batch approach.

    Parameters:
    -----------
    x: tensor of samples to be clustered
    n_clusters: number of clusters, K parameter of the algorithm
    max_iterations: maximum number of iterations if it does not converge before
    batch_size: dimension of the batch-size to use in the mini-batch mode; set to None for the normal algorithm

    Returns:
    ----------
    Dictionary:
        labels -> clustered labels for the sample points
        inertia -> inertia score of the k-means
    """

    # Note: documentation suggests a batch size of 256 * number of cores to exploit parallelism

    # Create the appropriate clustering algorithm according to the batch_size
    if batch_size is None:
        kmeans = KMeans(n_clusters=n_clusters,
                        max_iter=max_iterations,
                        random_state=0)
    else:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                                 random_state=0,
                                 batch_size=batch_size,
                                 max_iter=max_iterations).fit_predict(x)

    silhouette_avg = None
    silhouette_sample_scores = None

    # Make predictions
    cluster_predictions = kmeans.fit_predict(x)

    # Compute additional metric scores if required
    if compute_scores:
        silhouette_avg, silhouette_sample_scores = compute_silouhette(x, cluster_predictions)

    return {
        "labels": cluster_predictions,
        "silhouette": silhouette_avg,
        "silhouette_sample_scores": silhouette_sample_scores,
        "inertia": kmeans.inertia_
    }


def dbscan(x, eps, min_samples, metric="euclidean", compute_scores=False, **kwargs):

    """
    Computes the DBSCAN algorithm.

    Parameters:
    -----------
    x: tensor of samples to be clustered
    eps: epsilon radius of the DBSCAN
    min_samples: minimum number of samples to put into a cluster
    metric: distance metric to be used

    Returns:
    --------
    Dictionary:
        labels -> clustered labels for the sample points
    """

    # Code left for easily switch to DBSCAN, if required
    '''
    dbscan_clustering = DBSCAN(eps=eps,
                               min_samples=min_samples,
                               metric=metric,
                               n_jobs=None)
    '''

    dbscan_clustering = OPTICS(
                               min_samples=min_samples,
                               metric=metric,
                               n_jobs=None)

    # Make predictions and compute the metrics
    cluster_predictions = dbscan_clustering.fit_predict(x)

    silhouette_avg = None
    silhouette_sample_scores = None

    if compute_scores:
        silhouette_avg, silhouette_sample_scores = compute_silouhette(x, cluster_predictions)

        if len(np.unique(cluster_predictions)) > 1:
            silhouette_avg = silhouette_score(x, cluster_predictions)
            silhouette_sample_scores = silhouette_samples(x, cluster_predictions)

    return {
        "labels": cluster_predictions,
        "silhouette": silhouette_avg,
        "silhouette_sample_scores": silhouette_sample_scores
    }

def gaussian_mixture(x, n_clusters, max_iter=300, n_init=3, compute_scores=False, get_params=False, **kwargs):

    gmm = GaussianMixture(n_components=n_clusters,
                        covariance_type='diag',
                        max_iter=max_iter,
                        n_init=n_init,
                        random_state=0,
                        verbose=2,
                        verbose_interval=10)

    # Make predictions and compute the metrics
    gmm_fitted = gmm.fit(x)

    if not get_params:
        cluster_predictions = gmm.predict(x)

        silhouette_avg = None
        silhouette_sample_scores = None
        aic = None
        bic = None

        if compute_scores:
            silhouette_avg, silhouette_sample_scores = compute_silouhette(x, cluster_predictions)
            aic = gmm_fitted.aic(np.array(x))
            bic = gmm_fitted.bic(np.array(x))

        return {
            "labels": cluster_predictions,
            "silhouette": silhouette_avg,
            "silhouette_sample_scores": silhouette_sample_scores,
            "aic": aic,
            "bic": bic
        }

    else:
        return gmm_fitted.weights_, gmm_fitted.means_, gmm_fitted.covariances_
