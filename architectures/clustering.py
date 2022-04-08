# This module works as an interface for the sklearn to apply a clustering algorithm

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score  # , silhouette_samples

CLUSTERING_METHODS = ["kmeans", "dbscan"]


def k_means(x, n_clusters, max_iterations=300, batch_size=None):
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

    # Make predictions and compute the metrics
    cluster_predictions = kmeans.fit_predict(x)
    silhouette_avg = silhouette_score(x, cluster_predictions)

    return {"labels": kmeans.labels_, "silhouette": silhouette_avg, "inertia": kmeans.inertia_}


def dbscan(x, eps, min_samples, metric="euclidean"):
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

    dbscan_clustering = DBSCAN(eps=eps,
                               min_samples=min_samples,
                               metric=metric,
                               n_jobs=-1)

    # Make predictions and compute the metrics
    cluster_predictions = dbscan_clustering.fit_predict(x)
    silhouette_avg = silhouette_score(x, cluster_predictions)

    return {"labels": dbscan_clustering.labels_, "silhouette": silhouette_avg}
