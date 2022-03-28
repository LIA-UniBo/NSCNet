from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN

CLUSTERING_METHODS = ["kmeans", "dbscan"]

def k_means(x, n_clusters, max_iterations=300, batch_size=None):

    #Note: documentation suggests a batch size of 256 * number of cores to exploit parallelism

    if batch_size is None:
        kmeans = KMeans(n_clusters=n_clusters,
                        max_iter=max_iterations,
                        random_state=0).fit(x)

        return {"labels": kmeans.labels_, "inertia": kmeans.inertia_}

    else:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                                random_state=0,
                                batch_size=batch_size,
                                max_iter=max_iterations).fit(x)

        return {"labels": kmeans.labels_, "inertia": kmeans.inertia_}

def dbscan(x, eps, min_sample, metric="euclidean"):

    dbscan_clustering = DBSCAN(eps=eps,
                        min_samples=min_samples,
                        metric=metric,
                        n_jobs=-1).fit(x)

    return {"labels":dbscan_clustering.labels_}
