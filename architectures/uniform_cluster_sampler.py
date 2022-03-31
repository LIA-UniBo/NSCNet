import numpy as np

class ClusterSampler:

    def __init__(self, n_clusters, batch_size):

        self.n_clusters = n_clusters
        self.batch_size = batch_size

        self.clusters_indices = [[] for _ in range(n_clusters)]

        self.non_empty_clusters = []

    def segment_clusters(self, pseudo_labels):

        for index, cluster in enumerate(pseudo_labels):
            self.clusters_indices[cluster].append(index)

        self.update_non_empty_clusters()

    def update_non_empty_clusters(self):

        non_empty_clusters = list(range(self.n_clusters))
        for cluster, cluster_indices in enumerate(self.clusters_indices):
            if len(cluster_indices)==0:
                non_empty_clusters.remove(cluster)

        self.non_empty_clusters = non_empty_clusters

    def sample(self, x, y):

        cluster_types = np.random.choice(self.non_empty_clusters, size=self.batch_size)

        batch_indices = []
        for cluster_type in cluster_types:
            cluster_indices = self.clusters_indices[cluster_type]
            random_cluster_index = np.random.choice(cluster_indices)
            batch_indices.append(random_cluster_index)

        batch_x = np.take(x, batch_indices, axis=0)
        batch_y = np.take(y, batch_indices, axis=0)

        return batch_x, batch_y
