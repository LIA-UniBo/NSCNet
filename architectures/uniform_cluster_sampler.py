# This module is responsible for sampling batches so to have samples uniformly distributed across the clusters

import numpy as np


class ClusterSampler:

    def __init__(self, n_clusters, batch_size):

        self.n_clusters = n_clusters
        self.batch_size = batch_size

        # This list contains K=number_of_clusters lists containing the positions (indices) of all the samples in the
        # dataset belonging to that cluster
        self.clusters_indices = [[] for _ in range(n_clusters)]

        # This list contains the clusters that are not empty
        self.non_empty_clusters = []

        # This list contains the element indices that have not been already used during an epoch
        # It is reset every time the segment_cluster method is called
        # TODO: to be discussed
        self.usable_element_indices = []

    def segment_clusters(self, pseudo_labels):

        self.usable_element_indices.clear()

        # Save the indices of each sample in the dataset to the corresponding cluster list
        for index, cluster in enumerate(pseudo_labels):
            self.usable_element_indices.append(index)
            self.clusters_indices[cluster].append(index)

        self.update_non_empty_clusters()

    def update_non_empty_clusters(self):

        # List the values of the clusters that are not empty

        non_empty_clusters = list(range(self.n_clusters))
        for cluster, cluster_indices in enumerate(self.clusters_indices):
            if len(cluster_indices) == 0:
                non_empty_clusters.remove(cluster)

        self.non_empty_clusters = non_empty_clusters

    def sample(self, x, y):

        """
        Take the entire dataset and create a batch sampling with a uniform
        distribution across the clusters (pseudo_labels), ignoring the empty
        clusters.
        """

        # Choose N=batch-size pseudo_labels(excluding empty clusters) with a random uniform distribution
        cluster_types = np.random.choice(self.non_empty_clusters, size=self.batch_size)

        # Collect the indices of the values in the dataset to put inside the batch
        batch_indices = []
        for cluster_type in cluster_types:
            cluster_indices = self.clusters_indices[cluster_type]
            random_cluster_index = np.random.choice(
                cluster_indices)  # choose a random sample of the dataset with the specified cluster
            if random_cluster_index in self.usable_element_indices:
                batch_indices.append(random_cluster_index)
                self.usable_element_indices.remove(random_cluster_index)
            else:
                # print('Element already used. Taking another one randomly')
                random_cluster_index = np.random.choice(self.usable_element_indices)
                batch_indices.append(random_cluster_index)
                self.usable_element_indices.remove(random_cluster_index)

        # Create the batch taking the samples from the collected indices
        batch_x = np.take(x, batch_indices, axis=0)
        batch_y = np.take(y, batch_indices, axis=0)

        return batch_x, batch_y
