# This module is responsible for sampling batches so to have samples uniformly distributed across the clusters

import numpy as np


class ClusterSampler:

    def __init__(self, n_clusters, batch_size, use_all_dataset=False):
        """
        Init some class parameters.

        :param n_clusters: the number of possible clusters
        :param batch_size: the number of samples in a batch
        :param use_all_dataset: this flag should be set to True only under the assumption that the dataset
                                balanced, otherwise some batches could be representative only for those clusters
                                that contain many elements
        """

        self.use_all_dataset = use_all_dataset
        self.n_clusters = n_clusters
        self.batch_size = batch_size

        # This list contains K=number_of_clusters lists containing the positions (indices) of all the samples in the
        # dataset belonging to that cluster
        self.clusters_indices = [[] for _ in range(n_clusters)]

        # This list contains the clusters that are not empty
        self.non_empty_clusters = []

        # The following list:
        #  - contains the element indices that have not been already used during an epoch.
        #  - It is reset every time the segment_cluster method is called.
        #  - Used only if use_all_dataset parameter is True.
        self.usable_element_indices = []

    def segment_clusters(self, pseudo_labels):

        self.usable_element_indices.clear()

        # Save the indices of each sample in the dataset to the corresponding cluster list
        for index, cluster in enumerate(pseudo_labels):
            if self.use_all_dataset:
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

            # choose a random sample of the dataset with the specified cluster
            random_cluster_index = np.random.choice(cluster_indices)

            # If forcing the use of the entire dataset, then check that the element has not been used yet.
            # Otherwise, take another element randomly.
            if self.use_all_dataset:
                if random_cluster_index not in self.usable_element_indices:
                    random_cluster_index = np.random.choice(self.usable_element_indices)

                self.usable_element_indices.remove(random_cluster_index)

            batch_indices.append(random_cluster_index)

            '''
            # Add the element to the batch
            if random_cluster_index in self.usable_element_indices or not self.use_all_dataset:
                batch_indices.append(random_cluster_index)
                self.usable_element_indices.remove(random_cluster_index)
            else:
                random_cluster_index = np.random.choice(self.usable_element_indices)
                batch_indices.append(random_cluster_index)
                self.usable_element_indices.remove(random_cluster_index)
            '''

        # Create the batch taking the samples from the collected indices
        batch_x = np.take(x, batch_indices, axis=0)
        batch_y = np.take(y, batch_indices, axis=0)

        return batch_x, batch_y
