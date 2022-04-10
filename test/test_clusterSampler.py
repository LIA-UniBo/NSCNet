from unittest import TestCase

from architectures.nscnet.uniform_cluster_sampler import ClusterSampler
import numpy as np


class TestClusterSampler(TestCase):

    def test_sample_allElementsUsedInAEpoch(self):
        """
        Note: this test is valid only if the flag use_all_dataset is set to True
        """

        # Arrange
        X = ["a", "b", "c", "d", "e", "f"]
        y = [0, 1, 1, 2, 1, 3]
        used_x = set([])

        batch_size = 3

        # Act
        cluster_sampler = ClusterSampler(6, batch_size, use_all_dataset=True)
        cluster_sampler.segment_clusters(y)
        for a in range(int(len(y) / batch_size)):
            batch_x, batch_y = cluster_sampler.sample(X, y)
            used_x.update(batch_x.tolist())

        # Assert
        np.testing.assert_equal(len(used_x), len(X))

    def test_sample_noDuplicatesInABatch(self):
        """
        Note: this test is valid only if the flag use_all_dataset is set to True
        """

        # Arrange
        X = ["a", "b", "c", "d", "e", "f"]
        y = [0, 1, 1, 2, 1, 3]

        batch_size = 3
        batch_lenght = 0
        batch_lenght_unique = 0

        used_x = []

        # Act
        cluster_sampler = ClusterSampler(len(X), batch_size, use_all_dataset=True)
        cluster_sampler.segment_clusters(y)

        for a in range(10000):
            batch_x, batch_y = cluster_sampler.sample(X, y)
            used_x.extend(batch_x)
            batch_lenght += len(batch_x)
            batch_lenght_unique += len(np.unique(batch_x))
            cluster_sampler.segment_clusters(y)

        # Assert
        np.testing.assert_equal(batch_lenght, batch_lenght_unique)

    def test_sample_allNonEmptyClustersUsed(self):

        # Arrange
        X = np.array([])
        y = np.array([])
        n_clusters = 10

        for cluster in range(n_clusters):

            # Number of elements that are assigned to the current cluster
            # NOTE: the test fails very frequently if the number of elements
            # is too small (e.g., < 5 in this dummy test)
            n_elements = np.random.randint(0, 10)

            # Probability used for simulating empty clusters
            p = np.random.randint(0, 10)
            if p <= 2:
                print(f'\nremoving cluster {cluster} - prob: {p}')
                n_elements = 0

            X = np.append(X, np.random.randint(100, size=n_elements))
            y = np.append(y, np.repeat(cluster, n_elements)).astype(np.int32)

        batch_size = 3
        actual_non_empty_clusters = np.unique(y)

        # Act
        cluster_sampler = ClusterSampler(len(X), 3, use_all_dataset=False)
        cluster_sampler.segment_clusters(y)

        used_clusters = []
        for elem in range(0, len(X), batch_size):
            batch_x, batch_y = cluster_sampler.sample(X, y)
            used_clusters.extend(batch_y)

        used_clusters = np.unique(used_clusters)

        # Assert
        np.testing.assert_equal(actual_non_empty_clusters, used_clusters)
