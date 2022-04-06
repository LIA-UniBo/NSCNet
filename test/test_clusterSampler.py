from unittest import TestCase

from architectures.uniform_cluster_sampler import ClusterSampler
import numpy as np


class TestClusterSampler(TestCase):
    def test_sample_allElementsUsedInAEpoch(self):

        # Arrange
        x = ["a", "b", "c", "d", "e", "f"]
        y = [0, 1, 1, 2, 1, 3]
        used_x = set([])

        batch_size = 3

        # Act
        cluster_sampler = ClusterSampler(6, batch_size)
        cluster_sampler.segment_clusters(y)
        for a in range(int(len(y) / batch_size)):
            batch_x, batch_y = cluster_sampler.sample(x, y)
            used_x.update(batch_x.tolist())

        # Assert
        np.testing.assert_equal(len(used_x), len(x))

    def test_sample_noDuplicatesInABatch(self):

        # Arrange
        x = ["a", "b", "c", "d", "e", "f"]
        y = [0, 1, 1, 2, 1, 3]

        batch_size = 3
        batch_lenght = 0
        batch_lenght_unique = 0

        used_x = []

        # Act
        cluster_sampler = ClusterSampler(len(x), batch_size)
        cluster_sampler.segment_clusters(y)

        for a in range(10000):
            batch_x, batch_y = cluster_sampler.sample(x, y)
            used_x.extend(batch_x)
            batch_lenght += len(batch_x)
            batch_lenght_unique += len(np.unique(batch_x))
            cluster_sampler.segment_clusters(y)

        # Assert
        np.testing.assert_equal(batch_lenght, batch_lenght_unique)
