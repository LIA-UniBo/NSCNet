from unittest import TestCase

import numpy as np

from architectures.common.spec_augmentation import time_warp


class TestSpecAugmentation(TestCase):
    def test_time_warp(self):
        size = 10

        x = np.zeros((10, 10), dtype=np.float32)

        for i in range(0, size):
            elem = np.arange(i*size+1, i*size+1+size, dtype=np.float32)
            x[i, :] = elem

        x = np.expand_dims(x, axis=-1)

        time_warp(x, 4)

        print()