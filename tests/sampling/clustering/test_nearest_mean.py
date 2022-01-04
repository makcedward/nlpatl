import unittest
import numpy as np

from nlpatl.sampling.clustering import NearestMeanSampling


class TestSamplingNearestMean(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.array(
            [
                1.75512648,
                2.4474237,
                2.72141409,
                2.38541532,
                2.32101798,
                2.29712701,
                1.84427822,
                1.88990271,
                2.42033386,
                2.6152091,
                2.6152091,
                2.75662065,
                3.37551308,
                2.52874994,
                2.53957653,
            ]
        )
        cls.groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3])

    def test_sample(self):
        expected_indices = np.array([0, 6, 9, 13])

        num_sample = 1
        sampling = NearestMeanSampling()
        indices, _ = sampling.sample(self.data, self.groups, num_sample=num_sample)

        assert indices is not None, "No output"
        assert np.array_equal(indices, expected_indices), "Filtering incorrect result"
