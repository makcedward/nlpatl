import unittest
import numpy as np

from nlpatl.sampling.uncertainty import MismatchSampling


class TestSamplingMismatch(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.data1 = ['2', '1', '1', '2', '1', '2', '1', '2', '1', '2', '2', '1']
		cls.data2 = ['1', '1', '1', '1', '1', '1', '2', '2', '2', '2', '2', '2']

	def test_sample(self):
		expected_indices = np.array([0, 3, 5, 6, 8, 11])

		num_sample = len(self.data1)
		sampling = MismatchSampling()
	
		indices, values = sampling.sample(
			self.data1, self.data2, num_sample=num_sample)
		indices.sort()

		assert indices is not None, 'No output'
		assert np.array_equal(indices, expected_indices), \
			'Filtering incorrect result'