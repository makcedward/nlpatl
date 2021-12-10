import unittest
import numpy as np

from nlpatl.sampling.uncertainty import MarginSampling


class TestSamplingMargin(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.data = np.array([
		    [0.01689184, 0.02989921, 0.92348951, 0.0158317, 0.01388775], 
		    [0.03950194, 0.06295594, 0.75774054, 0.08782477, 0.0519768], 
		    [0.00507078, 0.06731898, 0.850905, 0.06388271, 0.01282254],
		    [0.03204307, 0.01932809, 0.91326549, 0.01549605, 0.0198673],
		    [0.01181161, 0.00393428, 0.04627477, 0.92171903, 0.0162603],
		    [0.02010514, 0.00241422, 0.03849712, 0.92863317, 0.01035035],
		    [0.04326279, 0.01329769, 0.02284383, 0.88952749, 0.0310682],
		    [0.15085014, 0.0128402, 0.05903652, 0.74374557, 0.03352757],
		    [0.04319251, 0.02102466, 0.10190563, 0.75316733, 0.08070987],
		    [0.03870851, 0.70293962, 0.1727936, 0.04652781, 0.03903046],
		    [0.00521765, 0.89092692, 0.06196143, 0.03363766, 0.00825634],
		    [0.72885295, 0.02342087, 0.06129882, 0.14188246, 0.04454489],
		    [0.71795835, 0.02464577, 0.07842602, 0.1400593, 0.03891056]
		])

	def test_sample(self):
		expected_indices = np.array([9, 12, 11])
		
		num_sample = 3
		sampling = MarginSampling()
	
		indices, values = sampling.sample(self.data, num_sample=num_sample)

		assert indices is not None, 'No output'
		assert len(indices) == len(values), 'Sample size of return'
		assert np.array_equal(indices, expected_indices), \
			'Filtering incorrect result'