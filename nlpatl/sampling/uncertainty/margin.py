from typing import Tuple
import numpy as np

from nlpatl.sampling import Sampling


class MarginSampling(Sampling):
	"""
		Sampling data points according to the margin confidence. Pick the lowest
			probabilies difference between the highest class and second higest class.

		:param name: Name of this sampling
		:type name: str
    """

	def __init__(self, name: str = 'margin_sampling'):	
		super().__init__(name=name)		

	def sample(self, data: np.ndarray, 
		num_sample: int) -> Tuple[np.ndarray, np.ndarray]:

		num_node = min(num_sample, len(data))

		# Calculate margin difference between first and second highest probabilties
		margin_diffs = np.partition(-data, 1, axis=1)
		margin_diffs = -margin_diffs[:, 0] + margin_diffs[:, 1]

		indices = np.argpartition(margin_diffs, num_node-1)[:num_node]
		return indices, margin_diffs[indices]
