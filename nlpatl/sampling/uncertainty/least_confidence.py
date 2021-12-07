from typing import Tuple
import numpy as np

from nlpatl.sampling import Sampling


class LeastConfidenceSampling(Sampling):
	"""
		:param str name: Name of this sampling
    """

	def __init__(self, name: str = 'least_confidence_sampling'):
		super().__init__(name=name)

	def sample(self, data: np.ndarray, 
		num_sample: int) -> Tuple[np.ndarray, np.ndarray]:

		num_node = min(num_sample, len(data))

		# Calucalte least confidence
		least_confidences = 1 - np.max(data, axis=1)
		indices = np.argpartition(-least_confidences, num_node-1)[:num_node]

		return indices, least_confidences[indices]
