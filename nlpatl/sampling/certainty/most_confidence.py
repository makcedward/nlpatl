from typing import Tuple
import numpy as np

from nlpatl.sampling import Sampling


class MostConfidenceSampling(Sampling):
	"""
		# https://markcartwright.com/files/wang2019active.pdf

		:param float threshold: Minimum probability of model prediction. Default
			value is 0.85
		:param str name: Name of this sampling
    """

	def __init__(self, threshold: float = 0.85,
		name: str = 'most_confidence_sampling'):
	
		super().__init__(name=name)

		self.threshold = threshold

	def sample(self, data: np.ndarray, 
		num_sample: int) -> Tuple[np.ndarray, np.ndarray]:

		"""
			:param Storage x: processed data
			:param int num_sample: Total number of sample for labeling
		"""

		num_node = min(num_sample, len(data))

		# Calucalte most confidence
		most_confidences = np.max(data, axis=1)
		indices = np.argwhere(most_confidences > self.threshold).flatten()
		indices = np.random.choice(indices, num_node)

		return indices, most_confidences[indices]
