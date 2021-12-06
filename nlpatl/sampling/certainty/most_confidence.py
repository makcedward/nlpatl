from scipy.stats import entropy
import numpy as np

from nlpatl.sampling import Sampling
from nlpatl.storage import Storage


class MostConfidenceSampling(Sampling):
	"""
		# https://markcartwright.com/files/wang2019active.pdf

		:param float threshold: Minimum probability of model prediction. Default
			value is 0.85
		:param str name: Name of this sampling

		>>> import nlpatl.learning as nl
		>>> model = nl.LeastConfidenceLearning()
    """

	def __init__(self, threshold: float = 0.85,
		name: str = 'most_confidence_sampling'):
	
		super().__init__(name=name)

		self.threshold = threshold

	def sample(self, data: Storage, num_sample: int) -> Storage:
		num_node = min(num_sample, len(data))

		# Calucalte most confidence
		most_confidences = np.max(data.values, axis=1)
		indices = np.argwhere(most_confidences > self.threshold).flatten()
		indices = np.random.choice(indices, num_node)

		data.keep(indices)
		
		# Replace probabilies by most_confidences
		data.values = most_confidences[indices]

		return data
