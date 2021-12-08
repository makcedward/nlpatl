from typing import Tuple
from scipy.stats import entropy
import numpy as np

from nlpatl.sampling import Sampling


class EntropySampling(Sampling):
	"""
		Sampling data points according to the entropy. Pick the highest N data points

		:param name: Name of this sampling
		:type name: str
    """

	def __init__(self, name: str = 'entropy_sampling'):
		super().__init__(name=name)

	def sample(self, data: np.ndarray, 
		num_sample: int) -> Tuple[np.ndarray, np.ndarray]:

		num_node = min(num_sample, len(data))

		# Calucalte entropy
		entropies = entropy(data, axis=1)
		indices = np.argpartition(-entropies, num_node-1)[:num_node]

		return indices, entropies[indices]
