from typing import Tuple
import numpy as np

from nlpatl.sampling import Sampling


class FarthestSampling(Sampling):
	"""
		Sampling data points according to the distances of cluster centriod. Picking n 
			farthest data points per number of cluster. 
			http://zhaoshuyang.com/static/documents/MAL2.pdf

		:param name: Name of this sampling
		:type name: str
    """
    
	def __init__(self, name: str = 'farthest_sampling'):
		super().__init__(name=name)

	def sample(self, data: np.ndarray, groups: np.ndarray,
		num_sample: int) -> Tuple[np.ndarray, np.ndarray]:

		to_be_keep_indices = []
		for group in np.unique(groups):
			indices = np.where(groups == group)[0]
			values = data[indices]

			num_node = min(num_sample, len(indices))

			# Get farthest distances
			local_indices = np.argpartition(-values, num_node-1)[:num_node]
			to_be_keep_indices.append(indices[local_indices])

		return np.concatenate(to_be_keep_indices), None
		