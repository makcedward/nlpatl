from typing import Tuple
import numpy as np

from nlpatl.sampling import Sampling


class NearestMeanSampling(Sampling):
	"""
		Sampling data points according to the distances of cluster centriod. Picking n
			nearest data points per number of cluster.

		:param name: Name of this sampling
		:type name: str
    """
    
	def __init__(self, name: str = 'nearest_mean_sampling'):
		super().__init__(name=name)

	def sample(self, data: np.ndarray, groups: np.ndarray,
		num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
	
		to_be_filter_indices = []
		for group in np.unique(groups):
			indices = np.where(groups == group)[0]
			values = data[indices]

			num_node = min(num_sample, len(indices))

			# get first n shortest distances
			local_indices = values.argpartition(num_node-1)[:num_node]
			to_be_filter_indices.append(indices[local_indices])

		return np.concatenate(to_be_filter_indices), None
