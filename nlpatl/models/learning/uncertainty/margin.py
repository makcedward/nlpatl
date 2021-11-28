from typing import List
import numpy as np

from nlpatl.models.learning.uncertainty.uncertainty import UncertaintySampling
from nlpatl.storage.storage import Storage


class MarginSampling(UncertaintySampling):
	def __init__(self, name: str = 'margin_sampling'):
		super().__init__(name=name)

	def keep_most_representative(self, data: Storage, num_sample: int) -> List[Storage]:
		# calculate margin difference between first and second highest probabilties
		values = np.partition(-data.values, 1, axis=1)
		values = -values[:, 0] + values[:, 1]
		data.values = values

		# Number of nodes may less than num_sample
		num_node = min(num_sample, len(data))

		indices = np.argpartition(values, num_node-1)[:num_node]
		data.filter(indices)

		return data
