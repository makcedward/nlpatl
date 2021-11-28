from typing import List
from scipy.stats import entropy
import numpy as np

from nlpatl.models.learning.uncertainty.uncertainty import UncertaintySampling
from nlpatl.storage.storage import Storage


class EntropySampling(UncertaintySampling):
	def __init__(self, name: str = 'entropy_sampling'):
		super().__init__(name=name)

	def keep_most_representative(self, data: Storage, num_sample: int) -> List[Storage]:
		# calcualte entropy
		values = entropy(data.values, axis=1)
		data.values = values

		# Number of nodes may less than num_sample
		num_node = min(num_sample, len(data))

		indices = np.argpartition(-data.values, num_node-1)[:num_node]
		data.filter(indices)

		return data
