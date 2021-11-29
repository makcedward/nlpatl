from typing import List
from scipy.stats import entropy
import numpy as np

from nlpatl.models.learning.uncertainty.uncertainty import UncertaintySampling
from nlpatl.storage.storage import Storage


class EntropySampling(UncertaintySampling):
	def __init__(self, name: str = 'entropy_sampling'):
		super().__init__(name=name)

	def keep_most_valuable(self, data: Storage, 
		num_sample: int) -> Storage:

		num_node = min(num_sample, len(data))

		# Calucalte entropy
		entropies = entropy(data.values, axis=1)
		indices = np.argpartition(-entropies, num_node-1)[:num_node]

		data.keep(indices)
		
		# Replace probabilies by entropies
		data.values = entropies[indices]

		return data
