from typing import List
import numpy as np

from nlpatl.models.learning.uncertainty.uncertainty import UncertaintySampling
from nlpatl.storage.storage import Storage


class MarginSampling(UncertaintySampling):
	def __init__(self, name: str = 'margin_sampling'):
		super().__init__(name=name)		

	def keep_most_valuable(self, data: Storage, 
		num_sample: int) -> Storage:

		num_node = min(num_sample, len(data))

		# Calculate margin difference between first and second highest probabilties
		margin_diffs = np.partition(-data.values, 1, axis=1)
		margin_diffs = -margin_diffs[:, 0] + margin_diffs[:, 1]

		indices = np.argpartition(margin_diffs, num_node-1)[:num_node]

		data.keep(indices)
		
		# Replace probabilies by margin differences
		data.values = margin_diffs[indices]

		return data
