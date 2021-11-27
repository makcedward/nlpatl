from typing import List
import numpy as np


class Storage:
	def __init__(self, features: [List[str], List[float], np.ndarray] = None, 
		indices: [List[int], np.ndarray] = -1, 
		group: [str, int] = None, 
		labels: [List[str], List[int]] = None, 
		values: np.ndarray = None, name: str ='storage'):
		self.name = name

		self.features = features
		self.indices = indices
		self.group = group
		self.labels = labels
		self.values = values

	def __len__(self):
		return len(self.indices)

	def filter(self, indices: np.ndarray):
		if self.features is not None:
			if type(self.features) is np.ndarray:
				self.features = self.features[indices]
			else:
				self.features = [self.features[i] for i in indices.tolist()]

		if self.indices is not None:
			self.indices = self.indices[indices]

		if self.labels is not None:
			if type(self.features) is np.ndarray:
				self.labels = self.labels[indices]
			else:
				self.labels = [self.labels[i] for i in indices.tolist()]
				
		if self.values is not None:
			self.values = self.values[indices]

	def __str__(self):
		return 'Group: {}, Indices: {}'.format(self.group, self.indices)
