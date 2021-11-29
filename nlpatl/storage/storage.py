from typing import List
import numpy as np


class Storage:
	def __init__(self, values: np.ndarray, indices: [List[int], np.ndarray] = None,
		features: [List[str], List[float], np.ndarray] = None, 
		labels: [List[str], List[int]] = None, groups: [List[str], List[int]] = None,
		name: str ='storage'):
		self.name = name

		self.features = features
		self.indices = np.array([i for i in range(len(values))]) if indices is None else indices
		self.groups = groups
		self.labels = labels
		self.values = values

	def __len__(self):
		return len(self.indices)

	def filter(self, indices: np.ndarray):
		indices = self.indices[~np.isin(self.indices, indices)]
		self.keep(indices)

	def keep(self, indices: np.ndarray):
		if self.features is not None:
			if type(self.features) is np.ndarray:
				self.features = self.features[indices]
			else:
				self.features = [self.features[i] for i in indices.tolist()]

		if self.indices is not None:
			self.indices = self.indices[indices]

		if self.labels:
			self.labels = [self.labels[i] for i in indices.tolist()]

		if self.groups:
			self.groups = [self.groups[i] for i in indices.tolist()]
				
		if self.values is not None:
			self.values = self.values[indices]

	def __str__(self):
		return 'Groups: {}, Indices: {}, Features: {}, Values: {}'.format(
			self.groups, self.indices, self.features, self.values)
