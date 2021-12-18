from typing import List, Union
import numpy as np


class Dataset:
	def __init__(self, values: np.ndarray, indices: Union[List[int], np.ndarray] = None,
		inputs: Union[List[str], List[float], np.ndarray] = None, 
		features: Union[List[float], np.ndarray] = None, 
		labels: Union[List[str], List[int]] = None, groups: Union[List[str], List[int]] = None,
		name: str ='dataset'):
		self.name = name

		self.inputs = inputs
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

	def _filter(self, data, indices):
		if data is None:
			return data
		if type(data) is np.ndarray:
			return data[indices]
		else:
			return [data[i] for i in indices.tolist()]

	def keep(self, indices: np.ndarray):
		self.inputs = self._filter(self.inputs, indices)
		self.features = self._filter(self.features, indices)
		self.indices = self._filter(self.indices, indices)
		self.labels = self._filter(self.labels, indices)
		self.groups = self._filter(self.groups, indices)
		self.values = self._filter(self.values, indices)

	def __str__(self):
		return 'Groups: {}, Indices: {}, Features: {}, Values: {}'.format(
			self.groups, self.indices, self.inputs, self.features, self.values)
