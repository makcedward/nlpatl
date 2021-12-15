from typing import List, Union
import numpy as np

try:
	import PIL
except ImportError:
	# No installation required if not using this function
	pass
try:
	import IPython
except ImportError:
	# No installation required if not using this function
	pass

from nlpatl.models.clustering import (
	Clustering,
	SkLearnClustering
)
from nlpatl.models.classification import (
	Classification,
	SkLearnClassification
)
from nlpatl.models.embeddings import (
	Embeddings,
	Transformers,
	TorchVision
)
from nlpatl.sampling import Sampling
from nlpatl.dataset import Dataset


class Learning:
	RETURN_TYPES = ['dict', 'object']
	DATA_TYPES = ['text', 'image', 'audio']

	def __init__(self, sampling: Sampling,
		embeddings_model: Embeddings = None, 
		clustering_model: Clustering = None, 
		classification_model: Classification = None, 
		multi_label: bool = False, 
		name: str = 'learning'):

		self.name = name
		self.multi_label = multi_label
		self.train_x = None
		self.train_y = None
		self.learn_indices = None
		self.learn_x = None
		self.learn_y = None
		self.unique_y = set()
		self.sampling = sampling
		self.embeddings_model = embeddings_model
		self.clustering_model = clustering_model
		self.classification_model = classification_model

	def validate(self, targets: List[str] = None):
		if not targets:
			return

		if 'embeddings' in targets:
			assert self.embeddings_model is not None, \
				'Embeddings model does not initialize yet.'
		if 'clustering' in targets:
			assert self.clustering_model is not None, \
				'Clustering model does not initialize yet.'
		if 'classification' in targets:
			assert self.classification_model is not None, \
				'Classification model does not initialize yet.'

	def get_return_object(self, d, return_type):
		assert return_type in self.RETURN_TYPES, \
				'`return_type` should be one of [`{}`] but not `{}`'.format(
					'`,`'.join(self.RETURN_TYPES), return_type)

		if return_type == 'dict':
			return d.__dict__
		elif return_type == 'object':
			return d

	def filter(self, data: Union[List[str], List[float], np.ndarray], 
		indices: np.ndarray) -> Union[List[str], List[float], np.ndarray]:

		if type(data) is np.ndarray:
			if type(indices) is list:
				indices = np.array(indices)
			return data[~np.isin(data, indices)]
		else:
			if type(indices) is np.ndarray:
				indices = indices.tolist()
			return [data[i] for i in range(len(data)) if i not in indices]

	def get_learn_data(self):
		"""
			Get all learn data points 

			:return: Learnt data points
			:rtype: Tuple of index list of int, x (:class:`numpy.ndarray`) 
				and y (:class:`numpy.ndarray`) 
		"""
		return self.learn_indices, self.learn_x, self.learn_y

	def clear_learn_data(self):
		"""
			Clear all learn data points 
		"""

		self.learn_indices = None
		self.learn_x = None
		self.learn_y = None

	def concatenate(self, data):
		if type(data[0] is list):
			return [c for d in data for c in d]
		if type(data[0] is np.ndarray):
			return np.concatenate(data)
		raise ValueError('Does not support {} data type yet'.format(type(data[0])))

	def get_annotated_data(self):
		"""
			Get all annotated data points (given and learnt data points)

			:return: Annotated data of training and learnt data points
			:rtype: Tuple of x (:class:`numpy.ndarray`) and y (:class:`numpy.ndarray`) 
		"""
		x = self.concatenate([d for d in [self.train_x, self.learn_x] if d])
		y = self.concatenate([d for d in [self.train_y, self.learn_y] if d])
		return x, y

	def init_unique_y(self, y):
		self.unique_y = set(list(y))

	def add_unique_y(self, y):
		y_data_type = type(next(iter(self.unique_y))) if self.unique_y else None

		if type(y) is list:
			for label in y:
				if y_data_type is int:
					label = int(label)
				self.unique_y.add(label)
		else:
			if y_data_type is int:
				y = int(y)
			self.unique_y.add(y)

	def learn(self, x: [List[str], List[int], List[float], np.ndarray], 
		y: [List[str], List[int]], include_leart_data: bool = True):
		"""
			Train the classification model.

			:param x: Raw data inputs. It can be text, number or numpy.
			:type x: list of string, int or float or :class:`np.ndarray`.
			:param y: Label of data inputs
			:type y: bool
			:param include_learn_data: Train the model whether including
				human annotated data and machine learning self annotated data.
				Default is True.
			:type include_learn_data: bool
		"""
		...

	def explore(self, 
		x: Union[List[str], List[int], List[float], np.ndarray], 
		return_type: str = 'dict', 
		num_sample: int = 10) -> Union[Dataset, dict]:
		"""
			Estimate the most valuable data points for annotation.

			:param x: Raw data inputs. It can be text, number or numpy (for image).
			:type x: list of string, int or float or :class:`np.ndarray`
			:param return_type: Data type of returning object. If `dict` is
				assigned. Return object is `dict`. Possible values are `dict`
				and `object`.
			:type return_type: str
			:param num_sample: Maximum number of data points for annotation.
			:type num_sample: int

			:return: The most valuable data points.
			:rtype: :class:`nlpatl.dataset.Dataset` objects or dict
		"""

		...	

	def educate(self, index: Union[str, int], 
		x: Union[str, int, float, np.ndarray],
		y: Union[str, int, List[str], List[int]]):
		"""
			Annotate data point. Only allowing annotate data point
			one by one. NOT batch.

			:param index: Index of data point.
			:type index: int
			:param x: Raw data input. It can be text, number or numpy (for image).
			:type x: string, int, float or :class:`np.ndarray`
			:param y: Label of data point
			:type y: string, int, list of string (multi-label case)
				or list or int (multi-label case)
		"""

		if type(x) in [str, int, float, list]:
			if self.learn_x:
				self.learn_x.append(x)
			else:
				self.learn_x = [x]
		elif type(x) is np.ndarray:
			if self.learn_x is not None:
				self.learn_x = np.concatenate((self.learn_x, np.array([x])), axis=0)
			else:
				self.learn_x = np.array([x])
		else:
			assert False, '{} data type does not support in `x` yet. '\
				'Only support `{}`'.format(type(x), '`,`'.join(
					['str', 'int', 'float', 'list', 'np.ndarray']))

		if self.unique_y:
			y_data_type = type(next(iter(self.unique_y)))
			if y_data_type is int:
				y = int(y)
		if self.learn_y:
			self.learn_y.append(y)
		else:
			self.learn_y = [y]
		self.add_unique_y(y)

		if self.learn_indices:
			self.learn_indices.append(index)
		else:
			self.learn_indices = [index]

	def show_in_notebook(self, result: Dataset, data_type: str = 'text'):
		i = 0
		while i < len(result.features):
			feature = result.features[i]
			_id = result.indices[i]
			metadata = '{}/{} Existing Label:{}\nID:{}\n'.format(
				i+1, len(result.features), 
				list(self.unique_y) if self.unique_y else [], _id)

			# Display
			if data_type == 'text':
				metadata += feature + '\n'
			elif data_type == 'image':
				IPython.display.display(PIL.Image.fromarray(feature))
			# elif data_type == 'audio':
				# IPython.display.display()

			label = input(metadata)
			if not (label or label.strip()):
				# Prompt same record again
				continue

			if self.multi_label:
				labels = label.split(',')
				labels = list(set([label for label in labels if label]))

				self.educate(_id, feature, labels)
				self.add_unique_y(labels)
			else:
				self.educate(_id, feature, label)
				self.add_unique_y(label)
			
			i += 1

	def explore_educate_in_notebook(self, 
		x: [List[str], List[int], List[float], np.ndarray],
		num_sample: int = 2, data_type: str = 'text'):
		"""
			Estimate the most valuable data points for annotation and
			annotate it in IPython Notebook. Executing `explore` function
			and `educate` function sequentially.

			:param x: Raw data inputs. It can be text, number or numpy (for image).
			:type x: list of string, int or float or :class:`np.ndarray`
			:param return_type: Data type of returning object. If `dict` is
				assigned. Return object is `dict`. Possible values are `dict`
				and `object`.
			:type return_type: str
			:param num_sample: Maximum number of data points for annotation.
			:type num_sample: int
			:param str data_type: Indicate the data format for displying in
				IPython Notebook. Possible values are `text` and `image`.
		"""

		assert data_type in self.DATA_TYPES, \
				'`data_type` should be one of [`{}`] but not `{}`'.format(
					'`,`'.join(self.DATA_TYPES), data_type)
		
		result = self.explore(x, num_sample=num_sample, return_type='object')
		self.show_in_notebook(result, data_type=data_type)
