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

from nlpatl.models.classification.classification import Classification
from nlpatl.models.clustering.clustering import Clustering
from nlpatl.models.embeddings.embeddings import Embeddings
from nlpatl.models.embeddings.transformers import Transformers
from nlpatl.models.embeddings.torchvision import TorchVision
from nlpatl.models.clustering.sklearn_clustering import (
	SkLearnClustering
)
from nlpatl.models.classification.sklearn_classification import (
	SkLearnClassification
)
from nlpatl.storage.storage import Storage


class Learning:
	RETURN_TYPES = ['dict', 'object']
	DATA_TYPES = ['text', 'image', 'audio']

	def __init__(self, multi_label: bool = False, 
		embeddings_model: Embeddings = None, 
		clustering_model: Clustering = None, 
		classification_model: Classification = None, 
		name: str = 'learning'):

		self.name = name
		self.multi_label = multi_label
		self.train_x = None
		self.train_y = None
		self.learn_id = None
		self.learn_x = None
		self.learn_y = None
		self.unique_y = set()
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
			return data[indices]
		else:
			return [data[i] for i in indices.tolist()]

	def get_learnt_data(self):
		return self.learn_id, self.learn_x, self.learn_y

	def init_unique_y(self, y):
		self.unique_y = set(list(y))

	def add_unique_y(self, y):
		y_data_type = type(next(iter(self.unique_y))) if self.unique_y else None

		if type(y) is list:
			for label in y:
				if y_data_type == int:
					label = int(label)
				self.unique_y.add(label)
		else:
			if y_data_type == int:
				y = int(y)
			self.unique_y.add(y)

	def train(self, x: object, y: object):
		...

	def keep_most_valuable(self, data: Storage, num_sample: int) -> Storage: 
		"""
			:param Storage x: processed data
			:param int num_sample: Total number of sample for labeling
			
			>>> model.keep_most_valuable(x=x)
		"""
		...

	def learn(self, x: [List[str], List[int], List[float], np.ndarray], 
		y: [List[str], List[int]], include_leart_data: bool = True):
		...

	def explore(self, x: List[str], return_type: str = 'dict', 
		num_sample: int = 2) -> List[object]:
		...	

	def educate(self, _id: Union[str, int], 
		x: Union[str, int, float, List[float], np.ndarray],
		y: Union[str, int, List[str], List[int]]):
		"""
			Expect label 1 record only. 
			_id: id
			x: List of floar or np.ndarray used for vector
			y: List is designed for multi-label scenario
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

		if self.learn_y:
			self.learn_y.append(y)
		else:
			self.learn_y = [y]
		self.add_unique_y(y)

		if self.learn_id:
			self.learn_id.append(_id)
		else:
			self.learn_id = [_id]

	def explore_educate_in_notebook(self, 
		x: [List[str], List[int], List[float], np.ndarray],
		num_sample: int = 2, data_type: str = 'text'):

		assert data_type in self.DATA_TYPES, \
				'`data_type` should be one of [`{}`] but not `{}`'.format(
					'`,`'.join(self.DATA_TYPES), data_type)
		
		result = self.explore(x, num_sample=num_sample, return_type='object')
		i = 0
		while i < len(result.features):
			feature = result.features[i]
			_id = result.indices[i]
			metadata = '{}/{} Existing Label:{}\nID:{}\n'.format(
				i+1, len(result.features), 
				list(self.unique_y) if self.unique_y else [], _id)

			# Display on notebook
			if data_type == 'text':
				metadata += feature
			elif data_type == 'image':
				IPython.display.display(PIL.Image.fromarray(feature))
			# elif data_type == 'audio':
				# TODO: externalize
				# import IPython
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