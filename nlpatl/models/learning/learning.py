from typing import List
import numpy as np

from nlpatl.models.embeddings.transformers import Transformers
from nlpatl.models.clustering.sklearn_clustering import (
	SkLearnClustering
)
from nlpatl.models.classification.sklearn_classification import (
	SkLearnClassification
)
from nlpatl.storage.storage import Storage


class Learning:
	RETURN_TYPES = ['dict', 'object']

	def __init__(self, x: [List[str], List[float], np.ndarray] = None,
		y: [List[str], List[int], np.ndarray] = None,
		name: str = 'learning'):

		self.name = name
		self.train_x = x
		self.train_y = y
		self.learn_x = None
		self.learn_y = None
		self.embeddings_model = None
		self.clustering_model = None
		self.classification_model = None

	def init_embeddings_model(self, model_name_or_path: str = 'bert-base-uncased',
		model: object = None, batch_size: int = 32, padding: bool = False, 
		truncation: bool = False, return_tensors: str = None):

		if model:
			self.embeddings_model = model
		else:
			self.embeddings_config = {
				'model_name_or_path': model_name_or_path,
				'batch_size': batch_size,
				'padding': padding,
				'truncation': truncation,
				'return_tensors': return_tensors
			}
			self.embeddings_model = Transformers(**self.embeddings_config)

	def init_clustering_model(self, model_name: str = 'kmeans', model: object = None, 
		model_config: dict = {}):

		possible_models = SkLearnClustering.get_mapping().keys()

		if model:
			self.model = model
		elif model_name in possible_models:
			self.clustering_model = SkLearnClustering(model_name, model_config)
			self.model_config = model_config
		else:
			raise ValueError('`{}` does not support. Supporting {} only'.format(
				model_name, '`' + '`'.join(possible_models) + '`'))

	def init_classification_model(self, model_name: str = 'logistic_regression',
		model: object = None, model_config: dict = {}):
		# model: object = None, batch_size: int = 32, model_config: dict = {}):

		possible_models = SkLearnClassification.get_mapping().keys()

		if model:
			self.model = model
		elif model_name in possible_models:
			self.classification_model = SkLearnClassification(model_name, model_config)
			self.model_config = model_config
		else:
			raise ValueError('`{}` does not support. Supporting {} only'.format(
				model_name, '`' + '`'.join(possible_models) + '`'))
		
		# self.model_config['batch_size'] = batch_size

	def validate(self, targets: List[str] = None):
		if not targets:
			return

		if 'embeddings' in targets:
			assert self.embeddings_model is not None, \
				'Embeddings model does not initialize yet. Run `init_embeddings_model` first'
		if 'clustering' in targets:
			assert self.clustering_model is not None, \
				'Clustering model does not initialize yet. Run `init_clusting_model` first'
		if 'classification' in targets:
			assert self.classification_model is not None, \
				'Classification model does not initialize yet. Run `init_classification_model` first'

	def get_return_object(self, d, return_type):
		assert return_type in self.RETURN_TYPES, \
				'`return_type` should be one of [`{}`] but not `{}`'.format(
					'`,`'.join(self.RETURN_TYPES), return_type)

		if return_type == 'dict':
			return d.__dict__
		elif return_type == 'object':
			return d

	def filter(self, data: [List[str], List[float], np.ndarray], 
		indices: np.ndarray) -> [List[str], List[float], np.ndarray]:

		if type(data) is np.ndarray:
			return data[indices]
		else:
			return [data[i] for i in indices.tolist()]

	def get_learnt_data(self):
		return self.learn_x, self.learn_y

	def get_unique_labels(self):
		labels = np.unique(np.concatenate([
			self.train_y if self.train_y else [], 
			self.learn_y if self.learn_y else [], 
		]))
		return set(labels.tolist())

	def train(self, x: object, y: object):
		...

	def keep_most_valuable(self, data: Storage, num_sample: int) -> Storage: 
		...

	def learn(self, x: [List[str], List[int], List[float], np.ndarray], 
		y: [List[str], List[int]], include_leart_data: bool = True):
		...

	def explore(self, x: List[str], return_type: str = 'dict', 
		num_sample: int = 2) -> List[object]:
		...

	def explore_educate_in_notebook(self, 
		x: [List[str], List[int], List[float], np.ndarray],
		num_sample: int = 2):
		
		result = self.explore(x, num_sample=num_sample, return_type='object')

		possible_labels = self.get_unique_labels()
		for i, feature in enumerate(result.features):
			label = input('{}/{}:{}\n\nExisting Label:{}\n'.format(
				i+1, len(result.features), feature, possible_labels if possible_labels else []))
			self.educate(feature, label)
			possible_labels.add(label)


	def educate(self, x: [str, int, float, List[float], np.ndarray],
		y: [str, int, List[str], List[int]]):
		"""
			Expect label 1 record only. 
			x: List of floar or np.ndarray used for vector
			y: List is designed for multi-label scenario
		"""

		if type(x) in [str, int, float, list]:
			if self.learn_x:
				self.learn_x.append(x)
			else:
				self.learn_x = [x]
		elif type(x) is np.ndarray:
			if self.learn_x:
				self.learn_x = np.hstack((self.learn_x, x))
			else:
				self.learn_x = np.array(x)
		else:
			assert False, '{} data type does not support in `x` yet. '\
				'Only support `{}`'.format(type(x), '`,`'.join(
					['str', 'int', 'float', 'list', 'np.ndarray']))

		if self.learn_y:
			self.learn_y.append(y)
		else:
			self.learn_y = [y]
