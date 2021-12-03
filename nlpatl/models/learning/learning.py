from typing import List
import numpy as np

try:
	import PIL
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

	def __init__(self, x: [List[str], List[float], np.ndarray] = None,
		y: [List[str], List[int], np.ndarray] = None,
		multi_label: bool = False, embeddings_model: Embeddings = None, 
		clustering_model: Clustering = None, 
		classification_model: Classification = None, 
		name: str = 'learning'):

		self.name = name
		self.multi_label = multi_label
		self.train_x = x
		self.train_y = y
		self.learn_id = None
		self.learn_x = None
		self.learn_y = None
		self.embeddings_model = embeddings_model
		self.clustering_model = clustering_model
		self.classification_model = classification_model

	# # TODO: revamp embs for image, text and audio
	# def init_embeddings_model(self, model_name_or_path: str = 'bert-base-uncased',
	# 	batch_size: int = 16, padding: bool = False, truncation: bool = False, 
	# 	return_tensors: str = None):

	# 	self.embeddings_config = {
	# 		'model_name_or_path': model_name_or_path,
	# 		'batch_size': batch_size,
	# 		'padding': padding,
	# 		'truncation': truncation,
	# 		'return_tensors': return_tensors
	# 	}
	# 	self.embeddings_model = Transformers(**self.embeddings_config)

	# def init_image_embeddings_model(self, model_name_or_path: str,
	# 	transform = None, batch_size: int = 16, 
	# 	model_config: dict = None):

	# 	self.embeddings_config = {
	# 		'model_name_or_path': model_name_or_path,
	# 		'batch_size': batch_size,
	# 		'transform': transform,
	# 		'model_config': model_config
	# 	}
	# 	self.embeddings_model = TorchVision(**self.embeddings_config)

	def init_clustering_model(self, model_name: str = 'kmeans', 
		model_config: dict = {}):

		possible_models = SkLearnClustering.get_mapping().keys()

		if model_name in possible_models:
			self.clustering_model = SkLearnClustering(model_name, model_config)
			self.model_config = model_config
		else:
			raise ValueError('`{}` does not support. Supporting {} only'.format(
				model_name, '`' + '`'.join(possible_models) + '`'))

	def init_classification_model(self, model_name: str = 'logistic_regression',
		model_config: dict = {}):

		possible_models = SkLearnClassification.get_mapping().keys()

		if model_name in possible_models:
			self.classification_model = SkLearnClassification(model_name, model_config)
			self.model_config = model_config
		else:
			raise ValueError('`{}` does not support. Supporting {} only'.format(
				model_name, '`' + '`'.join(possible_models) + '`'))

	def validate(self, targets: List[str] = None):
		if not targets:
			return

		if 'embeddings' in targets:
			assert self.embeddings_model is not None, \
				'Embeddings model does not initialize yet. Run `init_text_embeddings_model`' \
				'init_image_embeddings_model` or `init_audio_embeddings_model first'
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
		return self.learn_id, self.learn_x, self.learn_y

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

	def educate(self, _id: [str, int], 
		x: [str, int, float, List[float], np.ndarray],
		y: [str, int, List[str], List[int]]):
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

		possible_labels = self.get_unique_labels()
		i = 0
		while i < len(result.features):
			feature = result.features[i]
			_id = result.indices[i]
			metadata = '{}/{} Existing Label:{}\nID:{}\n'.format(
				i+1, len(result.features), 
				possible_labels if possible_labels else [], _id)

			# Display on notebook
			if data_type == 'text':
				metadata += feature
			elif data_type == 'image':
				# TODO: use IPython to reduce library dependency
				PIL.Image.fromarray(feature).show()
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
				for label in labels:
					possible_labels.add(label)
			else:
				self.educate(_id, feature, label)
				possible_labels.add(label)
			
			i += 1