from typing import List

from nlpatl.models.embeddings.transformers import Transformers
from nlpatl.models.clustering.sklearn import (
	SkLearn as SkLearnCluster
)
from nlpatl.models.classification.sklearn import (
	SkLearn as SkLearnClassifier
)
from nlpatl.storage.storage import Storage


class Labeling:
	def __init__(self, name='labeling'):
		self.name = name
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

		possible_models = SkLearnCluster.get_mapping().keys()

		if model:
			self.model = model
		elif model_name in possible_models:
			self.clustering_model = SkLearnCluster(model_name, model_config)
			self.model_config = model_config
		else:
			raise ValueError('`{}` does not support. Supporting {} only'.format(
				model_name, '`' + '`'.join(possible_models) + '`'))

	def init_classification_model(self, model_name: str = 'logistic_regression',
		model: object = None, model_config: dict = {}):
		# model: object = None, batch_size: int = 32, model_config: dict = {}):

		possible_models = SkLearnClassifier.get_mapping().keys()

		if model:
			self.model = model
		elif model_name in possible_models:
			self.classification_model = SkLearnClassifier(model_name, model_config)
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

	def train(self, x: object, y: object):
		...

	def keep_most_representative(self, data: Storage, num_sample: int) -> Storage: 
		...