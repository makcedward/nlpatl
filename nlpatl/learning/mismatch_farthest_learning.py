from typing import List, Union, Callable, Optional
from collections import defaultdict
import numpy as np

from nlpatl.models.clustering import (
	Clustering, SkLearnClustering
)
from nlpatl.models.classification import (
	Classification,
	SkLearnClassification,
	XGBoostClassification
)
from nlpatl.models.embeddings import (
	Embeddings,
	SentenceTransformers,
	Transformers,
	TorchVision
)

from nlpatl.learning import Learning
from nlpatl.sampling import Sampling
from nlpatl.sampling.certainty import (
	MostConfidenceSampling
)
from nlpatl.sampling.uncertainty import (
	EntropySampling,
	LeastConfidenceSampling,
	MarginSampling,
	MismatchSampling
)
from nlpatl.sampling.clustering import (
	NearestSampling,
	FarthestSampling
)
from nlpatl.dataset import Dataset

CLUSTERING_MODEL_FOR_MISMATCH_FARTHEST_MAPPING_NAMES = {
	'kmeans': SkLearnClustering
}

CLASSIFICATION_MODEL_FOR_MISMATCH_FARTHEST_MAPPING_NAMES = {
	'logistic_regression': SkLearnClassification,
	'svc': SkLearnClassification,
	'linear_svc': SkLearnClassification,
	'random_forest': SkLearnClassification,
	'xgboost': XGBoostClassification
}

EMBEDDINGS_MODEL_FOR_MISMATCH_FARTHEST_MAPPING_NAMES = {
	'sentence_transformers': SentenceTransformers,
	'transformers': Transformers,
	'torch_vision': TorchVision	
}

SAMPLING_FOR_MISMATCH_FARTHEST_MAPPING_NAMES = {
	'most_confidence': MostConfidenceSampling(),
	'entropy': EntropySampling(),
	'least_confidence': LeastConfidenceSampling(),
	'margin': MarginSampling(),
	'nearest': NearestSampling(),
	'fathest': FarthestSampling()
}


class MismatchFarthestLearning(Learning):
	"""
		| Applying mis-match first farthest traversal method apporach (with modification) 
			to annotate the most valuable data points. 
			You may refer to http://zhaoshuyang.com/static/documents/MAL2.pdf
			. Here is the pseudo:
		|	1. [NLPatl] Convert raw data to features (Embeddings model)
		|	2. [NLPatl] Train model and clustering data points (Clustering model)
		|	3. [NLPatl] Estmiate the most valuable data points (Sampling)
		|	4. [Human] Subject matter exepknrnts annotates the most valuable data points
		|	5. [NLPatl] Train classification model (Classification model)
		|	6. [NLPatl] Classify unlabeled data points and comparing the clustering model result
			according to the farthest mismatch data points
		|	7. Repeat Step 2 to 6 until acquire enough data points.
		
		:param sampling: Sampling method. Refer to nlpatl.sampling.
		:type sampling: :class:`nlpatl.sampling.Sampling`
		:param embeddings_model: Function for converting raw data to embeddings.
		:type embeddings_model: :class:`nlpatl.models.embeddings.Embeddings`
		:param classification_model: Function for classifying inputs
		:type classification_model: :class:`nlpatl.models.classification.Classification`
		:param multi_label: Indicate the classification model is multi-label or 
			multi-class (or binary). Default is False.
		:type multi_label: bool
		:param self_learn_threshold: The minimum threshold for classifying probabilities. Data
			will be labeled automatically if probability is higher than this value. Default is 0.9
		:type self_learn_threshold: float
		:param name: Name of this learning.
		:type name: str
	"""

	def __init__(self, clustering_sampling: Union[str, Sampling, Callable],
		embeddings: Embeddings, embeddings_type: str,
		classification: Union[str, Classification], 
		clustering: Union[str, Clustering],
		embeddings_model_config: Optional[dict] = None,
		classification_model_config: Optional[dict] = None,
		clustering_model_config: Optional[dict] = None,
		multi_label: bool = False,
		name: str = 'mismatch_farthest_learning'):

		super().__init__(sampling=None, multi_label=multi_label, name=name)

		self.clustering_sampling = self.init_sampling(clustering_sampling)

		self.embeddings_model_config = embeddings_model_config
		self.embeddings_name, self.embeddings_model = self.init_embeddings(
			embeddings, embeddings_type, embeddings_model_config
			)

		self.classification_model_config = classification_model_config
		self.classification_name, self.classification_model = self.build_classification_model(
			classification, 
			classification_model_config
			)
		
		self.clustering_model_config = clustering_model_config
		self.clustering_name, self.clustering_model = self.build_clustering_model(
			clustering, 
			clustering_model_config
			)

		self.mismatch_sampling = MismatchSampling()
		self.farthest_sampling = FarthestSampling()

	def init_sampling(self, sampling):
		if type(sampling) is str:
			sampling_func = \
				SAMPLING_FOR_MISMATCH_FARTHEST_MAPPING_NAMES.get(
					sampling, None)
			if not sampling_func:
				raise ValueError('`{}` does not support. Supporting {} only'.format(
					sampling, '`' + '`, `'.join(
						SAMPLING_FOR_MISMATCH_FARTHEST_MAPPING_NAMES.keys()) + '`'))

			return sampling_func.sample

		elif hasattr(sampling, '__call__'):
			return sampling
		else:
			raise ValueError('`{}` does not support. Supporting str or function only'.format(
				sampling))

	def init_embeddings(self, embeddings, embeddings_type, embeddings_model_config):
		if type(embeddings) is str:
			name = embeddings
			model = \
				EMBEDDINGS_MODEL_FOR_MISMATCH_FARTHEST_MAPPING_NAMES.get(
					embeddings_type, None)

			if model:
				model = model(embeddings, **embeddings_model_config)

			else:
				raise ValueError('`{}` does not support. Supporting {} only'.format(
					sampling, '`' + '`, `'.join(
						EMBEDDINGS_MODEL_FOR_MISMATCH_FARTHEST_MAPPING_NAMES.keys()) + '`'))

		# TODO: support function
		return name, model

	def build_classification_model(self, name=None, model_config=None):
		if not name:
			name = self.classification_name
		if not model_config:
			model_config = self.classification_model_config or {}

		return self.build_model(
			name, 
			CLASSIFICATION_MODEL_FOR_MISMATCH_FARTHEST_MAPPING_NAMES, 
			model_config)

	def build_clustering_model(self, name=None, model_config=None):
		if not name:
			name = self.clustering_name
		if model_config:
			for k,v in self.clustering_model_config.items():
				if k not in model_config:
					model_config[k] = v
		else:
			model_config = self.clustering_model_config or {}

		return self.build_model(
			name, 
			CLUSTERING_MODEL_FOR_MISMATCH_FARTHEST_MAPPING_NAMES, 
			model_config)

	def build_model(self, name_or_model, possible_models, model_config):
		if type(name_or_model) is str:
			name = name_or_model
			if name in possible_models:
				model = possible_models[name](model_config=model_config)
			else:
				raise ValueError('`{}` does not support. Supporting {} only'.format(
					name, '`' + '`'.join(
						possible_models.keys()) + '`'))
		# TODO: support function
		# TODO: check object
		else:
			name = 'custom'
			model = name_or_model

		return name, model

	def validate(self):
		super().validate(['embeddings', 'clustering', 'classification'])

	def learn_clustering(self, x: np.ndarray, model_config: dict):
		_, self.clustering_model = self.build_clustering_model(
			model_config=model_config
			)

		self.clustering_model.train(x)

	def learn_classifier(self, x: np.ndarray, y: Union[List[str], List[int]]):
		self.classification_model.train(x, y)

	def build_seq_encoder(self, labels: Union[List[str], List[int]]):
		encoded_values = []
		label_decoder = {}
		unique_y_encoder = defaultdict(list)

		for i, c in enumerate(labels):
			encoded_values.append(i)
			label_decoder[i] = c

		for k, v in label_decoder.items():
			unique_y_encoder[v].append(k)
		unique_y_encoder = {k:sorted(v) for k, v in unique_y_encoder.items()}

		return encoded_values, label_decoder, unique_y_encoder

	def explore_first_stage(self, x: np.ndarray, 
		num_sample: int = 1) -> Union[Dataset, dict]:

		self.validate()

		self.clustering_model.train(x)
		preds = self.clustering_model.predict_proba(x)
		
		indices, values = self.clustering_sampling.sample(
			preds.values, preds.groups, num_sample=1)
		preds.keep(indices)
		# Replace original probabilies by sampling values
		preds.values = values

		return preds

	def explore_second_stage(self, x: np.ndarray, num_sample: int = 2):
		# Get annotated dataset
		learn_ids, learn_x, learn_y = self.get_learn_data()
		encoded_learn_y, learn_y_decoder, unique_y_encoder = self.build_seq_encoder(
			learn_y)
		learn_x_features = self.embeddings_model.convert(learn_x)
		
		# Build unannotated dataset
		keep_indices = [_ for _ in range(len(x)) if _ not in learn_ids]
		unannotated_x_features = x[keep_indices]
		
		# Train clustering
		model_config = {
			'n_clusters': len(learn_x),
			'init': learn_x_features,
			'n_init': 1
		}
		self.learn_clustering(x=unannotated_x_features, 
			model_config=model_config)
		clustering_predictions = self.clustering_model.predict_proba(unannotated_x_features)
		clustering_preds = [learn_y_decoder[g] for g in clustering_predictions.groups]
		clustering_values = clustering_predictions.values

		# Train classifier
		self.learn_classifier(x=learn_x_features, y=learn_y)
		probs = self.classification_model.model.predict_proba(unannotated_x_features)
		preds = np.argmax(probs, axis=1)
		classification_preds = [self.classification_model.label_decoder[y] for y in preds]

		# Find mismatch
		mismatch_indices = self.mismatch_sampling.sample(
			clustering_preds, classification_preds, num_sample=len(clustering_preds))

		new_groups = np.array([unique_y_encoder[learn_y_decoder[g]][0] for g in clustering_predictions.groups])
		new_groups = new_groups[mismatch_indices].flatten()
		new_values = clustering_predictions.values[mismatch_indices].flatten()

		positions, values = self.farthest_sampling.sample(new_values, new_groups, num_sample)
		clustering_predictions.keep(positions)

		return clustering_predictions

	def explore_educate_in_notebook(self, 
		x: Union[List[str], List[int], List[float], np.ndarray],
		num_sample: int = 5, num_sample_per_cluster: int = 2, 
		data_type: str = 'text'):

		x_features = self.embeddings_model.convert(x)

		# First stage clustering learning
		valuable_data = self.explore_first_stage(
			x_features, num_sample=num_sample)
		valuable_data.features = [x[i] for i in valuable_data.indices]
		self.show_in_notebook(valuable_data, data_type=data_type)

		# Second stage mismatch-farthest
		safety_break_cnt = 20
		while len(self.learn_x) < num_sample and safety_break_cnt > 0:
			safety_break_cnt -= 1

			valuable_data = self.explore_second_stage(
				x=x_features, num_sample=num_sample_per_cluster)
			valuable_data.features = [x[i] for i in valuable_data.indices]

			self.show_in_notebook(valuable_data, data_type=data_type)
