from typing import List, Union, Callable, Optional
from collections import defaultdict
import numpy as np

from nlpatl.dataset import Dataset
from nlpatl.learning import Learning
from nlpatl.sampling import Sampling

import nlpatl.models.embeddings as nme
import nlpatl.models.clustering as nmclu
import nlpatl.models.classification as nmcla
import nlpatl.sampling.uncertainty as nsunc
import nlpatl.sampling.clustering as nsclu

SAMPLING_FOR_MISMATCH_FARTHEST_MAPPING_NAMES = {
	'nearest_mean': nsclu.NearestMeanSampling(),
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
		|	4. [Human] Subject matter experts annotates the most valuable data points
		|	5. [NLPatl] Train classification model (Classification model)
		|	6. [NLPatl] Classify unlabeled data points and comparing the clustering model result
			according to the farthest mismatch data points
		|	7. [Human] Subject matter experts annotates the most valuable data points
		|	8. Repeat Step 2 to 7 until acquire enough data points or reach other
			exit criteria.
		
		:param clustering_sampling: Clustering sampling method for stage 1 exploration. 
			Providing certified methods name (`nearest_mean`) or custom function.
		:type clustering_sampling: str or function
		:param embeddings: Function for converting raw data to embeddings. Providing 
			model name according to embeddings type. For example, `multi-qa-MiniLM-L6-cos-v1`
			for `sentence_transformers`. bert-base-uncased` for
			`transformers`. `vgg16` for `torch_vision`.
		:type embeddings: str or :class:`nlpatl.models.embeddings.Embeddings`
		:param embeddings_model_config: Configuration for embeddings models. Optional. Ignored
			if using custom embeddings class
		:type embeddings_model_config: dict
		:param embeddings_type: Type of embeddings. `sentence_transformers` for text, 
			`transformers` for text or `torch_vision` for image
		:type embeddings_type: str
		:param clustering: Function for clustering inputs. Either providing
			certified methods (`kmeans`) or custom function.
		:type clustering: str or :class:`nlpatl.models.clustering.Clustering`
		:param clustering_model_config: Configuration for clustering models. Optional. Ignored
			if using custom clustering class
		:type clustering_model_config: dict
		:param classification: Function for classifying inputs. Either providing
			certified methods (`logistic_regression`, `svc`, `linear_svc`, `random_forest`
			and `xgboost`) or custom function.
		:type classification: :class:`nlpatl.models.classification.Classification`
		:param classification_model_config: Configuration for classification models. Optional.
			Ignored if using custom classification class
		:type classification_model_config: dict
		:param multi_label: Indicate the classification model is multi-label or 
			multi-class (or binary). Default is False.
		:type multi_label: bool
		:param name: Name of this learning.
		:type name: str
	"""

	def __init__(self, clustering_sampling: Union[str, Callable],
		embeddings: Union[str, nme.Embeddings], 
		clustering: Union[str, nmclu.Clustering],
		classification: Union[str, nmcla.Classification], 
		embeddings_type: Optional[str] = None,
		embeddings_model_config: Optional[dict] = None,
		clustering_model_config: Optional[dict] = None,
		classification_model_config: Optional[dict] = None,
		multi_label: bool = False,
		name: str = 'mismatch_farthest_learning'):

		super().__init__(
			embeddings=embeddings, embeddings_type=embeddings_type,
			embeddings_model_config=embeddings_model_config,
			clustering=clustering, 
			clustering_model_config=clustering_model_config,
			classification=classification, 
			classification_model_config=classification_model_config,
			multi_label=multi_label, name=name)

		self.clustering_sampling = self.init_sampling(clustering_sampling)
		self.mismatch_sampling = nsunc.MismatchSampling().sample
		self.farthest_sampling = nsclu.FarthestSampling().sample

	def get_sampling_mapping(self):
		return SAMPLING_FOR_MISMATCH_FARTHEST_MAPPING_NAMES

	def validate(self):
		super().validate(['embeddings', 'clustering', 'classification'])

	def init_kmean(self, x: np.ndarray, model_config: dict=None):
		_, kmean = self.init_clustering_model('kmeans', model_config)
		kmean.train(x)
		return kmean

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
		
		indices, values = self.clustering_sampling(
			preds.values, preds.groups, num_sample=1)
		preds.keep(indices)
		# Replace original probabilies by sampling values
		preds.values = values

		return preds

	def explore_second_stage(self, x: np.ndarray, num_sample: int = 2):
		# Get annotated dataset
		learn_indices, learn_x, learn_x_features, learn_y = self.get_learn_data()
		encoded_learn_y, learn_y_decoder, unique_y_encoder = self.build_seq_encoder(
			learn_y)
		# TODO: cache
		learn_x_features = self.embeddings_model.convert(learn_x)
		
		# Build unannotated dataset
		keep_indices = [_ for _ in range(len(x)) if _ not in learn_indices]
		unannotated_x_features = x[keep_indices]
		
		# Train clustering
		model_config = {
			'n_clusters': len(learn_x_features),
			'init': learn_x_features,
			'n_init': 1
		}
		kmean = self.init_kmean(x=unannotated_x_features, 
			model_config=model_config)
		clustering_predictions = kmean.predict_proba(unannotated_x_features)
		clustering_preds = [learn_y_decoder[g] for g in clustering_predictions.groups]
		clustering_values = clustering_predictions.values

		# Train classifier
		self.learn_classifier(x=learn_x_features, y=learn_y)
		probs = self.classification_model.model.predict_proba(unannotated_x_features)
		preds = np.argmax(probs, axis=1)
		classification_preds = [self.classification_model.label_decoder[y] for y in preds]

		# Find mismatch
		mismatch_indices = self.mismatch_sampling(
			clustering_preds, classification_preds, num_sample=len(clustering_preds))

		new_groups = np.array([unique_y_encoder[learn_y_decoder[g]][0] for g in clustering_predictions.groups])
		new_groups = new_groups[mismatch_indices].flatten()
		new_values = clustering_predictions.values[mismatch_indices].flatten()

		positions, values = self.farthest_sampling(new_values, new_groups, num_sample)
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
		valuable_data.inputs = [x[i] for i in valuable_data.indices]
		self.show_in_notebook(valuable_data, data_type=data_type)

		# Second stage mismatch-farthest
		while len(self.learn_x) < num_sample:
			valuable_data = self.explore_second_stage(
				x=x_features, num_sample=num_sample_per_cluster)
			valuable_data.inputs = [x[i] for i in valuable_data.indices]

			self.show_in_notebook(valuable_data, data_type=data_type)
