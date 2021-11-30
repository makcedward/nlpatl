from typing import List
from collections import defaultdict
import numpy as np
from sklearn.cluster import (
	KMeans
)

from nlpatl.models.clustering.clustering import Clustering
from nlpatl.storage.storage import Storage

MODEL_FOR_SKLEARN_CLUSTERING_MAPPING_NAMES = {
    'kmeans': KMeans,
}


class SkLearnClustering(Clustering):
	def __init__(self, model_name: str = 'kmeans', model_config: dict = {}, 
		name: str = 'kmeans'):

		super().__init__(name)

		self.model_name = model_name
		self.model_config = model_config

		if model_name in MODEL_FOR_SKLEARN_CLUSTERING_MAPPING_NAMES:
			self.model = MODEL_FOR_SKLEARN_CLUSTERING_MAPPING_NAMES[model_name](
				**model_config)
		else:
			raise ValueError('`{}` does not support. Supporting {} only'.format(
				model_name, '`' + '`,`'.join(
					MODEL_FOR_SKLEARN_CLUSTERING_MAPPING_NAMES.keys()) + '`'))

	@staticmethod
	def get_mapping() -> dict:
		return MODEL_FOR_SKLEARN_CLUSTERING_MAPPING_NAMES

	def train(self, inputs: List[float]):
		self.model.fit(inputs)

	def predict_proba(self, inputs: List[float], 
		predict_config: dict={}) -> Storage:

		num_cluster = self.model.n_clusters

		clust_dists = self.model.transform(inputs)
		preds = self.model.predict(inputs, **predict_config)
		total_record = len(preds)

		indices = np.zeros(total_record, dtype=int)
		values = np.zeros(total_record, dtype=np.float)
		groups = [-1] * total_record
		start_pos = 0
		for label in range(self.model.n_clusters):
			label_indices = np.where(preds == label)[0]
			end_pos = start_pos + len(label_indices)

			indices[start_pos:end_pos] = label_indices
			values[start_pos:end_pos] = clust_dists[label_indices][:, label]
			groups[start_pos:end_pos] = [label] * len(label_indices)
			
			start_pos = end_pos

		return Storage(indices=indices, values=values, groups=groups)
