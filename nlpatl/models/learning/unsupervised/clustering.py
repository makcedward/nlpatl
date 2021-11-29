from typing import List
from collections import defaultdict
import numpy as np

from nlpatl.models.learning.learning import Learning
from nlpatl.storage.storage import Storage


class ClusteringSamlping(Learning):
	def __init__(self, name: str = 'clustering_samlping'):

		super().__init__(name=name)

	def validate(self):
		super().validate(['embeddings', 'clustering'])

	def keep_most_valuable(self, data: Storage, 
		num_sample: int) -> Storage:
	
		# TODO: Tuning me. Allocating `to_be_filter_indices` size first
		to_be_filter_indices = []
		for group in np.unique(data.groups):
			indices = np.where(data.groups == group)[0]
			values = data.values[indices]

			num_node = min(num_sample, len(indices))

			# get first n shortest distances
			local_indices = values.argpartition(num_node-1)[:num_node]
			to_be_filter_indices.append(indices[local_indices])

		data.keep(np.concatenate(to_be_filter_indices))

		return data

	def explore(self, inputs: List[str], return_type: str = 'dict', 
		num_sample: int = 2) -> [Storage, dict]:

		self.validate()

		features = self.embeddings_model.convert(inputs)

		self.clustering_model.train(features)
		preds = self.clustering_model.predict_prob(features)

		preds = self.keep_most_valuable(preds, num_sample=num_sample)

		preds.features = [inputs[i] for i in preds.indices.tolist()]

		return self.get_return_object(preds, return_type)
		