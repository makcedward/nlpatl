from typing import List
from collections import defaultdict
import numpy as np

from nlpatl.models.learning.learning import Learning
from nlpatl.storage.storage import Storage


class ClusteringSamlping(Learning):
	def __init__(self, x: [List[str], List[float], np.ndarray] = None,
		name: str = 'clustering_samlping'):

		super().__init__(name=name, x=x)

	def validate(self):
		super().validate(['embeddings', 'clustering'])

	def keep_most_representative(self, data: Storage, num_sample: int) -> Storage:
		# Number of nodes may less than num_sample
		num_node = min(num_sample, len(data))

		indices = data.values.argpartition(num_node-1)[:num_node]
		data.filter(indices)

		return data

	def query(self, inputs: List[str], return_type: str = 'dict', 
		num_sample: int = 2) -> List[object]:

		self.validate()

		features = self.embeddings_model.convert(inputs)

		self.clustering_model.train(features)
		preds = self.clustering_model.predict_prob(features)
		
		results = defaultdict(list)
		for label, pred in preds.items():
			result = self.keep_most_representative(
				pred, num_sample=num_sample)
			result.features = self.filter(inputs, result.indices)

			results[label] = self.get_return_object(result, return_type)
			
		return results
