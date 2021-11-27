from typing import List
from collections import defaultdict
import operator

from nlpatl.models.labeling.labeling import Labeling
from nlpatl.storage.storage import Storage


class TransferSamlping(Labeling):
	def __init__(self, name: str = 'cold_start'):
		super().__init__(name)

	def validate(self):
		super().validate(['embeddings', 'clustering'])

	def keep_most_representative(self, data: Storage, num_sample: int) -> Storage:
		# Number of nodes may less than num_sample
		num_node = min(num_sample, len(data))

		indices = data.values.argpartition(num_node-1)[:num_node]
		data.filter(indices)

		return data

	def generate(self, inputs: List[str], num_sample: int = 2) -> List[str]:
		self.validate()

		features = self.embeddings_model.convert(inputs)

		self.clustering_model.train(features)
		preds = self.clustering_model.predict_prob(features)
		
		results = defaultdict(list)
		for label, pred in preds.items():
			results[label] = self.keep_most_representative(
				pred, num_sample=num_sample)
			
		return results
