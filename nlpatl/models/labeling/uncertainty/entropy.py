import operator
from typing import List
from collections import defaultdict
import numpy as np

from nlpatl.models.labeling.labeling import Labeling
from nlpatl.storage.storage import Storage


class EntropySampling(Labeling):
	def __init__(self, name='entropy_sampling'):
		super().__init__(name=name)

	def validate(self):
		super().validate(['embeddings', 'classification'])

	def train(self, x: List[str], y: List[int]):
		self.classification_model.train(x, y)

	def keep_most_representative(self, data: Storage, num_sample: int):
		# Number of nodes may less than num_sample
		num_node = min(num_sample, len(data))

		indices = np.argpartition(-data.values, num_node)[:num_node]
		data.filter(indices)

		return data

	def generate(self, train_inputs: List[str], train_labels: List[int], 
		test_inputs: List[str], num_sample: int = 2) -> List[str]:

		self.validate()

		train_features = self.embeddings_model.convert(train_inputs)
		self.train(train_features, train_labels)

		test_features = self.embeddings_model.convert(test_inputs)
		preds = self.classification_model.predict_proba(test_features)

		results = defaultdict(list)
		for label, pred in preds.items():
			# replace classification's probabilties to entropies
			pred.values = self.classification_model.entropy(pred.values)

			results[label] = self.keep_most_representative(
				pred, num_sample=num_sample)
			
		return results

