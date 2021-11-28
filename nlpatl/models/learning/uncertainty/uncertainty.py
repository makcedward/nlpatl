from typing import List
from collections import defaultdict

from nlpatl.models.learning.learning import Learning
from nlpatl.storage.storage import Storage


class UncertaintySampling(Learning):
	def __init__(self, name: str = 'uncertainty_sampling'):
		super().__init__(name=name)

	def validate(self):
		super().validate(['embeddings', 'classification'])

	def train(self, x: List[str], y: List[int]):
		self.classification_model.train(x, y)

	def keep_most_representative(self, data: Storage, num_sample: int) -> List[Storage]:
		...

	def query(self, train_inputs: List[str], train_labels: List[int], 
		test_inputs: List[str], return_type: str = 'dict', 
		num_sample: int = 2) -> List[object]:

		self.validate()

		train_features = self.embeddings_model.convert(train_inputs)
		self.train(train_features, train_labels)

		test_features = self.embeddings_model.convert(test_inputs)
		preds = self.classification_model.predict_proba(test_features)

		results = defaultdict(list)
		for label, pred in preds.items():
			result = self.keep_most_representative(
				pred, num_sample=num_sample)
			result.features = self.filter(test_inputs, result.indices)

			results[label] = self.get_return_object(result, return_type)
			
		return results
