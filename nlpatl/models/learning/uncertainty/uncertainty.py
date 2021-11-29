from typing import List
from collections import defaultdict
import numpy as np

from nlpatl.models.learning.learning import Learning
from nlpatl.storage.storage import Storage


class UncertaintySampling(Learning):
	def __init__(self, name: str = 'uncertainty_sampling'):
		super().__init__(name=name)

	def validate(self):
		super().validate(['embeddings', 'classification'])

	def learn(self, x: [List[str], List[int], List[float], np.ndarray], 
		y: [List[str], List[int]], include_leart_data: bool = True):
		
		self.validate()

		if include_leart_data and self.learn_x is not None:
			if type(x) is np.ndarray and type(self.learn_x) is ndarray:
				x_features = self.embeddings_model.convert(
					np.concatenate((x, self.learn_x)))
			else:
				x_features = self.embeddings_model.convert(
					x+self.learn_x)

			# TODO: support multi-label
			y += self.learn_y
		else:
			x_features = self.embeddings_model.convert(x)

		self.classification_model.train(x_features, y)

	def explore(self, x: List[str], return_type: str = 'dict', 
		num_sample: int = 10) -> List[object]:

		self.validate()

		x_features = self.embeddings_model.convert(x)
		preds = self.classification_model.predict_proba(x_features)

		preds = self.keep_most_valuable(preds, num_sample=num_sample)

		preds.features = [x[i] for i in preds.indices.tolist()]

		return self.get_return_object(preds, return_type)
