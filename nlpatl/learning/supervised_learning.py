from typing import List, Union
from collections import defaultdict
import numpy as np

from nlpatl.models.classification import Classification
from nlpatl.models.embeddings import Embeddings
from nlpatl.learning import Learning
from nlpatl.sampling import Sampling
from nlpatl.storage import Storage


class SupervisedLearning(Learning):
	def __init__(self, 
		sampling: Sampling,
		embeddings_model: Embeddings,
		classification_model: Classification,
		multi_label: bool = False, 
		name: str = 'classification_learning'):

		super().__init__(sampling=sampling,
			embeddings_model=embeddings_model,
			classification_model=classification_model,
			multi_label=multi_label, 
			name=name)

	def validate(self):
		super().validate(['embeddings', 'classification'])

	def learn(self, x: Union[List[str], List[int], List[float], np.ndarray], 
		y: Union[List[str], List[int]], include_leart_data: bool = True):
		
		self.validate()

		self.train_x = x
		self.train_y = y

		if include_leart_data and self.learn_x is not None:
			if type(x) is np.ndarray and type(self.learn_x) is ndarray:
				x_features = self.embeddings_model.convert(
					np.concatenate((x, self.learn_x)))
			else:
				x_features = self.embeddings_model.convert(
					x+self.learn_x)

			y += self.learn_y
		else:
			x_features = self.embeddings_model.convert(x)

		self.init_unique_y(y)
		self.classification_model.train(x_features, y)

	def explore(self, x: List[str], return_type: str = 'dict', 
		num_sample: int = 10) -> List[object]:

		self.validate()

		x_features = self.embeddings_model.convert(x)
		preds = self.classification_model.predict_proba(x_features)

		indices, values = self.sampling.sample(preds.values, num_sample=num_sample)
		preds.keep(indices)
		# Replace original probabilies by sampling values
		preds.values = values

		preds.features = [x[i] for i in preds.indices.tolist()]

		return self.get_return_object(preds, return_type)
