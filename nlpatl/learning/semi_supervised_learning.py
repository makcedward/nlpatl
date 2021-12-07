from typing import List, Union, Callable
from collections import defaultdict
import numpy as np

from nlpatl.models.classification import Classification
from nlpatl.models.embeddings import Embeddings
from nlpatl.learning import Learning
from nlpatl.sampling import Sampling
from nlpatl.sampling.certainty import MostConfidenceSampling
from nlpatl.storage import Storage


class SemiSupervisedLearning(Learning):
	# https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0162075&type=printable

	def __init__(self, sampling: Sampling,
		multi_label: bool = False,
		embeddings_model: Embeddings = None, 
		classification_model: Classification = None, 
		self_learn_threshold: float = 0.9,
		name: str = 'semi_supervised_samlping'):

		super().__init__(multi_label=multi_label, 
			sampling=sampling,
			embeddings_model=embeddings_model,
			classification_model=classification_model,
			name=name)

		self.self_learn_id = None
		self.self_learn_x = None
		self.self_learn_y = None
		self.self_learn_threshold = self_learn_threshold

	def validate(self):
		super().validate(['embeddings', 'classification'])

	def get_annotated_data(self):
		x = self.concatenate([d for d in [
			self.train_x, self.learn_x, self.self_learn_x] if d])
		y = self.concatenate([d for d in [
			self.train_y, self.learn_y, self.self_learn_y] if d])
		return x, y

	def learn(self, x: Union[List[str], List[int], List[float], np.ndarray] = None, 
		y: Union[List[str], List[int]] = None, include_learnt_data: bool = True):
		
		self.validate()

		if x:
			self.train_x = x
		if y:
			self.train_y = y

		# TODO: cache features
		if include_learnt_data:
			x, y = self.get_annotated_data()

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

	def explore_educate_in_notebook(self, 
		x: [List[str], List[int], List[float], np.ndarray],
		num_sample: int = 2, data_type: str = 'text'):

		super().explore_educate_in_notebook(
			x=x, num_sample=num_sample, data_type=data_type)

		# Train model after human annotation
		self.learn()

		# Identify high confidence unannotated data
		unannotated_x = self.filter(x, self.learn_id)
		x_features = self.embeddings_model.convert(unannotated_x)
		preds = self.classification_model.predict_proba(x_features)

		most_confidence_sampling = MostConfidenceSampling(
			threshold=self.self_learn_threshold)
		indices, values = most_confidence_sampling.sample(preds.values, len(preds))
		preds.keep(indices)
		# Replace original probabilies by sampling values
		preds.values = values

		# NOT original indices. These are filtered indices 
		indices = preds.indices
		self.self_learn_x = self.filter(unannotated_x, indices)
		self.self_learn_y = self.filter(preds.groups, indices)