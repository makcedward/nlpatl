from typing import List, Union, Callable
from collections import defaultdict
import numpy as np

from nlpatl.models.classification import Classification
from nlpatl.models.embeddings import Embeddings
from nlpatl.learning import Learning
from nlpatl.sampling import Sampling
from nlpatl.sampling.certainty import MostConfidenceSampling
from nlpatl.dataset import Dataset


class SemiSupervisedLearning(Learning):
	"""
		| Applying both active learning and semi-supervised learning apporach to annotate the most
			valuable data points. You may refer to https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0162075&type=printable
			. Here is the pseudo:
		|	1. Convert raw data to features (Embeddings model)
		|	2. Train model and classifing data points (Classification model)
		|	3. Estmiate the most valuable data points (Sampling)
		|	4. Subject matter experts annotates the most valuable data points
		|	5. Retrain classification model
		|	6. Classify unlabeled data points and labeling those confidences are higher than `self_learn_threshold`
		|	7. Repeat Step 2 to 6 until acquire enough data points.
		
		:param sampling: Sampling method. Refer to nlpatl.sampling.
		:type sampling: :class:`nlpatl.sampling.Sampling`
		:param embeddings_model: Function for converting raw data to embeddings.
		:type embeddings_model: :class:`nlpatl.models.embeddings.Embeddings`
		:param classification_model: Function for classifying inputs
		:type classification_model: :class:`nlpatl.models.classification.Classification`
		:param multi_label: Indicate the classification model is multi-label or 
			multi-class (or binary). Default is False.
		:type multi_label: bool
		:param self_learn_threshold: The minimum threshold for classifying probabilities. Data
			will be labeled automatically if probability is higher than this value. Default is 0.9
		:type self_learn_threshold: float
		:param name: Name of this learning.
		:type name: str
	"""

	def __init__(self, sampling: Sampling,
		embeddings_model: Embeddings = None, 
		classification_model: Classification = None, 
		multi_label: bool = False,
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

	def get_self_learn_data(self):
		"""
			Get all self learnt data points 

			:return: Self learnt data points
			:rtype: Tuple of index list of int, x (:class:`numpy.ndarray`) 
				and y (:class:`numpy.ndarray`) 
		"""

		return self.self_learn_id, self.self_learn_x, self.self_learn_y

	def get_annotated_data(self):
		x = self.concatenate([d for d in [
			self.train_x, self.learn_x, self.self_learn_x] if d])
		y = self.concatenate([d for d in [
			self.train_y, self.learn_y, self.self_learn_y] if d])
		return x, y

	def learn(self, x: Union[List[str], List[int], List[float], np.ndarray] = None, 
		y: Union[List[str], List[int]] = None, include_learn_data: bool = True):

		self.validate()

		if x:
			self.train_x = x
		if y:
			self.train_y = y

		# TODO: cache features
		if include_learn_data:
			x, y = self.get_annotated_data()

		x_features = self.embeddings_model.convert(x)

		self.init_unique_y(y)
		self.classification_model.train(x_features, y)

	def explore(self, 
		x: Union[List[str], List[int], List[float], np.ndarray], 
		return_type: str = 'dict', 
		num_sample: int = 10) -> Union[Dataset, dict]:

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
		x: Union[List[str], List[int], List[float], np.ndarray],
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