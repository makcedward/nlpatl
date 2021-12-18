from typing import List, Union, Callable, Optional
from collections import defaultdict
import numpy as np

from nlpatl.models.classification import Classification
from nlpatl.models.embeddings import Embeddings
from nlpatl.sampling.certainty import MostConfidenceSampling
from nlpatl.learning import Learning
from nlpatl.dataset import Dataset


class SemiSupervisedLearning(Learning):
	"""
		| Applying both active learning and semi-supervised learning apporach to annotate the most
			valuable data points. You may refer to https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0162075&type=printable
			. Here is the pseudo:
		|	1. [NLPatl] Convert raw data to features (Embeddings model)
		|	2. [NLPatl] Train model and classifing data points (Classification model)
		|	3. [NLPatl] Estmiate the most valuable data points (Sampling)
		|	4. [Human] Subject matter experts annotates the most valuable data points
		|	5. [NLPatl] Retrain classification model
		|	6. [NLPatl] Classify unlabeled data points and labeling those confidences are higher than `self_learn_threshold`
		|	7. Repeat Step 2 to 6 until acquire enough data points.
		
		:param sampling: Sampling method for get the most valuable data points. 
			Providing certified methods name (`most_confidence`, `entropy`, 
			`least_confidence`, `margin`, `nearest_mean`, `fathest`)
			or custom function.
		:type sampling: str or function
		:param embeddings: Function for converting raw data to embeddings. Providing 
			model name according to embeddings type. For example, `multi-qa-MiniLM-L6-cos-v1`
			for `sentence_transformers`. bert-base-uncased` for
			`transformers`. `vgg16` for `torch_vision`.
		:type embeddings: str or :class:`nlpatl.models.embeddings.Embeddings`
		:param embeddings_model_config: Configuration for embeddings models. Optional. Ignored
			if using custom embeddings class
		:type embeddings_model_config: dict
		:param embeddings_type: Type of embeddings. `sentence_transformers` for text, 
			`transformers` for text or `torch_vision` for image
		:type embeddings_type: str
		:param classification: Function for classifying inputs. Either providing
			certified methods (`logistic_regression`, `svc`, `linear_svc`, `random_forest`
			and `xgboost`) or custom function.
		:type classification: :class:`nlpatl.models.classification.Classification`
		:param classification_model_config: Configuration for classification models. Optional.
			Ignored if using custom classification class
		:type classification_model_config: dict
		:type multi_label: bool
		:param self_learn_threshold: The minimum threshold for classifying probabilities. Data
			will be labeled automatically if probability is higher than this value. Default is 0.9
		:type self_learn_threshold: float
		:param name: Name of this learning.
		:type name: str
	"""

	def __init__(self, sampling: Union[str, Callable],
		embeddings: Union[str, Embeddings], 
		classification: Union[str, Classification], 
		embeddings_type: Optional[str] = None,
		embeddings_model_config: Optional[dict] = None,
		classification_model_config: Optional[dict] = None,
		multi_label: bool = False,
		self_learn_threshold: float = 0.9,
		name: str = 'semi_supervised_learning'):

		super().__init__(sampling=sampling,
			embeddings=embeddings, embeddings_type=embeddings_type,
			embeddings_model_config=embeddings_model_config,
			classification=classification, 
			classification_model_config=classification_model_config,
			multi_label=multi_label, name=name)

		self.most_confidence_sampling = MostConfidenceSampling(
			threshold=self_learn_threshold).sample
		self.self_learn_indices = None
		self.self_learn_x = None
		self.self_learn_x_features = None
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

		return self.self_learn_indices, self.self_learn_x, \
			self.self_learn_x_features, self.self_learn_y

	# def get_annotated_data(self):
	# 	x = self.concatenate([d for d in [
	# 		self.train_x, self.learn_x, self.self_learn_x] if d])
	# 	x_features = self.concatenate([d for d in [
	# 		self.train_x_features, self.learn_x_features, self.self_learn_x_features] if d])
	# 	y = self.concatenate([d for d in [
	# 		self.train_y, self.learn_y, self.self_learn_y] if d])
	# 	return x, x_features, y

	def learn(self, x: Union[List[str], List[int], List[float], np.ndarray] = None, 
		y: Union[List[str], List[int]] = None, include_learn_data: bool = True):

		self.validate()

		if include_learn_data:
			all_x = self.concatenate(
				[d for d in [x , self.learn_x, self.self_learn_x] if d])
			all_y = self.concatenate(
				[d for d in [y , self.learn_y, self.self_learn_y] if d])
		else:
			all_x = x
			all_y = y

		self.add_unique_y(all_y)

		x_features = self.embeddings_model.convert(all_x)
		self.classification_model.train(x_features, all_y)

	def explore(self, 
		x: Union[List[str], List[int], List[float], np.ndarray], 
		return_type: str = 'dict', 
		num_sample: int = 10) -> Union[Dataset, dict]:

		self.validate()

		x_features = self.embeddings_model.convert(x)
		preds = self.classification_model.predict_proba(x_features)

		indices, values = self.sampling(preds.values, num_sample=num_sample)
		preds.keep(indices)
		# Replace original probabilies by sampling values
		preds.values = values

		preds.inputs = [x[i] for i in preds.indices.tolist()]

		return self.get_return_object(preds, return_type)

	def explore_educate_in_notebook(self, 
		x: Union[List[str], List[int], List[float], np.ndarray],
		num_sample: int = 2, data_type: str = 'text'):

		super().explore_educate_in_notebook(
			x=x, num_sample=num_sample, data_type=data_type)

		# Train model after human annotation
		self.learn()

		# Identify high confidence unannotated data
		unannotated_x = self.filter(x, self.learn_indices)
		x_features = self.embeddings_model.convert(unannotated_x)
		preds = self.classification_model.predict_proba(x_features)

		indices, values = self.most_confidence_sampling(preds.values, len(preds))
		if len(indices) > 0:
			preds.keep(indices)
			# Replace original probabilies by sampling values
			preds.values = values

			# NOT original indices. These are filtered indices 
			indices = preds.indices
			# self.self_learn_x_indices = self.filter(unannotated_x, indices)
			self.self_learn_x = self.filter(unannotated_x, indices)
			self.self_learn_x_features = self.filter(preds.features, indices)
			self.self_learn_y = self.filter(preds.groups, indices)
