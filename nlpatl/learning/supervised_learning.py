from typing import List, Union, Callable, Optional
from collections import defaultdict
import numpy as np

from nlpatl.models.classification import Classification
from nlpatl.models.embeddings import Embeddings
from nlpatl.learning import Learning
from nlpatl.dataset import Dataset


class SupervisedLearning(Learning):
	"""
		| Applying typical active learning apporach to annotate the most valuable data points. Here is the pseudo:
		|	1. [NLPatl] Convert raw data to features (Embeddings model)
		|	2. [NLPatl] Train model and classifing data points (Classification model)
		|	3. [NLPatl] Estmiate the most valuable data points (Sampling)
		|	4. [Human] Subject matter experts annotates the most valuable data points
		|	5. Repeat Step 2 to 4 until acquire enough data points.
		
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
		:param multi_label: Indicate the classification model is multi-label or 
			multi-class (or binary). Default is False.
		:type multi_label: bool
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
		name: str = 'supervised_learning'):

		super().__init__(sampling=sampling,
			embeddings=embeddings, embeddings_type=embeddings_type,
			embeddings_model_config=embeddings_model_config,
			classification=classification, 
			classification_model_config=classification_model_config,
			multi_label=multi_label, name=name)

	def validate(self):
		super().validate(['embeddings', 'classification'])

	def learn(self, x: Union[List[str], List[int], List[float], np.ndarray], 
		y: Union[List[str], List[int]], include_learn_data: bool = True):
		
		self.validate()

		self.train_x = x
		self.train_y = y

		x_features = self.embeddings_model.convert(x)

		if include_learn_data and self.learn_x_features is not None:
			x_features = np.concatenate((x_features, self.learn_x_features))
			y += self.learn_y
			
		self.init_unique_y(y)
		self.classification_model.train(x_features, y)

	def explore(self, x: List[str], return_type: str = 'dict', 
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
