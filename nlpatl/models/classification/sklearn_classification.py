"""
	sci-kit learn classification wrapper
"""

from typing import List, Union
from collections import defaultdict
from sklearn.linear_model import (
	LogisticRegression
)
from sklearn.svm import (
	SVC,
	LinearSVC
)
from sklearn.ensemble import (
	RandomForestClassifier
)
import numpy as np

from nlpatl.models.classification.classification import Classification
from nlpatl.storage.storage import Storage

MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES = {
	'logistic_regression': LogisticRegression,
	'svc': SVC,
	'linear_svc': LinearSVC,
	'random_forest': RandomForestClassifier
}


class SkLearnClassification(Classification):
	"""
		:param str model_name: sci-kit learn classification model name
		:param dict model_config: Custom model paramateters
		:param str name: Name of this classification

		>>> import nlpatl.models.classification as nmcla
		>>> model = nmcla.SkLearnClassification()
    """

	def __init__(self, model_name: str = 'logistic_regression', model_config: dict = {}, 
		name: str = 'sklearn'):

		super().__init__(name=name)

		self.model_name = model_name
		self.model_config = model_config

		if model_name in MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES:
			self.model = MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES[model_name](
				**model_config)
		else:
			raise ValueError('`{}` does not support. Supporting {} only'.format(
				model_name, '`' + '`'.join(
					MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES.keys()) + '`'))

	@staticmethod
	def get_mapping() -> dict:
		return MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES

	def train(self, x: np.ndarray, 
		y: Union[np.ndarray, List[str], List[int], List[List[str]], List[List[int]]]):

		"""
			:param np.ndarray x: Raw features
			:param list/np.array y: label

			>>> model.train(x=x, y=y)
		"""

		self.build_label_encoder(y)
		y_encoded = [self.label_encoder[lab] for lab in y]
		self.model.fit(x, y_encoded)

	def predict_proba(self, x: np.ndarray, predict_config: dict={}) -> Storage:
		"""
			:param np.ndarray x: Features
			:param dict predict_config: Custom model prediction paramateters

			>>> model.predict_proba(x=x)
		"""

		probs = self.model.predict_proba(x, **predict_config)
		preds = np.argmax(probs, axis=1)

		return Storage(
			values=probs,
			groups=preds.tolist())
