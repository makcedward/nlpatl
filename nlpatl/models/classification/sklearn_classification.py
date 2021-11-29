from typing import List
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
try:
	from xgboost import XGBClassifier
except ImportError:
	# No installation required if not using this function
	pass
import numpy as np

from nlpatl.models.classification.classification import Classification
from nlpatl.storage.storage import Storage

MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES = {
	'logistic_regression': LogisticRegression,
	'svc': SVC,
	'linear_svc': LinearSVC,
	'random_forest': RandomForestClassifier
}
try:
	MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES['xgboost'] = XGBClassifier
except NameError:
	# No installation required if not using this class
	pass


class SkLearnClassification(Classification):
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

	def train(self, x: np.array, y: [np.array, List[str]]):
		self.build_label_encoder(y)
		y_encoded = [self.label_encoder[lab] for lab in y]
		self.model.fit(x, y_encoded)

	def predict_proba(self, x, predict_config: dict={}) -> np.array:
		probs = self.model.predict_proba(x, **predict_config)
		preds = np.argmax(probs, axis=1)

		return Storage(
			values=probs,
			groups=preds.tolist())
