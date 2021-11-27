from __future__ import annotations
import numpy as np
from collections import defaultdict
from sklearn.linear_model import (
	LogisticRegression
)
from sklearn.svm import (
	SVC
)

from nlpatl.models.classification.classification import Classification
from nlpatl.storage.storage import Storage

MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES = {
    'logistic_regression': LogisticRegression,
    'svc': SVC
}


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

	def build_label_encoder(self, labels: List[str, int]):
		uni_labels = sorted(set(labels))

		self.label_encoder = {c:i for i, c in enumerate(uni_labels)}
		self.label_decoder = {i:c for c, i in self.label_encoder.items()}

	def train(self, x: np.array, y: [np.array, List[str]]) -> SkLearn:
		self.build_label_encoder(y)
		y_encoded = [self.label_encoder[lab] for lab in y]
		self.model.fit(x, y_encoded)
		return self

	def predict_proba(self, x, predict_config: dict={}) -> np.array:
		probs = self.model.predict_proba(x, **predict_config)
		preds = np.argmax(probs, axis=1)

		results = defaultdict(Storage)
		for label in sorted(self.model.classes_):
			indices = np.where(preds == label)[0]

			results[label] = Storage(
				indices=indices, group=label,
				values=probs[indices])

		return results



