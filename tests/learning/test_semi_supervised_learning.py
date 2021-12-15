from typing import List, Union
from datasets import load_dataset
import unittest
import numpy as np

from nlpatl.models.embeddings import Transformers
from nlpatl.models.classification import (
	Classification, 
	SkLearnClassification, 
	XGBoostClassification
)
from nlpatl.learning import SemiSupervisedLearning
from nlpatl.dataset import Dataset


class TestLearningSemiSupervised(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		texts = load_dataset('ag_news')['train']['text']
		labels = load_dataset('ag_news')['train']['label']
		cls.train_texts = texts[0:5] + texts[200:205] + texts[1000:1005]
		cls.train_labels = labels[0:5] + labels[200:205] + labels[1000:1005]
		cls.test_texts = texts[0:10] + texts[200:210]
		cls.test_labels = labels[0:10] + labels[200:210]

		cls.transformers_embeddings_model = Transformers(
			'bert-base-uncased', nn_fwk='pt', padding=True, 
			batch_size=3)
		cls.sklearn_classification_model = SkLearnClassification(
			'logistic_regression',
			model_config={'max_iter': 500})
		cls.xgboost_classification_model = XGBoostClassification(
			model_config={
				'use_label_encoder': False,
				'eval_metric': 'logloss'
			})

	def test_explore_by_sklearn(self):
		learning = SemiSupervisedLearning(
			sampling='most_confidence',
			embeddings=self.transformers_embeddings_model,
			classification=self.sklearn_classification_model
			)

		learning.learn(self.train_texts, self.train_labels)
		result = learning.explore(self.test_texts)
		assert result, 'No output'

	def test_explore_by_xgboost(self):
		learning = SemiSupervisedLearning(
			sampling='most_confidence',
			embeddings=self.transformers_embeddings_model,
			classification=self.xgboost_classification_model
			)
		
		learning.learn(self.train_texts, self.train_labels)
		result = learning.explore(self.test_texts)
		assert result, 'No output'
		assert result['features'], 'Missed features attribute'

	def test_custom_classification_model(self):
		class CustomClassification(Classification):
			def __init__(self, model):
				self.model = model

			def train(self, x: np.array, 
				y: [np.array, List[str], List[int], List[List[str]], List[List[int]]]):
				"""
					Do training here
					e.g. self.model.train(x, y)
				""" 
				...

			def predict_proba(self, x, predict_config: dict={}) -> Union[Dataset, object]:
				"""
					Do probability prediction here
					e.g. preds = self.model.predict_prob(x, **predict_config)
				"""
				probs = np.random.rand(len(x), 3)
				preds = np.argmax(probs, axis=1)

				return Dataset(
					values=probs,
					groups=preds.tolist())

		learning = SemiSupervisedLearning(
			sampling='most_confidence',
			embeddings=self.transformers_embeddings_model,
			classification=CustomClassification(model=None),
			multi_label=True
			)

		learning.learn(self.train_texts, self.train_labels)

		assert True, 'Unable to apply custom classification model'
