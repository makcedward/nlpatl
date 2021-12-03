from typing import List
from datasets import load_dataset
import unittest
import datasets
import numpy as np

from nlpatl.models.embeddings import (
	Embeddings, 
	Transformers
)
from nlpatl.models.classification import (
	Classification, 
	SkLearnClassification, 
	XGBoostClassification
)
from nlpatl.models import (
	EntropyLearning, 
	SupervisedLearning
)
from nlpatl.storage import Storage


class TestModelLearningSupervised(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		texts = load_dataset('ag_news')['train']['text']
		labels = load_dataset('ag_news')['train']['label']
		cls.train_texts = texts[0:5] + texts[200:205] + texts[1000:1005]
		cls.train_labels = labels[0:5] + labels[200:205] + labels[1000:1005]
		cls.test_texts = texts[0:10] + texts[200:210]
		cls.test_labels = labels[0:10] + labels[200:210]

		cls.transformers_embeddings_model = Transformers(
			'bert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		cls.sklearn_classification_model = SkLearnClassification(
			'logistic_regression',
			model_config={'max_iter': 500})
		cls.xgboost_classification_model = XGBoostClassification(
			model_config={
				'use_label_encoder': False,
				'eval_metric': 'logloss'
			})

	def test_no_model(self):
		learning = SupervisedLearning(
			embeddings_model=None,
			classification_model=None
			)

		with self.assertRaises(Exception) as error:
			learning.explore(self.train_texts, self.train_labels, self.test_texts)
		assert 'Embeddings model does not initialize yet' in str(error.exception), \
			'Does not initialize embeddings model but still able to run'

		learning.embeddings_model = self.transformers_embeddings_model

		with self.assertRaises(Exception) as error:
			learning.explore(self.train_texts, self.train_labels, self.test_texts)
		assert 'Classification model does not initialize yet' in str(error.exception), \
			'Does not initialize classification model but still able to run'

	def test_explore_by_sklearn(self):
		learning = EntropyLearning(
			embeddings_model=self.transformers_embeddings_model,
			classification_model=self.sklearn_classification_model
			)

		learning.learn(self.train_texts, self.train_labels)
		result = learning.explore(self.test_texts)
		assert result, 'No output'

	def test_explore_by_xgboost(self):
		learning = EntropyLearning(
			embeddings_model=self.transformers_embeddings_model,
			classification_model=self.xgboost_classification_model
			)
		
		learning.learn(self.train_texts, self.train_labels)
		result = learning.explore(self.test_texts)
		assert result, 'No output'
		assert result['features'], 'Missed features attribute'

	def test_custom_embeddings_model(self):
		class CustomEmbeddings(Embeddings):
			def convert(self, inputs: List[str]) -> np.ndarray:
				return np.random.rand(len(inputs), 5)

		learning = EntropyLearning(multi_label=True, 
			embeddings_model=CustomEmbeddings(),
			classification_model=self.sklearn_classification_model
			)

		learning.learn(self.train_texts, self.train_labels)

		assert True, 'Unable to apply custom embeddings model'

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

			def predict_proba(self, x, predict_config: dict={}) -> Storage:
				"""
					Do probability prediction here
					e.g. preds = self.model.predict_prob(x, **predict_config)
				"""
				probs = np.random.rand(len(x), 3)
				preds = np.argmax(probs, axis=1)

				return Storage(
					values=probs,
					groups=preds.tolist())

		learning = EntropyLearning(multi_label=True, 
			embeddings_model=self.transformers_embeddings_model,
			classification_model=CustomClassification(model=None))

		learning.learn(self.train_texts, self.train_labels)

		assert True, 'Unable to apply custom classification model'

	def test_custom_supervised_learning_method(self):
		class CustomLearning(SupervisedLearning):
			def __init__(self, multi_label: bool = False, 
				embeddings_model: Embeddings = None, 
				classification_model: Classification = None):

				super().__init__(multi_label=multi_label, 
					embeddings_model=embeddings_model,
					classification_model=classification_model)

			def keep_most_valuable(self, data: Storage, 
				num_sample: int) -> Storage:
				"""
					Find your most valuable data by using data.values
					e.g. scores = score(data.values)
				"""

				data.keep(np.array([0, 1, 2]))
				return data

		learning = CustomLearning(
			embeddings_model=self.transformers_embeddings_model,
			classification_model=self.sklearn_classification_model
			)

		learning.learn(self.train_texts, self.train_labels)
		learning.explore(self.train_texts)

		assert True, 'Unable to apply custom learning method'