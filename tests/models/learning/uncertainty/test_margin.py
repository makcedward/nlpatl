import unittest
import datasets
from datasets import load_dataset

from nlpatl.models.embeddings import Transformers
from nlpatl.models.classification import SkLearnClassification
from nlpatl.models import MarginLearning


class TestModelLearningMargin(unittest.TestCase):
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

	def test_learning(self):
		learning = MarginLearning(
			embeddings_model=self.transformers_embeddings_model,
			classification_model=self.sklearn_classification_model
			)
	
		learning.learn(self.train_texts, self.train_labels)
		result = learning.explore(self.test_texts)
		assert result, 'No output'
		assert result['features'], 'Missed features attribute'
