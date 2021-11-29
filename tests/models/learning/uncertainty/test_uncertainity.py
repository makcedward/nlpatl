import unittest
import datasets
from datasets import load_dataset

from nlpatl.models import EntropySampling


class TestModelLearningUncertaintySampling(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		texts = load_dataset('ag_news')['train']['text']
		labels = load_dataset('ag_news')['train']['label']
		cls.train_texts = texts[0:5] + texts[200:205] + texts[1000:1005]
		cls.train_labels = labels[0:5] + labels[200:205] + labels[1000:1005]
		cls.test_texts = texts[0:10] + texts[200:210]
		cls.test_labels = labels[0:10] + labels[200:210]

	def test_no_model(self):
		learning = EntropySampling()

		with self.assertRaises(Exception) as error:
			learning.explore(self.train_texts, self.train_labels, self.test_texts)
		assert 'Embeddings model does not initialize yet' in str(error.exception), \
			'Does not initialize embeddings model but still able to run'

		learning.init_embeddings_model(
			'distilbert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		with self.assertRaises(Exception) as error:
			learning.explore(self.train_texts, self.train_labels, self.test_texts)
		assert 'Classification model does not initialize yet' in str(error.exception), \
			'Does not initialize classification model but still able to run'

	def test_explore_by_sklearn(self):
		learning = EntropySampling()

		learning.init_embeddings_model(
			'bert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		model_config = {'max_iter': 500}
		learning.init_classification_model('logistic_regression',
			model_config=model_config)
	
		learning.learn(self.train_texts, self.train_labels)
		result = learning.explore(self.test_texts)
		assert result, 'No output'

	def test_explore_by_xgboost(self):
		learning = EntropySampling()

		learning.init_embeddings_model(
			'bert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)

		model_config = {
			'use_label_encoder': False,
			'eval_metric': 'logloss'
		}
		learning.init_classification_model('xgboost', model_config=model_config)
	
		learning.learn(self.train_texts, self.train_labels)
		result = learning.explore(self.test_texts)
		assert result, 'No output'
		assert result['features'], 'Missed features attribute'
