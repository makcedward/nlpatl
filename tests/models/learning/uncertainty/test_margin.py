import unittest
import datasets
from datasets import load_dataset

from nlpatl.models.learning.uncertainty.margin import MarginSampling


class TestModelLearningMarginSampling(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		texts = load_dataset('ag_news')['train']['text']
		labels = load_dataset('ag_news')['train']['label']
		cls.train_texts = texts[0:5] + texts[200:205] + texts[1000:1005]
		cls.train_labels = labels[0:5] + labels[200:205] + labels[1000:1005]
		cls.test_texts = texts[0:10] + texts[200:210]

	def test_samlping(self):
		learing = MarginSampling()

		learing.init_embeddings_model(
			'bert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		model_config = {'max_iter': 500}
		learing.init_classification_model('logistic_regression',
			model_config=model_config)
	
		outputs = learing.query(self.train_texts, self.train_labels, self.test_texts)
		assert outputs, 'No output'
