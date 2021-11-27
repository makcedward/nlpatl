import unittest
import datasets
from datasets import load_dataset

from nlpatl.models.labeling.uncertainty.entropy import EntropySampling


class TestModelLabelingEntropySampling(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		texts = load_dataset('ag_news')['train']['text']
		labels = load_dataset('ag_news')['train']['label']
		cls.train_texts = texts[0:10] + texts[200:210]
		cls.train_labels = labels[0:10] + labels[200:210]
		cls.test_texts = texts[0:10] + texts[200:210]

	def test_no_model(self):
		labeling = EntropySampling()

		with self.assertRaises(Exception) as error:
			labeling.generate(self.train_texts, self.train_labels, self.test_texts)
		assert 'Embeddings model does not initialize yet' in str(error.exception), \
			'Does not initialize embeddings model but still able to run'

		labeling.init_embeddings_model(
			'distilbert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		with self.assertRaises(Exception) as error:
			labeling.generate(self.train_texts, self.train_labels, self.test_texts)
		assert 'Classification model does not initialize yet' in str(error.exception), \
			'Does not initialize classification model but still able to run'

	def test_genearte(self):
		labeling = EntropySampling()

		labeling.init_embeddings_model(
			'bert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		labeling.init_classification_model(
			'logistic_regression')
	
		outputs = labeling.generate(self.train_texts, self.train_labels, self.test_texts)

		assert outputs, 'No output'