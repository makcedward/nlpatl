import unittest
import datasets
from datasets import load_dataset

from nlpatl.models.labeling.unsupervised.transfer import TransferSamlping


class TestModelLabelingColdStart(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.train_texts = load_dataset('ag_news')['train']['text'][:20]

	def test_no_model(self):
		labeling = TransferSamlping()

		with self.assertRaises(Exception) as error:
			labeling.generate(self.train_texts)
		assert 'Embeddings model does not initialize yet' in str(error.exception), \
			'Does not initialize embeddings model but still able to run'

		labeling.init_embeddings_model(
			'distilbert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		with self.assertRaises(Exception) as error:
			labeling.generate(self.train_texts)
		assert 'Clustering model does not initialize yet' in str(error.exception), \
			'Does not initialize clustering model but still able to run'

	def test_genearte(self):
		labeling = TransferSamlping()
		labeling.init_embeddings_model(
			'bert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		labeling.init_clustering_model(
			'kmeans', model_config={})

		outputs = labeling.generate(self.train_texts)

		assert outputs, 'No output'