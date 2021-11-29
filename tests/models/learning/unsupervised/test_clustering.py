import unittest
import datasets
from datasets import load_dataset

from nlpatl.models import ClusteringSamlping


class TestModelLearningClusteringSampling(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.train_texts = load_dataset('ag_news')['train']['text'][:20]

	def test_no_model(self):
		learning = ClusteringSamlping()

		with self.assertRaises(Exception) as error:
			learning.explore(self.train_texts)
		assert 'Embeddings model does not initialize yet' in str(error.exception), \
			'Does not initialize embeddings model but still able to run'

		learning.init_embeddings_model(
			'distilbert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		with self.assertRaises(Exception) as error:
			learning.explore(self.train_texts)
		assert 'Clustering model does not initialize yet' in str(error.exception), \
			'Does not initialize clustering model but still able to run'

	def test_explore(self):
		learning = ClusteringSamlping()
		learning.init_embeddings_model(
			'bert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		learning.init_clustering_model(
			'kmeans', model_config={})

		result = learning.explore(self.train_texts)

		assert result, 'No output'
		assert result['features'], 'Missed features attribute'
