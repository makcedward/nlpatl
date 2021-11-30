from typing import List
from datasets import load_dataset
import unittest
import datasets
import numpy as np

from nlpatl.models.embeddings import Embeddings
from nlpatl.models.clustering.clustering import Clustering
from nlpatl.models.learning import UnsupervisedLearning
from nlpatl.models.learning.unsupervised.clustering import ClusteringLearning
from nlpatl.storage.storage import Storage


class TestModelLearningUnsupervised(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.train_texts = load_dataset('ag_news')['train']['text'][:20]

	def test_no_model(self):
		learning = UnsupervisedLearning()

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

	def test_custom_embeddings_model(self):
		class CustomEmbeddings(Embeddings):
			def convert(self, inputs: List[str]) -> np.ndarray:
				return np.random.rand(len(inputs), 5)

		learning = ClusteringLearning(multi_label=True, 
			embeddings_model=CustomEmbeddings())
		model_config = {'max_iter': 500}
		learning.init_clustering_model(
			'kmeans', model_config={})
		learning.explore(self.train_texts)

		assert True, 'Unable to apply custom embeddings model'

	def test_custom_clustering_model(self):
		class CustomClustering(Clustering):
			def __init__(self, model):
				self.model = model

			def train(self, x: np.array):
				"""
					Do training here
					e.g. self.model.train(x)
				""" 
				...

			def predict_proba(self, x, predict_config: dict={}) -> Storage:
				"""
					Do calculation here
					e.g. preds = self.model.cal(x, **predict_config)
				"""
				indices = np.array(list(range(len(x))))
				values = np.random.rand(len(x))
				groups = np.random.randint(0, 3, len(x))

				return Storage(
					indices=indices,
					values=values,
					groups=groups.tolist())

		learning = ClusteringLearning(multi_label=True, 
			clustering_model=CustomClustering(model=None))
		learning.init_embeddings_model(
			'bert-base-uncased', return_tensors='pt', padding=True,
			batch_size=3)
		learning.explore(self.train_texts)

		assert True, 'Unable to apply custom clustering model'
