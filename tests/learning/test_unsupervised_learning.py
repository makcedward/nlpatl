from typing import List
from datasets import load_dataset
import unittest
import datasets
import numpy as np

from nlpatl.models.embeddings import (
	Embeddings,
	Transformers
)
from nlpatl.models.clustering import (
	Clustering, 
	SkLearnClustering
)
from nlpatl.learning import UnsupervisedLearning
from nlpatl.sampling.unsupervised import ClusteringSampling
from nlpatl.dataset import Dataset


class TestLearningUnsupervised(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.train_texts = load_dataset('ag_news')['train']['text'][:20]

		cls.transformers_embeddings_model = Transformers(
			'bert-base-uncased', nn_fwk='pt', padding=True, 
			batch_size=3)
		cls.sklearn_clustering_model = SkLearnClustering('kmeans')
		cls.clustering_sampling = ClusteringSampling()

	def test_no_model(self):
		learning = UnsupervisedLearning(
			sampling=self.clustering_sampling
			)

		with self.assertRaises(Exception) as error:
			learning.explore(self.train_texts)
		assert 'Embeddings model does not initialize yet' in str(error.exception), \
			'Does not initialize embeddings model but still able to run'

		learning.embeddings_model = self.transformers_embeddings_model
		with self.assertRaises(Exception) as error:
			learning.explore(self.train_texts)
		assert 'Clustering model does not initialize yet' in str(error.exception), \
			'Does not initialize clustering model but still able to run'

	def test_custom_embeddings_model(self):
		class CustomEmbeddings(Embeddings):
			def convert(self, inputs: List[str]) -> np.ndarray:
				return np.random.rand(len(inputs), 5)

		learning = UnsupervisedLearning(
			sampling=self.clustering_sampling,
			embeddings_model=CustomEmbeddings(),
			clustering_model=self.sklearn_clustering_model,
			multi_label=True, 
			)
		learning.explore(self.train_texts)

		assert True, 'Unable to apply custom embeddings model'
