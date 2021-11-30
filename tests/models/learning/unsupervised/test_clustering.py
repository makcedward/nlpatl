from typing import List
from datasets import load_dataset
import unittest
import datasets
import numpy as np

from nlpatl.models.embeddings import Embeddings
from nlpatl.models.clustering.clustering import Clustering
from nlpatl.models.learning.unsupervised.clustering import ClusteringLearning
from nlpatl.storage.storage import Storage


class TestModelLearningClustering(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.train_texts = load_dataset('ag_news')['train']['text'][:20]

	def test_explore(self):
		learning = ClusteringLearning()
		learning.init_embeddings_model(
			'bert-base-uncased', return_tensors='pt', padding=True,
			batch_size=3)
		learning.init_clustering_model(
			'kmeans', model_config={})

		result = learning.explore(self.train_texts)

		assert result, 'No output'
		assert result['features'], 'Missed features attribute'
