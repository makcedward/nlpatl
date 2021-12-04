from typing import List
from datasets import load_dataset
import unittest
import datasets
import numpy as np

from nlpatl.models.embeddings import Transformers
from nlpatl.models.clustering import SkLearnClustering
from nlpatl.models import ClusteringLearning
from nlpatl.storage import Storage


class TestModelLearningClustering(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.train_texts = load_dataset('ag_news')['train']['text'][:20]

		cls.transformers_embeddings_model = Transformers(
			'bert-base-uncased', nn_fwk='pt', padding=True, 
			batch_size=3)
		cls.sklearn_clustering_model = SkLearnClustering('kmeans')

	def test_explore(self):
		learning = ClusteringLearning(
			embeddings_model=self.transformers_embeddings_model,
			clustering_model=self.sklearn_clustering_model)

		result = learning.explore(self.train_texts)

		assert result, 'No output'
		assert result['features'], 'Missed features attribute'
