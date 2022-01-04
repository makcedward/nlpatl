from datasets import load_dataset
import unittest
import numpy as np

from nlpatl.models.embeddings import Transformers
from nlpatl.models.classification import (
	Classification, 
	SkLearnClassification, 
	XGBoostClassification
)
from nlpatl.models.clustering import (
	SkLearnClustering, 
	SkLearnExtraClustering
)
from nlpatl.learning import MismatchFarthestLearning


class TestLearningMismatchFarthest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		texts = load_dataset('ag_news')['train']['text']
		labels = load_dataset('ag_news')['train']['label']
		cls.train_texts = texts[0:5] + texts[200:205] + texts[1000:1005]
		cls.train_labels = labels[0:5] + labels[200:205] + labels[1000:1005]
		cls.test_texts = texts[0:10] + texts[200:210]
		cls.test_labels = labels[0:10] + labels[200:210]

		cls.transformers_embeddings_model = Transformers(
			'bert-base-uncased', nn_fwk='pt', padding=True, 
			batch_size=3)
		cls.sklearn_clustering_model = SkLearnClustering(
			'kmeans',
			model_config={'n_clusters': 3})
		cls.sklearn_extra_clustering_model = SkLearnExtraClustering('kmedoids')
		cls.sklearn_classification_model = SkLearnClassification(
			'logistic_regression',
			model_config={'max_iter': 500})
		cls.xgboost_classification_model = XGBoostClassification(
			model_config={
				'use_label_encoder': False,
				'eval_metric': 'logloss'
			})

		cls.learning = MismatchFarthestLearning(
			clustering_sampling='nearest_mean',
			embeddings=cls.transformers_embeddings_model,
			classification=cls.sklearn_classification_model,
			clustering=cls.sklearn_clustering_model
			)

		cls.train_features = cls.learning.embeddings_model.convert(cls.train_texts)

	def tearDown(self):
		self.learning.clear_learn_data()

	def test_explore_first_stage(self):
		result = self.learning.explore_first_stage(self.train_features)

		assert result, 'No result return'
		assert len(result.indices) > 0, 'Empty result'

	def test_explore_second_stage(self):
		sklearn_extra_clustering_learning = MismatchFarthestLearning(
			clustering_sampling='nearest_mean',
			embeddings=self.transformers_embeddings_model,
			classification=self.sklearn_classification_model,
			clustering=self.sklearn_extra_clustering_model
			)

		for learning in [sklearn_extra_clustering_learning, self.learning]:
			learn_indices  = np.array([1, 3, 5])
			learning.learn_indices = learn_indices
			learning.learn_x = [self.train_texts[idx] for idx in learn_indices]
			learning.learn_x_features = [self.train_features[idx] for idx in learn_indices]
			learning.learn_y = [self.train_labels[idx] for idx in learn_indices]

			result = learning.explore_second_stage(x=self.train_features)

			assert result, 'No result return'
			assert len(result.indices) > 0, 'Empty result'
