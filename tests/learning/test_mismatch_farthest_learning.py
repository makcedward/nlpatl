from datasets import load_dataset
import unittest
import numpy as np

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

		cls.learning = MismatchFarthestLearning(
			clustering_sampling='nearest_mean',
			embeddings='bert-base-uncased', embeddings_type='transformers',
			embeddings_model_config={'nn_fwk': 'pt', 'padding': True, 'batch_size':8},
			clustering='kmeans', clustering_model_config={'n_clusters': 3},
			classification='logistic_regression'   
			)

		cls.train_features = cls.learning.embeddings_model.convert(cls.train_texts)
		# cls.test_features = cls.learning.embeddings_model.convert(cls.test_texts)

	def tearDown(self):
		self.learning.clear_learn_data()

	def test_explore_first_stage(self):
		result = self.learning.explore_first_stage(self.train_features)

		assert result, 'No result return'
		assert len(result.indices) > 0, 'Empty result'

	def test_explore_second_stage(self):
		learn_indices  = np.array([1, 3, 5])
		self.learning.learn_indices = learn_indices
		self.learning.learn_x = [self.train_texts[idx] for idx in learn_indices]
		self.learning.learn_x_features = [self.train_features[idx] for idx in learn_indices]
		self.learning.learn_y = [self.train_labels[idx] for idx in learn_indices]

		result = self.learning.explore_second_stage(x=self.train_features)

		assert result, 'No result return'
		assert len(result.indices) > 0, 'Empty result'
