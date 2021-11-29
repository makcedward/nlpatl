import unittest

from nlpatl.models.clustering.sklearn_clustering import SkLearnClustering


class TestModelClusteringSkLearn(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.train_features = [
			[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
			[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]
		]
		cls.train_labels = [
			0, 0, 1, 1, 0, 0,
			2, 2, 1, 1, 2, 1
		]

	def test_parameters(self):
		clustering = SkLearnClustering()
		assert 8 == clustering.model.n_clusters, \
			'Invalid when using default parameters'

		model_config = {}
		clustering = SkLearnClustering(model_config=model_config)
		assert 8 == clustering.model.n_clusters, \
			'Invalid when passing emtpy parameters'

		model_config = {'n_clusters': 4}
		clustering = SkLearnClustering(model_config=model_config)
		assert 4 == clustering.model.n_clusters, \
			'Invalid when passing parameter'

	def test_cluster(self):
		clustering = SkLearnClustering()
		clustering.train(self.train_features)
		result = clustering.predict_prob(self.train_features)

		num_actual_class = clustering.model.n_clusters
		num_expected_class = len(set(result.groups))

		assert num_actual_class == num_expected_class, \
			'{} expected clusters is different from {} actual clusters'.format(
				num_actual_class, num_expected_class)
		assert result.groups, 'Missed groups attribute'
		assert result.values is not None, 'Missed values attribute'
		assert result.indices is not None, 'Missed indices attribute'
