import unittest

from nlpatl.models.clustering.sklearn import SkLearn


class TestModelClusteringKmeans(unittest.TestCase):
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
		clustering = SkLearn()
		assert 8 == clustering.model.n_clusters, \
			'Invalid when using default parameters'

		model_config = {}
		clustering = SkLearn(model_config=model_config)
		assert 8 == clustering.model.n_clusters, \
			'Invalid when passing emtpy parameters'

		model_config = {'n_clusters': 4}
		clustering = SkLearn(model_config=model_config)
		assert 4 == clustering.model.n_clusters, \
			'Invalid when passing parameter'

	def test_cluster(self):
		clustering = SkLearn()
		clustering.train(self.train_features)
		results = clustering.predict_prob(self.train_features)

		num_actual_class = clustering.model.n_clusters
		num_expected_class = len(results)

		assert num_actual_class == num_expected_class, \
			'{} expected clusters is different from {} actual clusters'.format(
				num_actual_class, num_expected_class)
		