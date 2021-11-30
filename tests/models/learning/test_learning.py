from datasets import load_dataset
import unittest
import datasets
import numpy as np

from nlpatl.models.learning.supervised_learning import SupervisedLearning
from nlpatl.storage.storage import Storage

from nlpatl.models import (
	ClusteringLearning,
	EntropyLearning
)


class TestModelLearning(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		texts = load_dataset('ag_news')['train']['text']
		labels = load_dataset('ag_news')['train']['label']
		cls.train_texts = texts[0:5] + texts[200:205] + texts[1000:1005]
		cls.train_labels = labels[0:5] + labels[200:205] + labels[1000:1005]
		cls.test_texts = texts[0:10] + texts[200:210]
		cls.test_labels = labels[0:10] + labels[200:210]

	def test_unsupervised_explore(self):
		learning = ClusteringLearning()
		learning.init_embeddings_model(
			'bert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		learning.init_clustering_model(
			'kmeans', model_config={})

		result = learning.explore(self.train_texts)

		assert not learning.learn_x, 'Learnt something at the beginning'

		for feature, group in zip(result['features'], result['groups']):
			learning.educate(feature, group)

		assert learning.learn_x, 'Unable to explore'

	def test_uncertainty_educate(self):
		learning = EntropyLearning()
		learning.init_embeddings_model(
			'bert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		model_config = {'max_iter': 500}
		learning.init_classification_model('logistic_regression',
			model_config=model_config)

		learning.learn(self.train_texts, self.train_labels)
		first_result = learning.explore(self.test_texts, num_sample=2)

		assert not learning.learn_x, 'Learnt something at the beginning'

		test_texts = self.test_texts
		for feature, group in zip(first_result['features'], first_result['groups']):
			learning.educate(feature, group)

		assert learning.learn_x, 'Unable to explore'

		# retrain with learnt data
		learning.learn(self.train_texts, self.train_labels)
		second_result = learning.explore(self.test_texts, num_sample=2)

		# Expect learnt record should be be picked again
		assert len(second_result['features']) > 0, 'No learnt data'
		for f in second_result['features']:
			assert f not in first_result['features'], 'Low quality of learnt data'

	def test_return_type(self):
		learning = ClusteringLearning()
		learning.init_embeddings_model(
			'bert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		learning.init_clustering_model(
			'kmeans', model_config={})

		result = learning.explore(self.train_texts, return_type='object')

		assert result.features, 'Not object format'

		learning = EntropyLearning()
		learning.init_embeddings_model(
			'bert-base-uncased', return_tensors='pt', padding=True, 
			batch_size=3)
		model_config = {'max_iter': 500}
		learning.init_classification_model('logistic_regression',
			model_config=model_config)

		learning.learn(self.train_texts, self.train_labels)
		result = learning.explore(self.test_texts, num_sample=2, 
			return_type='object')

		assert result.features, 'Not object format'

	def test_educate_multi_label(self):
		learning = ClusteringLearning(multi_label=True)

		expected_labels = [
			['1'],
			['1', '2'],
			['3', '4']
		]

		for i in range(3):
			learning.educate(self.train_texts[i], expected_labels[i])

		learn_x, learn_y = learning.get_learnt_data()
		assert expected_labels == learn_y, 'Unable to learn multi label'
