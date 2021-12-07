from datasets import load_dataset
import unittest
import datasets
import numpy as np

from nlpatl.models.embeddings import Transformers
from nlpatl.models.clustering import SkLearnClustering
from nlpatl.models.classification import SkLearnClassification
from nlpatl.learning import (
	SupervisedLearning,
	UnsupervisedLearning
)
from nlpatl.sampling.uncertainty import EntropySampling
from nlpatl.sampling.unsupervised import ClusteringSampling
from nlpatl.storage import Storage


class TestLearning(unittest.TestCase):
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
		cls.sklearn_clustering_model = SkLearnClustering('kmeans')
		cls.sklearn_classification_model = SkLearnClassification(
			'logistic_regression',
			model_config={'max_iter': 500})
		cls.entropy_sampling = EntropySampling()
		cls.clustering_sampling = ClusteringSampling()

	def test_unsupervised_explore(self):
		learning = UnsupervisedLearning(
			sampling=self.clustering_sampling,
			embeddings_model=self.transformers_embeddings_model, 
			clustering_model=self.sklearn_clustering_model)

		result = learning.explore(self.train_texts)

		assert not learning.learn_x, 'Learnt something at the beginning'

		for index, feature, group in zip(
			result['indices'], result['features'], result['groups']):

			learning.educate(index, feature, group)

		assert learning.learn_x, 'Unable to explore'

	def test_uncertainty_educate(self):
		learning = SupervisedLearning(
			sampling=self.entropy_sampling,
			embeddings_model=self.transformers_embeddings_model,
			classification_model=self.sklearn_classification_model
			)
		learning.learn(self.train_texts, self.train_labels)
		first_result = learning.explore(self.test_texts, num_sample=2)

		assert not learning.learn_x, 'Learnt something at the beginning'

		test_texts = self.test_texts
		for index, feature, group in zip(
			first_result['indices'], first_result['features'], 
			first_result['groups']):

			learning.educate(index, feature, group)

		assert learning.learn_x, 'Unable to explore'

		# retrain with learnt data
		learning.learn(self.train_texts, self.train_labels)
		second_result = learning.explore(self.test_texts, num_sample=2)

		# Expect learnt record should be be picked again
		assert len(second_result['features']) > 0, 'No learnt data'
		for f in second_result['features']:
			assert f not in first_result['features'], 'Low quality of learnt data'

	def test_return_type(self):
		learning = UnsupervisedLearning(
			sampling=self.clustering_sampling,
			embeddings_model=self.transformers_embeddings_model, 
			clustering_model=self.sklearn_clustering_model)
		result = learning.explore(self.train_texts, return_type='object')

		assert result.features, 'Not object format'

		learning = SupervisedLearning(
			sampling=self.entropy_sampling,
			embeddings_model=self.transformers_embeddings_model, 
			classification_model=self.sklearn_classification_model
			)
		learning.learn(self.train_texts, self.train_labels)
		result = learning.explore(self.test_texts, num_sample=2, 
			return_type='object')

		assert result.features, 'Not object format'

	def test_educate_multi_label(self):
		learning = UnsupervisedLearning(
			sampling=self.entropy_sampling,
			multi_label=True)

		expected_labels = [
			['1'],
			['1', '2'],
			['3', '4']
		]

		for i in range(3):
			learning.educate(i, self.train_texts[i], expected_labels[i])

		learn_indices, learn_x, learn_y = learning.get_learnt_data()
		assert expected_labels == learn_y, 'Unable to learn multi label'
