from datasets import load_dataset
import unittest

from nlpatl.learning import UnsupervisedLearning
	

class TestLearningUnsupervised(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.train_texts = load_dataset('ag_news')['train']['text'][:20]

		cls.learning = UnsupervisedLearning(
			sampling='nearest_mean',
			embeddings='bert-base-uncased', embeddings_type='transformers',
			embeddings_model_config={'nn_fwk': 'pt', 'padding': True, 'batch_size':8},
			clustering='kmeans',
			multi_label=False, 
			)

	def tearDown(self):
		self.learning.clear_learn_data()

	def test_explore(self):
		result = self.learning.explore(self.train_texts)

		assert not self.learning.learn_x, 'Learnt something at the beginning'

		for index, feature, group in zip(
			result['indices'], result['features'], result['groups']):

			self.learning.educate(index, feature, group)

		assert self.learning.learn_x, 'Unable to explore'
