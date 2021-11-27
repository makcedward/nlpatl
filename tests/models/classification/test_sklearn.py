import unittest

from nlpatl.models.classification.sklearn import SkLearn


class TestModelClassificationSkLearn(unittest.TestCase):
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
		classification = SkLearn('logistic_regression')
		assert 'l2' == classification.model.penalty, \
			'Invalid when using default parameters'

		model_config = {}
		classification = SkLearn('logistic_regression', 
			model_config=model_config)
		assert 'l2' == classification.model.penalty, \
			'Invalid when passing emtpy parameters'

		model_config = {'penalty': 'l1'}
		classification = SkLearn('logistic_regression', 
			model_config=model_config)
		assert 'l1' == classification.model.penalty, \
			'Invalid when passing parameter'

	def test_classify(self):
		classification = SkLearn('logistic_regression')
		classification.train(self.train_features, self.train_labels)
		results = classification.predict_proba(self.train_features)

		num_actual_class = len(classification.model.classes_)
		num_expected_class = len(results)

		assert num_actual_class == num_expected_class, \
			'{} expected classes is different from {} actual classes'.format(
				num_actual_class, num_expected_class)
		