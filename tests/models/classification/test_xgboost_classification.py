import unittest
import sklearn
import xgboost

from nlpatl.models.classification import XGBoostClassification


class TestModelClassificationXGBoost(unittest.TestCase):
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
		model_config = {}
		classification = XGBoostClassification(
			model_config=model_config)
		assert type(classification.model) is xgboost.sklearn.XGBClassifier, \
			'Unable to initialize XGBoost'

	def test_classify(self):
		classification = XGBoostClassification()
		classification.train(self.train_features, self.train_labels)
		result = classification.predict_proba(self.train_features)

		num_actual_class = len(classification.model.classes_)
		num_expected_class = len(set(result.groups))

		assert num_actual_class == num_expected_class, \
			'{} expected classes is different from {} actual classes'.format(
				num_actual_class, num_expected_class)
		assert result.groups, 'Missed groups attribute'
		assert result.values is not None, 'Missed values attribute'
		assert result.indices is not None, 'Missed indices attribute'
