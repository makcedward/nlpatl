import unittest
import sklearn
import xgboost

from nlpatl.models.classification.sklearn_classification import SkLearnClassification


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
		classification = SkLearnClassification('logistic_regression')
		assert 'l2' == classification.model.penalty, \
			'Invalid when using default parameters'

		model_config = {}
		classification = SkLearnClassification('logistic_regression', 
			model_config=model_config)
		assert 'l2' == classification.model.penalty, \
			'Invalid when passing emtpy parameters'

		model_config = {'penalty': 'l1'}
		classification = SkLearnClassification('logistic_regression', 
			model_config=model_config)
		assert 'l1' == classification.model.penalty, \
			'Invalid when passing parameter'

		model_config = {'gamma': 'auto'}
		classification = SkLearnClassification('svc', 
			model_config=model_config)
		assert 'auto' == classification.model.gamma, \
			'Invalid when passing emtpy parameters'
		assert type(classification.model) is sklearn.svm._classes.SVC, \
			'Unable to initialize SVM'

		classification = SkLearnClassification('linear_svc')
		assert type(classification.model) is sklearn.svm._classes.LinearSVC, \
			'Unable to initialize Linear SVM'

		model_config = {}
		classification = SkLearnClassification('xgboost', 
			model_config=model_config)
		assert type(classification.model) is xgboost.sklearn.XGBClassifier, \
			'Unable to initialize XGBoost'

	def test_classify(self):
		classification = SkLearnClassification('logistic_regression')
		classification.train(self.train_features, self.train_labels)
		results = classification.predict_proba(self.train_features)

		num_actual_class = len(classification.model.classes_)
		num_expected_class = len(results)

		assert num_actual_class == num_expected_class, \
			'{} expected classes is different from {} actual classes'.format(
				num_actual_class, num_expected_class)
		