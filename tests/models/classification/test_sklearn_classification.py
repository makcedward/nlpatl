import unittest
import sklearn

from nlpatl.models.classification import SkLearnClassification


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

		classification = SkLearnClassification('random_forest')
		assert type(classification.model) is sklearn.ensemble._forest.RandomForestClassifier, \
			'Unable to initialize Random Forest'

		classification = SkLearnClassification('sgd')
		assert type(classification.model) is sklearn.linear_model._stochastic_gradient.SGDClassifier, \
			'Unable to initialize SGD'

		classification = SkLearnClassification('knn')
		assert type(classification.model) is sklearn.neighbors._classification.KNeighborsClassifier, \
			'Unable to initialize KNN'

		classification = SkLearnClassification('gbdt')
		assert type(classification.model) is sklearn.ensemble._gb.GradientBoostingClassifier, \
			'Unable to initialize GBDT'

	def test_classify(self):
		classification = SkLearnClassification('logistic_regression')
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
