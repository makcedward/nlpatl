"""
	sci-kit learn classification wrapper
"""

from typing import List, Union
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from nlpatl.models.classification.classification import Classification
from nlpatl.dataset import Dataset

MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES = {
    "logistic_regression": LogisticRegression,
    "svc": SVC,
    "linear_svc": LinearSVC,
    "random_forest": RandomForestClassifier,
    "sgd": SGDClassifier,
    "knn": KNeighborsClassifier,
    "gbdt": GradientBoostingClassifier,
}


class SkLearnClassification(Classification):
    """
    A wrapper of sci-kit learn classification class.

    :param model_name: sci-kit learn classification model name. Possible values
            are `logistic_regression`, `svc`, `linear_svc` and `random_forest`.
    :type model_name: str
    :param model_config: Model paramateters. Refer to https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
    :type model_config: dict
    :param name: Name of this classification
    :type name: str

    >>> import nlpatl.models.classification as nmcla
    >>> model = nmcla.SkLearnClassification()
    """

    def __init__(
        self,
        model_name: str = "logistic_regression",
        model_config: dict = {},
        name: str = "sklearn_classification",
    ):

        super().__init__(name=name)

        self.model_name = model_name
        self.model_config = model_config

        if model_name in MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES:
            self.model = MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES[model_name](
                **model_config
            )
        else:
            raise ValueError(
                "`{}` does not support. Supporting {} only".format(
                    model_name,
                    "`"
                    + "`".join(MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES.keys())
                    + "`",
                )
            )

    @staticmethod
    def get_mapping() -> dict:
        return MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES

    def train(
        self,
        x: np.ndarray,
        y: Union[np.ndarray, List[str], List[int], List[List[str]], List[List[int]]],
    ):

        self.build_label_encoder(y)
        y_encoded = [self.label_encoder[lab] for lab in y]
        self.model.fit(x, y_encoded)

    def predict_proba(self, x: np.ndarray, predict_config: dict = {}) -> Dataset:
        """
        :param x: Raw features
        :type x: np.ndarray
        :param predict_config: Model prediction paramateters. Refer to https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
        :type model_config: dict

        :return: Feature and probabilities
        :rtype: :class:`nlptatl.dataset.Dataset`
        """

        probs = self.model.predict_proba(x, **predict_config)
        preds = np.argmax(probs, axis=1)

        return Dataset(features=x, values=probs, groups=preds.tolist())
