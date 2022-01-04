import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    # No installation required if not using this function
    pass

from nlpatl.dataset import Dataset
from nlpatl.models.classification import SkLearnClassification


class XGBoostClassification(SkLearnClassification):
    """
    A wrapper of xgboost classification class.

    :param model_config: Model paramateters. Refer to https://xgboost.readthedocs.io/en/stable/python/python_api.html
    :type model_config: dict
    :param name: Name of this classification
    :type name: str

    >>> import nlpatl.models.classification as nmcla
    >>> model = nmcla.XGBoostClassification()
    """

    def __init__(self, model_config: dict = {}, name: str = "xgboost_classification"):

        super().__init__(name=name)

        self.model_name = "xgboost"
        self.model_config = model_config

        self.model = XGBClassifier(**model_config)

    @staticmethod
    def get_mapping() -> dict:
        return MODEL_FOR_SKLEARN_CLASSIFICATION_MAPPING_NAMES

    def predict_proba(self, x: np.ndarray, predict_config: dict = {}) -> Dataset:
        """
        :param x: Raw features
        :type x: np.ndarray
        :param predict_config: Model prediction paramateters. Refer to https://xgboost.readthedocs.io/en/stable/python/python_api.html
        :type model_config: dict

        :return: Feature and probabilities
        :rtype: :class:`nlptatl.dataset.Dataset`
        """
        return super().predict_proba(x=x, predict_config=predict_config)
