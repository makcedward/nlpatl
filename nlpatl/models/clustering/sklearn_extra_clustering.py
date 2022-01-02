from typing import List, Union
from collections import defaultdict
import numpy as np
from sklearn_extra.cluster import (
	KMedoids
)

from nlpatl.models.clustering import SkLearnClustering
from nlpatl.dataset import Dataset

MODEL_FOR_SKLEARN_EXTRA_CLUSTERING_MAPPING_NAMES = {
    'kmedoids': KMedoids
}


class SkLearnExtraClustering(SkLearnClustering):
	"""
		A wrapper of sci-kit learn extra clustering class.

		:param model_name: sci-kit learn extra clustering model name. Possible values
			are `kmedoids`.
		:type model_name: str
		:param model_config: Model paramateters. Refer to https://scikit-learn-extra.readthedocs.io/en/stable/api.html#clustering
		:type model_config: dict
		:param name: Name of this clustering
		:type name: str

		>>> import nlpatl.models.clustering as nmclu
		>>> model = nmclu.SkLearnExtraClustering()
    """

	def __init__(self, model_name: str = 'kmedoids', model_config: dict = {}, 
		name: str = 'sklearn_extra_clustering'):

		super().__init__(model_name=model_name, model_config=model_config, name=name)

	@staticmethod
	def get_mapping() -> dict:
		return MODEL_FOR_SKLEARN_EXTRA_CLUSTERING_MAPPING_NAMES

	def train(self, x: Union[List[float], np.ndarray]):
		self.model.fit(x)

	def predict_proba(self, x: np.ndarray, predict_config: dict={}) -> Dataset:
		"""
			:param x: Raw features
			:type x: np.ndarray
			:param predict_config: Model prediction paramateters. Refer to https://scikit-learn-extra.readthedocs.io/en/stable/api.html#clustering
			:type model_config: dict

			:return: Feature and probabilities
			:rtype: :class:`nlptatl.dataset.Dataset`
		"""
		return super().predict_proba(x=x, predict_config=predict_config)
