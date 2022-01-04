from typing import List, Union
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans

from nlpatl.models.clustering import Clustering
from nlpatl.dataset import Dataset

MODEL_FOR_SKLEARN_CLUSTERING_MAPPING_NAMES = {
    "kmeans": KMeans,
}


class SkLearnClustering(Clustering):
    """
    A wrapper of sci-kit learn clustering class.

    :param model_name: sci-kit learn clustering model name. Possible values
            are `kmeans`.
    :type model_name: str
    :param model_config: Model paramateters. Refer to https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
    :type model_config: dict
    :param name: Name of this clustering
    :type name: str

    >>> import nlpatl.models.clustering as nmclu
    >>> model = nmclu.SkLearnClustering()
    """

    def __init__(
        self,
        model_name: str = "kmeans",
        model_config: dict = {},
        name: str = "sklearn_clustering",
    ):

        super().__init__(name)

        self.model_name = model_name
        self.model_config = model_config

        if model_name in self.get_mapping():
            self.model = self.get_mapping()[model_name](**model_config)
        else:
            raise ValueError(
                "`{}` does not support. Supporting {} only".format(
                    model_name, "`" + "`,`".join(self.get_mapping().keys()) + "`"
                )
            )

    @staticmethod
    def get_mapping() -> dict:
        return MODEL_FOR_SKLEARN_CLUSTERING_MAPPING_NAMES

    def train(self, x: Union[List[float], np.ndarray]):
        self.model.fit(x)

    def predict_proba(self, x: List[float], predict_config: dict = {}) -> Dataset:

        """
        :param x: Raw features
        :type x: np.ndarray
        :param predict_config: Model prediction paramateters. Refer to https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
        :type predict_config: dict

        :return: Feature and probabilities
        :rtype: :class:`nlptatl.dataset.Dataset`
        """

        clust_dists = self.model.transform(x)
        preds = self.model.predict(x, **predict_config)
        total_record = len(preds)

        indices = np.zeros(total_record, dtype=int)
        values = np.zeros(total_record, dtype=np.float)
        groups = [-1] * total_record
        start_pos = 0
        for label in range(self.model.n_clusters):
            label_indices = np.where(preds == label)[0]
            end_pos = start_pos + len(label_indices)

            indices[start_pos:end_pos] = label_indices
            values[start_pos:end_pos] = clust_dists[label_indices][:, label]
            groups[start_pos:end_pos] = [label] * len(label_indices)

            start_pos = end_pos

        return Dataset(features=x, indices=indices, values=values, groups=groups)
