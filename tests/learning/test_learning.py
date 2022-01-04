from typing import List, Union, Tuple
from datasets import load_dataset
import unittest
import datasets
import numpy as np

from nlpatl.models.embeddings import Embeddings, Transformers
from nlpatl.models.clustering import SkLearnClustering
from nlpatl.models.classification import SkLearnClassification
from nlpatl.learning import (
    SupervisedLearning,
    UnsupervisedLearning,
    SemiSupervisedLearning,
)
from nlpatl.dataset import Dataset


class TestLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts = load_dataset("ag_news")["train"]["text"]
        labels = load_dataset("ag_news")["train"]["label"]
        cls.train_texts = texts[0:5] + texts[200:205] + texts[1000:1005]
        cls.train_labels = labels[0:5] + labels[200:205] + labels[1000:1005]
        cls.test_texts = texts[0:10] + texts[200:210]
        cls.test_labels = labels[0:10] + labels[200:210]

        cls.transformers_embeddings_model = Transformers(
            "bert-base-uncased", nn_fwk="pt", padding=True, batch_size=3
        )
        cls.sklearn_clustering_model = SkLearnClustering("kmeans")
        cls.sklearn_classification_model = SkLearnClassification(
            "logistic_regression", model_config={"max_iter": 500}
        )

    def test_uncertainty_educate(self):
        learning = SupervisedLearning(
            sampling="entropy",
            embeddings=self.transformers_embeddings_model,
            classification=self.sklearn_classification_model,
        )
        learning.learn(self.train_texts, self.train_labels)
        first_result = learning.explore(self.test_texts, num_sample=2)

        assert not learning.learn_x, "Learnt something at the beginning"

        test_texts = self.test_texts
        for index, inputs, feature, group in zip(
            first_result["indices"],
            first_result["inputs"],
            first_result["features"],
            first_result["groups"],
        ):

            learning.educate(index, inputs, feature, group)

        assert learning.learn_x, "Unable to explore"

        # retrain with learnt data
        learning.learn(self.train_texts, self.train_labels)
        second_result = learning.explore(self.test_texts, num_sample=2)

        # Expect learnt record should be be picked again
        assert len(second_result["features"]) > 0, "No learnt data"
        for f in second_result["inputs"]:
            assert f not in first_result["inputs"], "Low quality of learnt data"

    def test_return_type(self):
        learning = UnsupervisedLearning(
            sampling="nearest_mean",
            embeddings=self.transformers_embeddings_model,
            clustering=self.sklearn_clustering_model,
        )
        result = learning.explore(self.train_texts, return_type="object")

        assert result.inputs, "Not object format"

        learning = SupervisedLearning(
            sampling="entropy",
            embeddings=self.transformers_embeddings_model,
            classification=self.sklearn_classification_model,
        )
        learning.learn(self.train_texts, self.train_labels)
        result = learning.explore(self.test_texts, num_sample=2, return_type="object")

        assert result.inputs, "Not object format"

    def test_educate_multi_label(self):
        learning = UnsupervisedLearning(
            sampling="entropy",
            embeddings=self.transformers_embeddings_model,
            clustering=self.sklearn_clustering_model,
            multi_label=True,
        )

        expected_labels = [["1"], ["1", "2"], ["3", "4"]]

        for i in range(3):
            x_features = learning.embeddings_model.convert(self.train_texts[i])[0]
            learning.educate(i, self.train_texts[i], x_features, expected_labels[i])

        learn_indices, learn_x, learn_x_features, learn_y = learning.get_learn_data()
        assert expected_labels == learn_y, "Unable to learn multi label"

    def test_custom_sampling(self):
        def custom_sampling(
            data: np.ndarray, num_sample: int
        ) -> Tuple[np.ndarray, np.ndarray]:

            return np.array([1, 3, 5]), None

        learning = SupervisedLearning(
            sampling=custom_sampling,
            embeddings="bert-base-uncased",
            embeddings_type="transformers",
            embeddings_model_config={"nn_fwk": "pt", "padding": True, "batch_size": 8},
            classification="logistic_regression",
        )

        learning.learn(self.train_texts, self.train_labels)
        learning.explore(self.test_texts)

        assert True, "Unable to apply custom sampling in SupervisedLearning"

    def test_custom_embeddings_model(self):
        class CustomEmbeddings(Embeddings):
            def convert(self, inputs: List[str]) -> np.ndarray:
                return np.random.rand(len(inputs), 5)

        learning = UnsupervisedLearning(
            sampling="nearest_mean",
            embeddings=CustomEmbeddings(),
            clustering="kmeans",
            multi_label=False,
        )
        learning.explore(self.train_texts)

        assert True, "Unable to apply custom embeddings model in UnsupervisedLearning"

        learning = SupervisedLearning(
            sampling="entropy",
            embeddings=CustomEmbeddings(),
            classification=self.sklearn_classification_model,
            multi_label=True,
        )

        learning.learn(self.train_texts, self.train_labels)

        assert True, "Unable to apply custom embeddings model in SupervisedLearning"

        learning = SemiSupervisedLearning(
            sampling="most_confidence",
            multi_label=True,
            embeddings=CustomEmbeddings(),
            classification=self.sklearn_classification_model,
        )

        learning.learn(self.train_texts, self.train_labels)

        assert True, "Unable to apply custom embeddings model in SemiSupervisedLearning"
