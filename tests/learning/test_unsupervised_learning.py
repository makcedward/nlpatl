from datasets import load_dataset
import unittest
import sklearn_extra

from nlpatl.learning import UnsupervisedLearning


class TestLearningUnsupervised(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_texts = load_dataset("ag_news")["train"]["text"][:20]

        cls.learning = UnsupervisedLearning(
            sampling="nearest_mean",
            embeddings="bert-base-uncased",
            embeddings_type="transformers",
            embeddings_model_config={"nn_fwk": "pt", "padding": True, "batch_size": 8},
            clustering="kmeans",
            multi_label=False,
        )

    def tearDown(self):
        self.learning.clear_learn_data()

    def test_sklearn_extra(self):
        # backup original config
        clustering_model_config = self.learning.clustering_model_config
        clustering_name = self.learning.clustering_name
        clustering_model = self.learning.clustering_model

        new_clustering_model_config = {}
        (
            self.learning.clustering_name,
            self.learning.clustering_model,
        ) = self.learning.init_clustering_model("kmedoids", new_clustering_model_config)
        assert (
            type(self.learning.clustering_model.model)
            is sklearn_extra.cluster._k_medoids.KMedoids
        ), "Unable to initialize KMedoids"

        result = self.learning.explore(self.train_texts)
        assert not self.learning.learn_x, "Learnt something at the beginning"

        # reset original params
        self.learning.clustering_model_config = clustering_model_config
        self.learning.clustering_name = clustering_name
        self.learning.clustering_model = clustering_model

    def test_explore(self):
        result = self.learning.explore(self.train_texts)

        assert not self.learning.learn_x, "Learnt something at the beginning"

        for index, inputs, feature, group in zip(
            result["indices"], result["inputs"], result["features"], result["groups"]
        ):

            self.learning.educate(index, inputs, feature, group)

        assert self.learning.learn_x, "Unable to explore"
