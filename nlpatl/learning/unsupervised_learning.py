from typing import List, Union, Callable, Optional

from nlpatl.models.clustering import Clustering
from nlpatl.models.embeddings import Embeddings
from nlpatl.learning import Learning
from nlpatl.dataset import Dataset


class UnsupervisedLearning(Learning):
    """
    | Applying unsupervised learning apporach to annotate the most valuable data points.
            You may refer to https://homepages.tuni.fi/tuomas.virtanen/papers/active-learning-sound.pdf.
            Here is the pseudo:
    |	1. [NLPatl] Convert raw data to features (Embeddings model)
    |	2. [NLPatl] Train model and clustering data points (Clustering model)
    |	3. [NLPatl] Estmiate the most valuable data points (Sampling)
    |	4. [Human] Subject matter experts annotates the most valuable data points
    |	5. Repeat Step 2 to 4 until acquire enough data points.

    :param sampling: Sampling method for get the most valuable data points.
            Providing certified methods name (`most_confidence`, `entropy`,
            `least_confidence`, `margin`, `nearest_mean`, `fathest`)
            or custom function.
    :type sampling: str or function
    :param embeddings: Function for converting raw data to embeddings. Providing
            model name according to embeddings type. For example, `multi-qa-MiniLM-L6-cos-v1`
            for `sentence_transformers`. bert-base-uncased` for
            `transformers`. `vgg16` for `torch_vision`.
    :type embeddings: str or :class:`nlpatl.models.embeddings.Embeddings`
    :param embeddings_model_config: Configuration for embeddings models. Optional. Ignored
            if using custom embeddings class
    :type embeddings_model_config: dict
    :param embeddings_type: Type of embeddings. `sentence_transformers` for text,
            `transformers` for text or `torch_vision` for image
    :type embeddings_type: str
    :param clustering: Function for clustering inputs. Either providing
            certified methods (`kmeans`) or custom function.
    :type clustering: str or :class:`nlpatl.models.clustering.Clustering`
    :param clustering_model_config: Configuration for clustering models. Optional. Ignored
            if using custom clustering class
    :type clustering_model_config: dict
    :param multi_label: Indicate the classification model is multi-label or
            multi-class (or binary). Default is False.
    :type multi_label: bool
    :param name: Name of this learning.
    :type name: str
    """

    def __init__(
        self,
        sampling: Union[str, Callable],
        embeddings: Union[str, Embeddings],
        clustering: Union[str, Clustering],
        embeddings_type: Optional[str] = None,
        embeddings_model_config: Optional[dict] = None,
        clustering_model_config: Optional[dict] = None,
        multi_label: bool = False,
        name: str = "unsupervised_learning",
    ):

        super().__init__(
            sampling=sampling,
            embeddings=embeddings,
            embeddings_type=embeddings_type,
            embeddings_model_config=embeddings_model_config,
            clustering=clustering,
            clustering_model_config=clustering_model_config,
            multi_label=multi_label,
            name=name,
        )

    def validate(self):
        super().validate(["embeddings", "clustering"])

    def explore(
        self, inputs: List[str], return_type: str = "dict", num_sample: int = 2
    ) -> Union[Dataset, dict]:

        self.validate()

        features = self.embeddings_model.convert(inputs)

        self.clustering_model.train(features)
        preds = self.clustering_model.predict_proba(features)

        indices, values = self.sampling(
            preds.values, preds.groups, num_sample=num_sample
        )
        preds.keep(indices)
        # Replace original probabilies by sampling values
        preds.values = values

        preds.inputs = [inputs[i] for i in preds.indices.tolist()]

        return self.get_return_object(preds, return_type)
