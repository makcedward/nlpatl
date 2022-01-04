from typing import List
import numpy as np

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    # No installation required if not using this function
    pass

from nlpatl.models.embeddings import Embeddings


class SentenceTransformers(Embeddings):
    """
    A wrapper of transformers class.

    :param model_name_or_path: sentence transformers model name.
    :type model_name_or_path: str
    :param batch_size: Batch size of data processing. Default is 16
    :type batch_size: int
    :param model_config: Model paramateters. Refer to https://www.sbert.net/docs/pretrained_models.html
    :type model_config: dict
    :param name: Name of this embeddings
    :type name: str

    >>> import nlpatl.models.embeddings as nme
    >>> model = nme.SentenceTransformers()
    """

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 16,
        name: str = "sentence_transformers",
    ):

        super().__init__(batch_size=batch_size, name=name)

        self.model_name_or_path = model_name_or_path
        self.model = SentenceTransformer(model_name_or_path)
        self.model.eval()

    def convert(self, x: List[str]) -> np.ndarray:
        results = []
        for i, batch_inputs in enumerate(self.batch(x, self.batch_size)):
            with torch.no_grad():
                results.extend(self.model.encode(batch_inputs))
        return np.array(results)
