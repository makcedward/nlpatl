from typing import Tuple
import numpy as np

from nlpatl.sampling import Sampling


class MostConfidenceSampling(Sampling):
    """
    Sampling data points if the confidence is higher than threshold. Refer to https://markcartwright.com/files/wang2019active.pdf

    :param threshold: Minimum probability of model prediction. Default
            value is 0.85
    :type threshold: float
    :param name: Name of this sampling
    :type name: str
    """

    def __init__(self, threshold: float = 0.85, name: str = "most_confidence_sampling"):

        super().__init__(name=name)

        self.threshold = threshold

    def sample(
        self, data: np.ndarray, num_sample: int
    ) -> Tuple[np.ndarray, np.ndarray]:

        num_node = min(num_sample, len(data))

        # Calucalte most confidence
        most_confidences = np.max(data, axis=1)
        indices = np.argwhere(most_confidences > self.threshold).flatten()

        # It is possible that no result
        if len(indices) > 0:
            indices = np.random.choice(indices, num_node)
            return indices, most_confidences[indices]

        return np.empty(0), np.empty(0)
