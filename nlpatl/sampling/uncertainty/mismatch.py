from typing import Tuple, List, Union
from scipy.stats import entropy
import numpy as np

from nlpatl.sampling import Sampling


class MismatchSampling(Sampling):
    """
    Sampling data points according to the mismatch. Pick the N data points
            randomly.

    :param name: Name of this sampling
    :type name: str
    """

    def __init__(self, name: str = "mismatch_sampling"):
        super().__init__(name=name)

    def sample(
        self,
        data1: Union[List[str], List[int], List[float], np.ndarray],
        data2: Union[List[str], List[int], List[float], np.ndarray],
        num_sample: int,
    ) -> Tuple[np.ndarray, np.ndarray]:

        assert len(data1) == len(data2), "Two list of data have different size."

        # Find mismatch
        mismatch_indices = []
        for i, (d1, d2) in enumerate(zip(data1, data2)):
            if d1 != d2:
                mismatch_indices.append(i)

        num_node = min(num_sample, len(mismatch_indices))

        mismatch_indices = np.random.choice(mismatch_indices, num_node, replace=False)

        return mismatch_indices, None
