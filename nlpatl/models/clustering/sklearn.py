from __future__ import annotations
from typing import List

from nlpatl.models.clustering.clustering import Clustering


class SkLearn(Clustering):
    def __init__(self, name: str = 'sklearn'):
        super().__init__(name=name)

    def fit(self, inputs: List[float]) -> SkLearn:
        self.model.fit(inputs)
        return self

    def cluster(self, inputs: List[float], num_nearest: int = 5) -> List[int]:
        ...

    def save(self) -> None:
        ...
