from typing import List
import numpy as np

from nlpatl.dataset import Dataset


class Clustering:
	def __init__(self, name: str = 'clustering'):
		self.name = name

	def train(self, x: List[float]):
		"""
			:param x: Raw features
			:type x: np.ndarray
		"""
		...

	def predict_proba(self, x: np.ndarray, 
		predict_config: dict={}) -> Dataset:
		...
