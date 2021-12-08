from typing import List, Union
import numpy as np

from nlpatl.dataset import Dataset

class Classification:
	def __init__(self, name: str = 'classification'):
		self.name = name

	def build_label_encoder(self, labels: Union[List[str], List[int]]):
		uni_labels = sorted(set(labels))

		self.label_encoder = {c:i for i, c in enumerate(uni_labels)}
		self.label_decoder = {i:c for c, i in self.label_encoder.items()}

	@staticmethod
	def get_mapping(self) -> dict:
		...

	def train(self, x: np.ndarray, 
		y: Union[np.ndarray, List[str], List[int], List[List[str]], List[List[int]]]):
		"""
			:param x: Raw features
			:type x: np.ndarray
			:param y: Label of data inputs
			:type y: list of string, int or float or :class:`np.ndarray`.
		"""
		...

	def predict_proba(self, x: np.ndarray, predict_config: dict={}) -> Dataset:
		...
	