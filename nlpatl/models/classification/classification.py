from scipy.stats import entropy
import numpy as np


class Classification:
	def __init__(self, name='classification'):
		self.name = name

	@staticmethod
	def get_mapping(self) -> dict:
		...

	def predict_proba(self, x):
		...

	def entropy(self, probs):
		return entropy(probs, axis=1)

