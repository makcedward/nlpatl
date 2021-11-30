from typing import List
from nlpatl.storage.storage import Storage


class Clustering:
	def __init__(self, name: str = 'clustering'):
		self.name = name

	def train(self, inputs: List[float]):
		...

	def predict_proba(self, inputs: List[float], 
		predict_config: dict={}) -> Storage:
		...
