from typing import List, Union
import numpy as np

class Embeddings:
	def __init__(self, batch_size: int = 16, name: str = 'embeddings'):
		self.batch_size = batch_size
		self.name = name

	def convert(self, inputs: Union[List[str], List[np.ndarray]]) -> List[float]:
		...
	
	def batch(self, arr: List, n: int = 1):
		size = len(arr)
		for i in range(0, size, n):
			yield arr[i: min(i+n, size)]