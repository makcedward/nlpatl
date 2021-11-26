from typing import List


class Embeddings:
	def __init__(self, name: str):
		self.name = name

	def convert(self, inputs: List[str]) -> List[float]:
		...
	
	def batch(self, arr: List, n: int = 1):
		size = len(arr)
		for i in range(0, size, n):
			yield arr[i: min(i+n, size)]