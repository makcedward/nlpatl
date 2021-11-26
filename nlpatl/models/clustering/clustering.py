from typing import List


class Clustering:
	def __init__(self, name: str):
		self.name = name

	def convert(self, inputs: List[str]) -> List[float]:
		...