from typing import Tuple
import numpy as np


class Sampling:
	def __init__(self, name: str = 'sampling'):
		self.name = name

	def sample(self, data: np.ndarray, 
		num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
		...