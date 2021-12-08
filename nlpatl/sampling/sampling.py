from typing import Tuple
import numpy as np


class Sampling:
	def __init__(self, name: str = 'sampling'):
		self.name = name

	def sample(self, x: np.ndarray, 
		num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
		"""
			:param x: Values of determine the sampling
			:type x: :class: `np.ndarray`
			:param num_sample: Total number of sample for labeling
			:type num_sample: int

			:return: Tuple of target indices and sampling values
			:rtype: Tuple of :class:`numpy.ndarray`, :class:`numpy.ndarray`
		"""
		...