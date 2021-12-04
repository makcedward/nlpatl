from typing import List
import numpy as np

try:
	import torch
	from sentence_transformers import SentenceTransformer
except ImportError:
	# No installation required if not using this function
	pass

from nlpatl.models.embeddings import Embeddings


class SentenceTransformers(Embeddings):
	"""
		:param str model_name_or_path: Sentence transformers model or path name
		:param int batch_size: Batch size of data processing. Default is 16
		:param dict model_config: Custom model paramateters
		:param str name: Name of this embeddings

		>>> import nlpatl.models.embeddings as nme
		>>> model = nme.SentenceTransformers()
    """

	def __init__(self, model_name_or_path: str, batch_size: int = 16, 
		name: str = 'sentence_transformers'):

		super().__init__(batch_size=batch_size, name=name)

		self.model_name_or_path = model_name_or_path
		self.model = SentenceTransformer(model_name_or_path)
		self.model.eval()

	def convert(self, x: List[str]) -> np.ndarray:
		"""
			:param list x: Raw features

			:return np.ndarray: Embeddings
			
			>>> model.convert(x=x)
		"""

		results = []
		for i, batch_inputs in enumerate(self.batch(x, self.batch_size)):
			with torch.no_grad():
				results.extend(self.model.encode(batch_inputs))
		return np.array(results)
