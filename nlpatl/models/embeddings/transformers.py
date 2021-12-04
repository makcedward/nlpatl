from typing import List
import numpy as np

try:
	from transformers import (
		AutoTokenizer, 
		AutoModel, TFAutoModel 
	)
except ImportError:
	# No installation required if not using this function
	pass
try:
	import torch
except ImportError:
	# No installation required if not using this function
	pass
try:
	import tensorflow as tf
except ImportError:
	# No installation required if not using this function
	pass

from nlpatl.models.embeddings.embeddings import Embeddings


class Transformers(Embeddings):
	"""
		:param str model_name_or_path: Transformers model or path name
		:param int batch_size: Batch size of data processing. Default is 16
		:param bool padding: Inputs may not have same size. Set True to pad it.
			Default is False
		:param bool truncation: Inputs may not have same size. Set True to 
			truncate it. Default is False
		:param str nn_fwk: Neual network framework. Either pt (for PyTorch) or
			tf (for TensorFlow)
		:param dict model_config: Custom model paramateters
		:param str name: Name of this embeddings

		>>> import nlpatl.models.embeddings as nme
		>>> model = nme.Transformers()
    """

	def __init__(self, model_name_or_path: str, batch_size: int = 16, 
		padding: bool = False, truncation: bool = False, 
		nn_fwk: str = None, name: str = 'transformers'):

		super().__init__(batch_size=batch_size, name=name)

		self.model_name_or_path = model_name_or_path
		self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
		if nn_fwk == 'pt':
			self.model = AutoModel.from_pretrained(model_name_or_path)
			self.model.eval()
		elif nn_fwk == 'tf':
			self.model = TFAutoModel.from_pretrained(model_name_or_path)
			# TODO: have eval ?

		self.padding = padding
		self.truncation = truncation
		self.nn_fwk = nn_fwk

	def convert(self, inputs: List[str]) -> np.ndarray:
		"""
			:param list x: Raw features

			:return np.ndarray: Embeddings
			
			>>> model.convert(x=x)
		"""

		results = []
		for batch_inputs in self.batch(inputs, self.batch_size):
			ids = self.tokenizer(
				batch_inputs, 
				return_tensors=self.nn_fwk, 
				padding=self.padding, 
				truncation=self.truncation)

			# TODO: for tensorflow?
			with torch.no_grad():
				output = self.model(**ids)
				assert 'pooler_output' in output.keys(), \
					'This model (`{}`) does not provide single embeddings for ' \
					'input. Switch to use other type of transformers model such as '\
					'`bert-base-uncased` or `roberta-base`.'.format(self.model_name_or_path)

				results.append(output['pooler_output'])

		if self.nn_fwk == 'pt':
			return torch.cat(results).numpy()
		elif self.nn_fwk == 'tf':
			return tf.concat(results, axis=0).numpy()
