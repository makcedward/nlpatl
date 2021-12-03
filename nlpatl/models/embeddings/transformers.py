from typing import List
from transformers import (
	AutoTokenizer, 
	AutoModel, TFAutoModel 
)
import numpy as np

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
	def __init__(self, model_name_or_path: str, batch_size: int = 16, 
		padding: bool = False, truncation: bool = False, 
		return_tensors: str = None, name: str = 'transformer'):

		super().__init__(batch_size=batch_size, name=name)

		self.model_name_or_path = model_name_or_path
		self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
		if return_tensors == 'pt':
			self.model = AutoModel.from_pretrained(model_name_or_path)
			self.model.eval()
		elif return_tensors == 'tf':
			self.model = TFAutoModel.from_pretrained(model_name_or_path)
			# TODO: have eval ?

		self.padding = padding
		self.truncation = truncation
		self.return_tensors = return_tensors

	def convert(self, inputs: List[str]) -> np.ndarray:
		results = []
		for batch_inputs in self.batch(inputs, self.batch_size):
			ids = self.tokenizer(
				batch_inputs, 
				return_tensors=self.return_tensors, 
				padding=self.padding, 
				truncation=self.truncation)

			with torch.no_grad():
				output = self.model(**ids)
				assert 'pooler_output' in output.keys(), \
					'This model (`{}`) does not provide single embeddings for ' \
					'input. Switch to use other type of transformers model such as '\
					'`bert-base-uncased` or `roberta-base`.'.format(self.model_name_or_path)

				results.append(output['pooler_output'])

		if self.return_tensors == 'pt':
			return torch.cat(results).numpy()
		elif self.return_tensors == 'tf':
			return tf.concat(results, axis=0).numpy()
