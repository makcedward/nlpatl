from typing import List, Optional, Callable
import numpy as np

try:
	import torch
	import torchvision

	MODEL_FOR_TORCH_VISION_MAPPING_NAMES = {
		'resnet18': torchvision.models.resnet18,
		'alexnet': torchvision.models.alexnet,
		'vgg16': torchvision.models.vgg16
	}
except ImportError:
	# No installation required if not using this function
	pass

from nlpatl.models.embeddings.embeddings import Embeddings


class TorchVision(Embeddings):
	"""
		:param str model_name_or_path: transformers model or path name
		:param int batch_size: Batch size of data processing. Default is 16
		:param dict model_config: Custom model paramateters. Default is loading
			pretrained model.
		:param func transform: Preprocessing function
		:param str name: Name of this embeddings

		>>> import nlpatl.models.embeddings as nme
		>>> model = nme.Transformers()
    """

	def __init__(self, model_name_or_path: str, batch_size: int = 16, 
		model_config: dict = {'pretrained': True},
		transform: Optional[Callable] = None, 
		name: str = 'torchvision'):
		
		super().__init__(batch_size=batch_size, name=name)

		self.model_name_or_path = model_name_or_path
		self.model_config = model_config
		self.transform = transform

		if model_name_or_path in MODEL_FOR_TORCH_VISION_MAPPING_NAMES:
			self.model = MODEL_FOR_TORCH_VISION_MAPPING_NAMES[model_name_or_path](
				**model_config)
			self.model.eval()
		else:
			raise ValueError('`{}` does not support. Supporting {} only'.format(
				model_name_or_path, '`' + '`,`'.join(
					MODEL_FOR_TORCH_VISION_MAPPING_NAMES.keys()) + '`'))

	@staticmethod
	def get_mapping() -> dict:
		return MODEL_FOR_TORCH_VISION_MAPPING_NAMES

	def convert(self, x: List[np.ndarray]) -> np.ndarray:
		"""
			:param np.ndarray x: Raw features
			
			>>> model.convert(x=x)
		"""

		results = []
		for batch_inputs in self.batch(x, self.batch_size):
			with torch.no_grad():
				features = [
				self.transform(img) if self.transform else img for img in batch_inputs]
			results.append(self.model(torch.stack(features)))

		return torch.cat(results).detach().numpy()
