from sklearn.datasets import load_sample_images
import unittest
import torchvision.transforms as transforms

from nlpatl.models.embeddings.torchvision import TorchVision


class TestModelEmbeddingsTorchVision(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		dataset = load_sample_images()
		
		transformation = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		cls.train_images = [transformation(img) for img in dataset.images * 3]

	def test_convert(self):
		embeddings = TorchVision(model_name_or_path='vgg16',
			batch_size=3)
		embs = embeddings.convert(self.train_images)

		assert len(self.train_images) == len(embs), \
			'Number of input does not equal to number of outputs'

	def test_unsupport_model(self):
		with self.assertRaises(Exception) as error:
			TorchVision(model_name_or_path='unsupported',
				batch_size=3)
		assert 'does not support. Supporting' in str(error.exception), \
			'Unable to handle unsupported embeddings model'
