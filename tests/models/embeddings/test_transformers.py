import unittest

from nlpatl.models.embeddings.transformers import Transformers


class TestModelEmbeddingsTransformers(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.train_texts = ['The quick brown fox jumps of the lazy dog.'] * 5

	def test_convert(self):
		embeddings = Transformers(model_name_or_path='bert-base-uncased',
			batch_size=3, padding=True, return_tensors='pt')
		embs = embeddings.convert(self.train_texts)

		assert len(self.train_texts) == len(embs), \
			'Number of input does not equal to number of outputs'

	def test_unsupport_model(self):
		embeddings = Transformers(model_name_or_path='distilbert-base-uncased',
			batch_size=3, padding=True, return_tensors='pt')

		with self.assertRaises(Exception) as error:
			embs = embeddings.convert(self.train_texts)
		assert 'does not provide single embeddings for input' in str(error.exception), \
			'Unable to handle unsupported embeddings model'