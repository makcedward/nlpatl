import unittest

from nlpatl.models.embeddings import SentenceTransformers


class TestModelEmbeddingsTransformers(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.train_texts = ['The quick brown fox jumps of the lazy dog.'] * 5

		cls.embeddings = SentenceTransformers(
			model_name_or_path='multi-qa-MiniLM-L6-cos-v1',
			batch_size=4)

	def test_convert(self):
		embs = self.embeddings.convert(self.train_texts)

		assert len(self.train_texts) == len(embs), \
			'Number of input does not equal to number of outputs'
