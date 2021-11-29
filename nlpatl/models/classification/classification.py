from typing import List


class Classification:
	def __init__(self, name='classification'):
		self.name = name

	@staticmethod
	def get_mapping(self) -> dict:
		...

	def predict_proba(self, x):
		...

	def build_label_encoder(self, labels: [List[str], List[int]]):
		uni_labels = sorted(set(labels))

		self.label_encoder = {c:i for i, c in enumerate(uni_labels)}
		self.label_decoder = {i:c for c, i in self.label_encoder.items()}
