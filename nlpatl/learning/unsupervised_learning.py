from typing import List
from collections import defaultdict
import numpy as np

from nlpatl.models.clustering.clustering import Clustering
from nlpatl.models.embeddings.embeddings import Embeddings
from nlpatl.learning.learning import Learning
from nlpatl.storage.storage import Storage


class UnsupervisedLearning(Learning):
	def __init__(self, multi_label: bool = False, 
		embeddings_model: Embeddings = None, 
		clustering_model: Clustering = None, 
		name: str = 'unsupervised_samlping'):

		super().__init__(multi_label=multi_label, 
			embeddings_model=embeddings_model,
			clustering_model=clustering_model,
			name=name)

	def validate(self):
		super().validate(['embeddings', 'clustering'])

	def explore(self, inputs: List[str], return_type: str = 'dict', 
		num_sample: int = 2) -> [Storage, dict]:

		self.validate()

		features = self.embeddings_model.convert(inputs)

		self.clustering_model.train(features)
		preds = self.clustering_model.predict_proba(features)

		preds = self.keep_most_valuable(preds, num_sample=num_sample)

		preds.features = [inputs[i] for i in preds.indices.tolist()]

		return self.get_return_object(preds, return_type)
		