from typing import List

from nlpatl.models.clustering import Clustering
from nlpatl.models.embeddings import Embeddings
from nlpatl.learning import Learning
from nlpatl.sampling import Sampling
from nlpatl.storage import Storage


class UnsupervisedLearning(Learning):
	def __init__(self, 
		sampling: Sampling,
		embeddings_model: Embeddings = None, 
		clustering_model: Clustering = None, 
		multi_label: bool = False, 
		name: str = 'unsupervised_samlping'):

		super().__init__(sampling=sampling,
			embeddings_model=embeddings_model,
			clustering_model=clustering_model,
			multi_label=multi_label, 
			name=name)

	def validate(self):
		super().validate(['embeddings', 'clustering'])

	def explore(self, inputs: List[str], return_type: str = 'dict', 
		num_sample: int = 2) -> [Storage, dict]:

		self.validate()

		features = self.embeddings_model.convert(inputs)

		self.clustering_model.train(features)
		preds = self.clustering_model.predict_proba(features)

		indices, values = self.sampling.sample(
			preds.values, preds.groups, num_sample=num_sample)
		preds.keep(indices)
		# Replace original probabilies by sampling values
		preds.values = values

		preds.features = [inputs[i] for i in preds.indices.tolist()]

		return self.get_return_object(preds, return_type)
		