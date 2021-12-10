from typing import List, Union

from nlpatl.models.clustering import Clustering
from nlpatl.models.embeddings import Embeddings
from nlpatl.learning import Learning
from nlpatl.sampling import Sampling
from nlpatl.dataset import Dataset


class UnsupervisedLearning(Learning):
	"""
		| Applying unsupervised learning apporach to annotate the most valuable data points. 
			You may refer to https://homepages.tuni.fi/tuomas.virtanen/papers/active-learning-sound.pdf. 
			Here is the pseudo:
		|	1. Convert raw data to features (Embeddings model)
		|	2. Train model and clustering data points (Clustering model)
		|	3. Estmiate the most valuable data points (Sampling)
		|	4. Subject matter experts annotates the most valuable data points
		|	5. Repeat Step 2 to 4 until acquire enough data points.
		
		:param sampling: Sampling method. Refer to nlpatl.sampling.
		:type sampling: :class:`nlpatl.sampling.Sampling`
		:param embeddings_model: Function for converting raw data to embeddings.
		:type embeddings_model: :class:`nlpatl.models.embeddings.Embeddings`
		:param clustering_model: Function for clustering inputs
		:type clustering_model: :class:`nlpatl.models.clustering.Clustering`
		:param multi_label: Indicate the classification model is multi-label or 
			multi-class (or binary). Default is False.
		:type multi_label: bool
		:param name: Name of this learning.
		:type name: str
	"""

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
		num_sample: int = 2) -> Union[Dataset, dict]:

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
		