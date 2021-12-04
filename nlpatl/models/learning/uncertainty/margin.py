import numpy as np

from nlpatl.models.classification.classification import Classification
from nlpatl.models.embeddings.embeddings import Embeddings
from nlpatl.models.learning.supervised_learning import SupervisedLearning
from nlpatl.storage.storage import Storage


class MarginLearning(SupervisedLearning):
	"""
		:param bool multi_label: Problem is mulit-label or not. Default is False
		:param obj embeddings_model: Embeddings models from 
			nlpatl.models.embeddings
		:param obj classification_model: Classification models from 
			nlpatl.models.embeddings
		:param str name: Name of this embeddings

		>>> import nlpatl.models.learning as nml
		>>> model = nml.MarginLearning()
    """

	def __init__(self, multi_label: bool = False, 
		embeddings_model: Embeddings = None, 
		classification_model: Classification = None, 
		name: str = 'margin_sampling'):
	
		super().__init__(multi_label=multi_label, 
			embeddings_model=embeddings_model,
			classification_model=classification_model,
			name=name)		

	def keep_most_valuable(self, data: Storage, 
		num_sample: int) -> Storage:

		num_node = min(num_sample, len(data))

		# Calculate margin difference between first and second highest probabilties
		margin_diffs = np.partition(-data.values, 1, axis=1)
		margin_diffs = -margin_diffs[:, 0] + margin_diffs[:, 1]

		indices = np.argpartition(margin_diffs, num_node-1)[:num_node]

		data.keep(indices)
		
		# Replace probabilies by margin differences
		data.values = margin_diffs[indices]

		return data
