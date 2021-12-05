from scipy.stats import entropy
import numpy as np

from nlpatl.models.classification.classification import Classification
from nlpatl.models.embeddings.embeddings import Embeddings
from nlpatl.learning.supervised_learning import SupervisedLearning
from nlpatl.storage import Storage


class LeastConfidenceLearning(SupervisedLearning):
	"""
		:param bool multi_label: Problem is mulit-label or not. Default is False
		:param obj embeddings_model: Embeddings models from 
			nlpatl.models.embeddings
		:param obj classification_model: Classification models from 
			nlpatl.models.embeddings
		:param str name: Name of this embeddings

		>>> import nlpatl.learning as nl
		>>> model = nl.LeastConfidenceLearning()
    """

	def __init__(self, multi_label: bool = False, 
		embeddings_model: Embeddings = None, 
		classification_model: Classification = None, 
		name: str = 'least_confidence_sampling'):
	
		super().__init__(multi_label=multi_label, 
			embeddings_model=embeddings_model,
			classification_model=classification_model,
			name=name)

	def keep_most_valuable(self, data: Storage, 
		num_sample: int) -> Storage:

		num_node = min(num_sample, len(data))

		# Calucalte least confidence
		least_confidences = 1 - np.max(data.values, axis=1)
		indices = np.argpartition(-least_confidences, num_node-1)[:num_node]

		data.keep(indices)
		
		# Replace probabilies by least_confidences
		data.values = least_confidences[indices]

		return data
