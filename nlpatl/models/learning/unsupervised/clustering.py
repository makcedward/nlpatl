from typing import List
from collections import defaultdict
import numpy as np

from nlpatl.models.clustering.clustering import Clustering
from nlpatl.models.embeddings.embeddings import Embeddings
from nlpatl.models.learning.unsupervised_learning import UnsupervisedLearning
from nlpatl.storage.storage import Storage


class ClusteringLearning(UnsupervisedLearning):
	"""
		:param bool multi_label: Problem is mulit-label or not. Default is False
		:param obj embeddings_model: Embeddings models from 
			nlpatl.models.embeddings
		:param obj clustering_model: Clustering models from 
			nlpatl.models.embeddings
		:param str name: Name of this embeddings

		>>> import nlpatl.models.learning as nml
		>>> model = nml.ClusteringLearning()
    """
    
	def __init__(self, multi_label: bool = False, 
		embeddings_model: Embeddings = None, 
		clustering_model: Clustering = None, 
		name: str = 'clustering_learning'):

		super().__init__(multi_label=multi_label, 
			embeddings_model=embeddings_model,
			clustering_model=clustering_model,
			name=name)

	def keep_most_valuable(self, data: Storage, 
		num_sample: int) -> Storage:
	
		to_be_filter_indices = []
		for group in np.unique(data.groups):
			indices = np.where(data.groups == group)[0]
			values = data.values[indices]

			num_node = min(num_sample, len(indices))

			# get first n shortest distances
			local_indices = values.argpartition(num_node-1)[:num_node]
			to_be_filter_indices.append(indices[local_indices])

		data.keep(np.concatenate(to_be_filter_indices))

		return data
