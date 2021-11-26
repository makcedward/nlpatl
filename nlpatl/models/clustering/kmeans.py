from typing import List
import numpy as np
from sklearn.cluster import KMeans as SkLearnKMeans

from nlpatl.models.clustering.sklearn import SkLearn


class KMeans(SkLearn):
	def __init__(self, model_name: str = 'kmeans', model_config: dict = {}, 
		num_nearest: int = 2, name: str = 'kmeans'):
		super().__init__(name)

		self.model_config = model_config
		self.model = SkLearnKMeans(**model_config)

		self.num_nearest = num_nearest

	def cluster(self, inputs: List[float]) -> List[int]:
		num_cluster = self.model.n_clusters

		clust_dists = self.model.transform(inputs)
		labels = self.model.predict(inputs)

		results = []
		for cluster in range(self.model.n_clusters):
			data_points = np.where(labels == cluster)[0]

			# number of cluster may less than desired n nearest nodes
			num_node = min(self.num_nearest, len(data_points))

			positions = clust_dists[data_points][:,cluster].argpartition(
				num_node-1)[:num_node]
			indices = data_points[positions]
			results.append(indices)

		return results
