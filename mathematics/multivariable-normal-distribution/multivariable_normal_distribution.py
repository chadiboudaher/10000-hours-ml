import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

mean = np.array([0, 0])
cov = np.array([[1, 0.5], [0.5, 1]])
point = np.array([1, 1])

euclidean_dist = np.linalg.norm(point - mean)
print(f"Euclidean Distance: {euclidean_dist}")

inv_cov = np.linalg.inv(cov)
mahalanobis_dist = np.sqrt((point - mean).T @ inv_cov @ (point - mean))
print(f"Mahalanobis Distance: {mahalanobis_dist}")

mahalanobis_dist_scipy = distance.mahalanobis(point, mean, np.linalg.inv(cov))
print(f"Mahalanobis Distance (scipy): {mahalanobis_dist_scipy}")