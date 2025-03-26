import numpy as np
from sklearn.neighbors import KDTree

def PCA(points):
    centroid = np.mean(points, axis=0)
    Q = points - centroid
    cov_matrix = 1/points.shape[0] * Q.T @ Q
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    return eigenvalues, eigenvectors


def compute_normals(query_points, cloud_points, radius, k = 30):
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    all_eigenvectors = np.zeros((cloud_points.shape[0], 3, 3))

    tree = KDTree(cloud_points)
    ind = tree.query_radius(query_points, r=radius)
    #dist, ind = tree.query(query_points, k=k)

    for i in range(len(ind)):
        neighbors = cloud_points[ind[i]]
        _, eigenvectors = PCA(neighbors)
        all_eigenvectors[i,:,:] = eigenvectors

    return all_eigenvectors[:, :, 0]