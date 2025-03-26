#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#



def PCA(points):
    centroid = np.mean(points, axis=0)
    Q = points - centroid
    cov_matrix = 1/points.shape[0] * Q.T @ Q
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    return eigenvalues, eigenvectors



def compute_local_PCA(query_points, cloud_points, radius, k = 30):
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    all_eigenvalues = np.zeros((cloud_points.shape[0], 3))
    all_eigenvectors = np.zeros((cloud_points.shape[0], 3, 3))

    tree = KDTree(cloud_points)
    ind = tree.query_radius(query_points, r=radius)
    #dist, ind = tree.query(query_points, k=k)

    for i in range(len(ind)):
        neighbors = cloud_points[ind[i]]
        eigenvalues, eigenvectors = PCA(neighbors)
        all_eigenvalues[i,:] = eigenvalues
        all_eigenvectors[i,:,:] = eigenvectors

    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):
    epsilon = 1e-4

    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)
    normals = all_eigenvectors[:, :, 0]

    verticality = 2*np.arcsin(np.abs(normals[:,2]))/np.pi
    linearity = 1 - all_eigenvalues[:,1]/(all_eigenvalues[:,0] + epsilon)
    planarity = (all_eigenvalues[:,1] - all_eigenvalues[:,2])/(all_eigenvalues[:,0] + epsilon)
    sphericity = all_eigenvalues[:,2]/(all_eigenvalues[:,0] + epsilon)

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

		
    # Normal computation
    # ******************
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, 0.50)
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])


    if True:
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        verticality, linearity, planarity, sphericity = compute_features(cloud, cloud, 0.50)

        # Save cloud
        write_ply('../Lille_street_small_normals_4_features.ply', (cloud, verticality, linearity, planarity, sphericity), ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])
