#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree, NearestNeighbors

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):
    # YOUR CODE
    neighborhoods = np.array([])
    for query in queries:
        for point in supports:
            if np.linalg.norm(point - query) < radius:
                neighborhoods = np.append(neighborhoods, point)

    return neighborhoods


def brute_force_KNN(queries, supports, k):
    # YOUR CODE
    neighborhoods = np.array([])
    for query in queries:
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(supports)
        distances, indices = knn.kneighbors([query])
        np.append(neighborhoods, supports[indices])

    return neighborhoods

def radius_kdtree(queries, supports, radius, leaf_size=2):
    tree = KDTree(supports, leaf_size=leaf_size)
    ind = tree.query_radius(queries, r=radius)
    neighborhoods = supports[ind[0]]
    return neighborhoods


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if False:
        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

 

    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:
        # Define the search parameters
        num_queries = 1000
        radius = 0.2

        # YOUR CODE
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        times = []
        leaf_size_max = 100
        leaf_size_list = range(1, leaf_size_max)
        t_min = 10
        for i in range(1, leaf_size_max):
            tree = KDTree(points, leaf_size=i)
            t3 = time.time()
            ind = tree.query_radius(queries, r=radius)
            neighborhoods = points[ind[0]]
            t4 = time.time()
            times.append(t4 - t3)
            if t4 - t3 < t_min:
                optimal_leaf_size = i
                t_min = t4 - t3


        plt.xlabel("leaf size")
        plt.ylabel("time")
        plt.plot(leaf_size_list, times)
        plt.show()

        print('spherical neighborhoods (with kdtree) computed in {:.3f} seconds and with an optimal leaf_size of {:d}'.format(t_min, optimal_leaf_size))

        times = []
        radius_list = np.linspace(0.01, 1.3, 30)
        for radius in radius_list:
            tree = KDTree(points, leaf_size=optimal_leaf_size)
            t5 = time.time()
            ind = tree.query_radius(queries, r=radius)
            neighborhoods = points[ind[0]]
            t6 = time.time()
            times.append(t6 - t5)

        plt.xlabel("radius")
        plt.ylabel("time")
        plt.plot(radius_list, times)
        plt.show()

        tree = KDTree(points, leaf_size=optimal_leaf_size)
        t7 = time.time()
        ind = tree.query_radius(queries, r=0.2)
        neighborhoods = points[ind[0]]
        t8 = time.time()

        total_spherical_time = points.shape[0] * (t8 - t7) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} seconds'.format(total_spherical_time))




        
        
        
        