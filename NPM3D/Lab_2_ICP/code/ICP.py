#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
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
from visu import show_ICP

import sys
import time


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # YOUR CODE
    barycenter_data = np.mean(data, axis=1).reshape(data.shape[0],1)
    barycenter_ref = np.mean(ref, axis=1).reshape(ref.shape[0],1)
    Q_data = data - barycenter_data
    Q_ref = ref - barycenter_ref

    H = Q_data @ Q_ref.T

    U, S, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        U[:,-1] *= -1
        R = Vh.T @ U.T

    T = barycenter_ref - R @ barycenter_data

    return R, T


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    # YOUR CODE
    rms = np.inf
    tree = KDTree(ref.T)
    last_rot = np.eye(data.shape[0])
    last_trans = np.zeros((data.shape[0], 1))

    for _ in range(max_iter):
        if rms < RMS_threshold:
            break
        else:
            dist, indices = tree.query(data_aligned.T, k=1)
            ref_neighbors = ref[:, indices].reshape(data.shape)
            neighbors_list.append(indices.squeeze())
            rot, trans = best_rigid_transform(data_aligned, ref_neighbors)
            data_aligned = rot @ data_aligned + trans
            trans = rot @ last_trans + trans
            rot = rot @ last_rot
            R_list.append(rot)
            T_list.append(trans)
            last_rot = rot
            last_trans = trans
            rms = np.sqrt(np.mean(np.linalg.norm(data_aligned - ref_neighbors, axis=0)))
            RMS_list.append(rms)

    return data_aligned, R_list, T_list, neighbors_list, RMS_list

def icp_point_to_point_fast(data, ref, max_iter, RMS_threshold, sampling_limit):
    '''
       Iterative closest point algorithm with a point to point strategy.
       Inputs :
           data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
           ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
           max_iter = stop condition on the number of iterations
           RMS_threshold = stop condition on the distance
           sampling_limit: control the number of point used at each iteration
       Returns :
           data_aligned = data aligned on reference cloud
           R_list = list of the (d x d) rotation matrices found at each iteration
           T_list = list of the (d x 1) translation vectors found at each iteration
           neighbors_list = At each iteration, you search the nearest neighbors of each data point in
           the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
           arrays at each iteration
       '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    # YOUR CODE
    rms = np.inf
    tree = KDTree(ref.T)
    last_rot = np.eye(data.shape[0])
    last_trans = np.zeros((data.shape[0], 1))


    for _ in range(max_iter):
        if rms < RMS_threshold:
            break
        else:
            sampling_indices = np.random.randint(0, data.shape[1], size=sampling_limit)
            dist, indices = tree.query(data_aligned[:, sampling_indices].T, k=1)
            ref_neighbors = ref[:, indices].reshape(data_aligned[:, sampling_indices].shape)
            neighbors_list.append(indices.squeeze())
            rot, trans = best_rigid_transform(data_aligned[:, sampling_indices], ref_neighbors)
            data_aligned = rot @ data_aligned + trans
            trans = rot @ last_trans + trans
            rot = rot @ last_rot
            R_list.append(rot)
            T_list.append(trans)
            last_rot = rot
            last_trans = trans

            dist, indices = tree.query(data_aligned.T, k=1)
            full_ref_neighbors = ref[:, indices].reshape(data.shape)
            rms = np.sqrt(np.mean(np.linalg.norm(data_aligned - full_ref_neighbors, axis=0)))
            RMS_list.append(rms)

    return data_aligned, R_list, T_list, neighbors_list, RMS_list


#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

		# Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        write_ply('../bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.8f}'.format(RMS_before))
        print(' After = {:.8f}'.format(RMS_after))
   

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'
        
        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))        

        # Apply ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 10, 1e-4)

        # Show ICP
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)

        # Plot RMS
        plt.plot(RMS_list)
        plt.show()
        

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'
        
        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        
        # Show ICP
        show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()

    # Bonus question
    if True:
        # Cloud paths
        bunny_o_path = '../data/Notre_Dame_Des_Champs_1.ply'
        bunny_p_path = '../data/Notre_Dame_Des_Champs_2.ply'

        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        t1 = time.time()
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point_fast(bunny_p, bunny_o, 1000, 1e-4, 1000)
        t2 = time.time()
        print("Time to compute ICP in seconds:", t2 - t1)

        # Show ICP
        show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)

        # Plot RMS
        plt.plot(RMS_list)
        plt.show()
