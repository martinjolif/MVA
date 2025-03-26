#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np
from numpy.ma.core import indices

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from normals import compute_normals



#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):
    point_plane = points[0]
    normal_plane = np.cross(points[1] - points[0], points[2] - points[0])
    return point_plane, normal_plane/(np.linalg.norm(normal_plane))

def aligned_normals(normals, ref_normal):
    """
    Parameters
    ----------
    normals: array of normals corresponding to the point of the point cloud
    ref_normal: normal which corresponds to the normal of the plane

    Returns
    -------
    angles of the normals with the reference normal of the plane
    """
    thetas = np.arccos(np.clip(np.dot(normals, ref_normal), -1, 1))

    return thetas




def in_plane(points, pt_plane, normal_plane, threshold_in=0.1, normals = None):
    points = points - pt_plane.T
    dot_products = np.abs(np.dot(points, normal_plane))

    if normals is not None:
        thetas = aligned_normals(normals, normal_plane)
        indexes = (dot_products < threshold_in) & (thetas < 0.25)   # 0.25 radian corresponds to 14°
    else:
        indexes = (dot_products < threshold_in)

    return indexes.squeeze()


def RANSAC(points, nb_draws=100, threshold_in=0.1, normals = None):
    best_vote = 3
    best_pt_plane = np.zeros((3,1))
    best_normal_plane = np.zeros((3,1))
    
    for _ in range(nb_draws):
        pt_plane, normal_plane = compute_plane(points[np.random.choice(len(points), 3, replace=False)])
        indexes = in_plane(points, pt_plane, normal_plane, threshold_in, normals = normals)
        if np.sum(indexes) > best_vote:
            best_vote = np.sum(indexes)
            best_pt_plane = pt_plane
            best_normal_plane = normal_plane
                
    return best_pt_plane, best_normal_plane, best_vote


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2):
    nb_points = len(points)
    plane_inds = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_inds = np.arange(0,nb_points)


    for label in range(nb_planes):
        best_pt_plane, best_normal_plane, _ = RANSAC(points[remaining_inds], nb_draws, threshold_in)
        indices_pt_plane = in_plane(points[remaining_inds], best_pt_plane, best_normal_plane, threshold_in)

        plane_inds = np.append(plane_inds, remaining_inds[indices_pt_plane])
        plane_labels = np.append(plane_labels, np.repeat(label, indices_pt_plane.sum()))
        remaining_inds = remaining_inds[~indices_pt_plane]
    
    return plane_inds, remaining_inds, plane_labels


def recursive_normals_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2, normals = None):
    nb_points = len(points)
    plane_inds = np.arange(0, 0)
    plane_labels = np.arange(0, 0)
    remaining_inds = np.arange(0, nb_points)

    radius = 0.1
    normals = compute_normals(points, points, radius)
    for label in range(nb_planes):
        best_pt_plane, best_normal_plane, _ = RANSAC(points[remaining_inds], nb_draws, threshold_in, normals = normals[remaining_inds])
        indices_pt_plane = in_plane(points[remaining_inds], best_pt_plane, best_normal_plane, threshold_in, normals = normals[remaining_inds])

        plane_inds = np.append(plane_inds, remaining_inds[indices_pt_plane])
        plane_labels = np.append(plane_labels, np.repeat(label, indices_pt_plane.sum()))
        remaining_inds = remaining_inds[~indices_pt_plane]

        #free memory
        del indices_pt_plane, best_pt_plane, best_normal_plane

    return plane_inds, remaining_inds, plane_labels



#------------------------------------------------------------------------------------------
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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    nb_points = len(points)
    

    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #
    
    print('\n--- 1) and 2) ---\n')
    
    # Define parameter
    threshold_in = 0.10

    # Take randomly three points
    pts = points[np.random.randint(0, nb_points, size=3)]
    
    # Computes the plane passing through the 3 points
    t0 = time.time()
    pt_plane, normal_plane = compute_plane(pts)
    t1 = time.time()
    print('plane computation done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    t0 = time.time()
    points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]

    # Save extracted plane and remaining points
    write_ply('../plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    #write_ply('../plane.ply', [points[plane_inds]], ['x', 'y', 'z'])
    #write_ply('../remaining_points_plane.ply', [points[remaining_inds]], ['x', 'y', 'z'])

    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #

    print('\n--- 3) ---\n')

    # Define parameters of RANSAC
    nb_draws = 100
    threshold_in = 0.10

    # Find best plane by RANSAC
    t0 = time.time()
    best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
    t1 = time.time()
    print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in)
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]

    # Save the best extracted plane and remaining points
    write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    #write_ply('../best_plane.ply', [points[plane_inds]], ['x', 'y', 'z'])
    #write_ply('../remaining_points_best_plane.ply', [points[remaining_inds]], ['x', 'y', 'z'])

    # Find "all planes" in the cloud
    # ***********************************
    #
    #

    print('\n--- 4) ---\n')
    
    # Define parameters of recursive_RANSAC
    nb_draws = 500
    threshold_in = 0.1
    nb_planes = 5
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_normals_RANSAC(points, nb_draws, threshold_in, nb_planes)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))

    # Save the best planes and remaining points
    write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_best_planes.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    #write_ply('../best_planes.ply', [points[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'plane_label'])
    #write_ply('../remaining_points_best_planes.ply', [points[remaining_inds]], ['x', 'y', 'z'])

    print('Done')
    