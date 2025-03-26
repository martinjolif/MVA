#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):

    # YOUR CODE
    decimated_points = points[:len(points):factor]
    decimated_colors = colors[:len(colors):factor]
    decimated_labels = labels[:len(labels):factor]

    return decimated_points, decimated_colors, decimated_labels


def grid_subsampling(points, voxel_size):
    minimum = np.min(points, axis=0)

    voxel_dict = {}
    for point in points:
        voxel_coordinates = tuple(((point - minimum)//voxel_size).astype(int))
        if voxel_coordinates not in voxel_dict:
            voxel_dict[voxel_coordinates] = []
        voxel_dict[voxel_coordinates].append(point)

    subsampled_points = np.zeros((len(voxel_dict), 3))
    for voxel_index, voxel_coordinates in enumerate(voxel_dict):
        barycenter = np.mean(voxel_dict[voxel_coordinates], axis=0)
        subsampled_points[voxel_index, :] = barycenter

    return subsampled_points



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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    
    print('Done')

    # Bonus question: grid subsampling
    if False:
        decimated_points = grid_subsampling(points, 0.2)
        write_ply('../grid_decimated.ply', [decimated_points], ['x', 'y', 'z'])
