import numpy as np
from numpy.linalg import norm as lnorm
import trimesh
import matplotlib.pyplot as plt
from skimage import io

r = 1
target_folder = 'trajectory_projects/wave_1/'

def get_trajectory_from_raster_image(filename='ibs_v5-01.png', do_plotting=True):
    image = io.imread(filename)[:,:,0]
    data0 = np.zeros(shape=(image.shape[0], 2))
    for i in range(image.shape[0]):
        data0[i, 0] = i/image.shape[0]*2*np.pi
        data0[i, 1] = np.argmin(image[i, :])/image.shape[1]*np.pi - np.pi/2
    data0[:, 1] -= data0[0, 1]
    data0 = data0[::5, :]
    print('Decimated to {0} elements'.format(data0.shape[0]))
    if do_plotting:
        print(data0[0, 1])
        print(data0[0, 1] - data0[-1, 1])
        plt.plot(data0[:, 0], data0[:, 1])
        plt.axis('equal')
        plt.show()
    return data0

def rotation_from_point_to_point(point, previous_point):
    vector_to_previous_point = previous_point - point
    axis_of_rotation = [vector_to_previous_point[1], -vector_to_previous_point[0], 0]
    theta = np.linalg.norm(vector_to_previous_point)
    rotation_matrix = trimesh.transformations.rotation_matrix(angle=-1*theta,
                                            direction=axis_of_rotation,
                                            point=[0, 0, 0])
    return rotation_matrix, theta

def rotation_to_previous_point(i, data):
    # make turn w.r.t. an apporopriate axis parallel to xy plane to get to the previous point.
    point = data[i]
    previous_point = data[i - 1]
    return rotation_from_point_to_point(point, previous_point)

def rotation_to_origin(index_in_trajectory, data):
    theta_sum = 0
    if index_in_trajectory == 0:
        combined_matrix = trimesh.transformations.identity_matrix()
    else:
        combined_matrix, theta = rotation_to_previous_point(index_in_trajectory, data)
        theta_sum += theta
        # go through the trajectory backwards and do consecutive rotations
        for i in reversed(list(range(1, index_in_trajectory))):
            rot_matr, theta = rotation_to_previous_point(i, data)
            theta_sum += theta
            combined_matrix = trimesh.transformations.concatenate_matrices(rot_matr,
                                                                           combined_matrix)
    print('Sum_theta = {0}'.format(theta_sum))
    return combined_matrix

# # sweeping parameter space for optimal match of the starting and ending orientation
# N = 30
# M = 30
# angles = np.zeros(shape=(N, M))
# xs = np.zeros_like(angles)
# ys = np.zeros_like(angles)
# for i, kx in enumerate(np.linspace(1.03, 1.05, N)):
#     for j, ky in enumerate(np.linspace(0.53, 0.54, M)):
#         data = np.copy(data0)
#         data[:, 0] = data[:, 0] * kx
#         data[:, 1] = data[:, 1] * ky# +  kx * np.sin(data0[:, 0])
#         rotation_of_entire_traj = trimesh.transformations.rotation_from_matrix(rotation_to_origin(data.shape[0]-1, data))
#         angle = rotation_of_entire_traj[0]
#         xs[i, j] = kx
#         ys[i, j] = ky
#         angles[i, j] = angle
#
# print('Min angle = {0}'.format(np.min(np.abs(angles))))
# f3 = plt.figure(3)
# plt.pcolormesh(xs, ys, np.abs(angles), cmap='viridis', vmin=0, vmax=0.05)
# plt.colorbar()
# plt.show()

# This part of code overlays the experimental trajectory over the theoretical one, for illustration in the paper.

def compute_shape(data0, kx, ky):
    data = np.copy(data0)
    data[:, 0] = data[:, 0] * kx
    data[:, 1] = data[:, 1] * ky  # +  kx * np.sin(data0[:, 0]/2)
    # This code actually computes the positions and orientations of the boxes_for_cutting, and saves each box to a file.
    # These boxes are later loaded to 3dsmax and subtracted from a sphere.
    rotation_of_entire_traj = trimesh.transformations.rotation_from_matrix(rotation_to_origin(data.shape[0] - 1, data))
    print(rotation_of_entire_traj)
    angle = rotation_of_entire_traj[0]
    print('Angle: {0}'.format(angle))

    np.save(target_folder + 'path_data', data)
    cut_size = 10
    base_box = trimesh.creation.box(extents=[cut_size*r,cut_size*r,cut_size*r],
                                    transform=trimesh.transformations.translation_matrix([0,0,-r-1*cut_size*r/2]))
    boxes_for_cutting = []
    for i, point in enumerate(data):
        # make a copy of the base box
        box_for_cutting = base_box.copy()
        # roll the sphere (without slipping) on the xy plane along with the box "glued" to it to the (0,0) point of origin
        box_for_cutting.apply_transform(rotation_to_origin(i, data))
        boxes_for_cutting.append(box_for_cutting.copy())

    for i, box in enumerate(boxes_for_cutting):
        print(i)
        box.export('cut_meshes/test_{0}.obj'.format(i))