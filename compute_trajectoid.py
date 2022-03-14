import numpy as np
from numpy.linalg import norm as lnorm
import trimesh
import matplotlib.pyplot as plt
from skimage import io
from mayavi import mlab

def get_trajectory_from_raster_image(filename, do_plotting=True):
    image = io.imread(filename)[:,:,0]
    trajectory_points = np.zeros(shape=(image.shape[0], 2))
    for i in range(image.shape[0]):
        trajectory_points[i, 0] = i/image.shape[0]*2*np.pi #assume that x dimension of path is 2*pi
        trajectory_points[i, 1] = np.argmin(image[i, :])/image.shape[1]*np.pi - np.pi/2
    trajectory_points[:, 1] -= trajectory_points[0, 1] # make path relative to first point
    trajectory_points = trajectory_points[::5, :] # decimation by a factor 5
    print('Decimated to {0} elements'.format(trajectory_points.shape[0]))
    if do_plotting:
        print(trajectory_points[0, 1])
        print(trajectory_points[0, 1] - trajectory_points[-1, 1])
        plt.plot(trajectory_points[:, 0], trajectory_points[:, 1])
        plt.axis('equal')
        plt.show()
    return trajectory_points

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
        net_rotation_matrix = trimesh.transformations.identity_matrix()
    else:
        net_rotation_matrix, theta = rotation_to_previous_point(index_in_trajectory, data)
        theta_sum += theta
        # go through the trajectory backwards and do consecutive rotations
        for i in reversed(list(range(1, index_in_trajectory))):
            matrix_of_rotation_to_previous_point, theta = rotation_to_previous_point(i, data)
            theta_sum += theta
            net_rotation_matrix = trimesh.transformations.concatenate_matrices(matrix_of_rotation_to_previous_point,
                                                                           net_rotation_matrix)
    # print('Sum_theta = {0}'.format(theta_sum))
    return net_rotation_matrix

def plot_mismatch_map_for_scale_tweaking(data0, N=30, M=30, kx_range=(0.1, 2), ky_range=(0.1, 2)):
    # sweeping parameter space for optimal match of the starting and ending orientation
    angles = np.zeros(shape=(N, M))
    xs = np.zeros_like(angles)
    ys = np.zeros_like(angles)
    for i, kx in enumerate(np.linspace(kx_range[0], kx_range[1], N)):
        for j, ky in enumerate(np.linspace(ky_range[0], ky_range[1], M)):
            data = np.copy(data0)
            data[:, 0] = data[:, 0] * kx
            data[:, 1] = data[:, 1] * ky# +  kx * np.sin(data0[:, 0])
            rotation_of_entire_traj = trimesh.transformations.rotation_from_matrix(rotation_to_origin(data.shape[0]-1, data))
            angle = rotation_of_entire_traj[0]
            xs[i, j] = kx
            ys[i, j] = ky
            angles[i, j] = angle

    print('Min angle = {0}'.format(np.min(np.abs(angles))))
    f3 = plt.figure(3)
    plt.pcolormesh(xs, ys, np.abs(angles), cmap='viridis', vmin=0, vmax=0.05)
    plt.colorbar()
    plt.show()

def compute_shape(data0, kx, ky, folder_for_path, folder_for_meshes='cut_meshes', core_radius=1,
                  cut_size = 10):
    data = np.copy(data0)
    data[:, 0] = data[:, 0] * kx
    data[:, 1] = data[:, 1] * ky
    # This code computes the positions and orientations of the boxes_for_cutting, and saves each box to a file.
    # These boxes are later loaded to 3dsmax and subtracted from a sphere
    rotation_of_entire_traj = trimesh.transformations.rotation_from_matrix(rotation_to_origin(data.shape[0] - 1, data))
    print(rotation_of_entire_traj)
    angle = rotation_of_entire_traj[0]
    print('Angle: {0}'.format(angle))

    np.save(folder_for_path + '/path_data', data)
    base_box = trimesh.creation.box(extents=[cut_size * core_radius, cut_size * core_radius, cut_size * core_radius],
                                    transform=trimesh.transformations.translation_matrix([0, 0, -core_radius - 1 * cut_size * core_radius / 2]))
    boxes_for_cutting = []
    for i, point in enumerate(data):
        # make a copy of the base box
        box_for_cutting = base_box.copy()
        # roll the sphere (without slipping) on the xy plane along with the box "glued" to it to the (0,0) point of origin
        # Not optimized here. Would become vastly faster if, for example, you cache the rotation matrices.
        box_for_cutting.apply_transform(rotation_to_origin(i, data))
        boxes_for_cutting.append(box_for_cutting.copy())

    for i, box in enumerate(boxes_for_cutting):
        print('Saving box for cutting: {0}'.format(i))
        box.export('{0}/test_{1}.obj'.format(folder_for_meshes, i))

def trace_on_sphere(data0, kx, ky, core_radius=1, do_plot=False):
    data = np.copy(data0)
    data[:, 0] = data[:, 0] * kx
    data[:, 1] = data[:, 1] * ky  # +  kx * np.sin(data0[:, 0]/2)
    point_at_plane = trimesh.PointCloud([[0, 0, -core_radius]])
    sphere_trace = []
    for i, point in enumerate(data):
        point_at_plane_copy = point_at_plane.copy()
        point_at_plane_copy.apply_transform(rotation_to_origin(i, data))
        sphere_trace.append(np.array(point_at_plane_copy.vertices[0]))
    sphere_trace = np.array(sphere_trace)
    if do_plot:
        # tube_radius=0.05
        tube_radius = 0.01
        # plot a simple sphere
        phi, theta = np.mgrid[0:np.pi:31j, 0:2 * np.pi:31j]
        r = 0.95
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        mlab.mesh(x, y, z, color=(0.7, 0.7, 0.7), opacity=0.736, representation='wireframe')
        # plot the trace
        l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(0, 0, 1),
                        tube_radius=tube_radius)
        mlab.show()
    return sphere_trace

def path_from_trace(sphere_trace, core_radius=1):
    sphere_trace_cloud = trimesh.PointCloud(sphere_trace)
    for i in range(sphere_trace.shape[0]):
        # find the vector of translation
        vector_downward = np.array([0, 0, -core_radius])
        assert np.isclose(vector_downward, sphere_trace[0]).all()
        to_next_point_of_trace = sphere_trace[i+1] - vector_downward
        axis_of_rotation = [to_next_point_of_trace[1], -to_next_point_of_trace[0], 0]
        theta = np.arccos(-sphere_trace[i+1, 2]/core_radius)
        rotation_matrix = trimesh.transformations.rotation_matrix(angle=-1*theta,
                                                direction=axis_of_rotation,
                                                point=[0, 0, 0])
        sphere_trace_cloud.apply_transform(rotation_matrix)
        sphere_trace = np.array(sphere_trace_cloud.vertices)
        print(sphere_trace_cloud)
