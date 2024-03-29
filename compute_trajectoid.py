import numpy as np
from numpy.linalg import norm as lnorm
import trimesh
import time
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from skimage import io
from math import atan2
from scipy.optimize import fsolve, brentq, minimize
from scipy import interpolate
from sklearn.metrics import pairwise_distances
from numba import jit
from scipy.signal import savgol_filter
from functools import lru_cache
from tqdm import tqdm
import logging
import sys

sys.setrecursionlimit(3000)

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

# from great_circle_arc import intersects\

USED_3D_PLOTTING_PACKAGE = 'mayavi'

# USED_3D_PLOTTING_PACKAGE = 'plotly'

# if USED_3D_PLOTTING_PACKAGE == 'mayavi':
#     from mayavi import mlab
# elif USED_3D_PLOTTING_PACKAGE == 'plotly':
#     import plotly.express as px

from mayavi import mlab
import plotly.graph_objects as go

last_path = np.array([0, 0])
cached_rotations_to_origin = dict()


@jit(nopython=True)
def numbacross(a, b):
    return [a[1] * b[2] - b[1] * a[2],
            -a[0] * b[2] + b[0] * a[2],
            a[0] * b[1] - b[0] * a[1]]


@jit(nopython=True)
def numbadotsign(a, b):
    x = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    if x > 0:
        r = 1
    elif x < 0:
        r = -1
    else:
        r = 0
    return r


@jit(nopython=True)
def intersects(A, B, C, D):
    ABX = numbacross(A, B)
    CDX = numbacross(C, D)
    T = numbacross(ABX, CDX)
    s = 0
    s += numbadotsign(numbacross(ABX, A), T)
    s += numbadotsign(numbacross(B, ABX), T)
    s += numbadotsign(numbacross(CDX, C), T)
    s += numbadotsign(numbacross(D, CDX), T)
    return (s == 4) or (s == -4)


def sort_path(arr2D):
    columnIndex = 0
    return arr2D[arr2D[:, columnIndex].argsort()]


def split_by_mask(signal, input_mask):
    mask = np.concatenate(([False], input_mask, [False]))
    idx = np.flatnonzero(mask[1:] != mask[:-1])
    return [signal[idx[i]:idx[i + 1]] for i in range(0, len(idx), 2)]


def better_mayavi_lights(fig):
    azims = [60, -60, 60, -60]
    elevs = [-30, -30, 30, 30]
    fig.scene._lift()
    for i, camera_light0 in enumerate(fig.scene.light_manager.lights):
        camera_light0.elevation = elevs[i]
        camera_light0.azimuth = azims[i]
        camera_light0.intensity = 0.5
        camera_light0.activate = True


def make_orbit_animation(folder_for_frames, nframes=60, elevation=60):
    mlab.view(elevation=elevation)
    for frame_id, azimuth in enumerate(np.linspace(0, 359, nframes)):
        mlab.view(azimuth=azimuth)
        # camera_radius = 4
        # mfig.scene.camera.position = [camera_radius*np.cos(azimuth), camera_radius*np.sin(azimuth), -2.30]
        # print(mfig.actors)
        # mfig.actors.actor.rotate_y(5)
        mlab.savefig(f'{folder_for_frames}/{frame_id:08d}.png')


def signed_angle_between_2d_vectors(vector1, vector2):
    """Calculate the signed angle between two 2-dimensional vectors using the atan2 formula.
    The angle is positive if rotation from vector1 to vector2 is counterclockwise, and negative
    of the rotation is clockwise. Angle is in radians.

    This is more numerically stable for angles close to 0 or pi than the acos() formula.
    """
    # make sure that vectors are 2d
    assert vector1.shape == (2,)
    assert vector2.shape == (2,)
    # Convert to 3D for making cross product
    vector1_ = np.append(vector1, 0) / np.linalg.norm(vector1)
    vector2_ = np.append(vector2, 0) / np.linalg.norm(vector2)
    return atan2(np.cross(vector1_, vector2_)[-1], np.dot(vector1_, vector2_))


def unsigned_angle_between_vectors(vector1, vector2):
    """Calculate the unsigned angle between two n-dimensional vectors using the atan2 formula.
    Angle is in radians.

    This is more numerically stable for angles close to 0 or pi than the acos() formula.
    """
    return atan2(np.linalg.norm(np.cross(vector1, vector2)), np.dot(vector1, vector2))


def rotate_2d(vector, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = (0, 0)
    px, py = vector

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return np.array([qx, qy])


def rotate_3d_vector(input_vector, axis_of_rotation, angle):
    point1_trimesh = trimesh.PointCloud([input_vector])
    rotation_matrix = trimesh.transformations.rotation_matrix(angle=angle,
                                                              direction=axis_of_rotation,
                                                              point=[0, 0, 0])
    point1_trimesh.apply_transform(rotation_matrix)
    return np.array(point1_trimesh.vertices[0])


def spherical_trace_is_self_intersecting(sphere_trace):
    # t0 = time.time()
    arcs = [[sphere_trace[i], sphere_trace[i + 1]] for i in range(sphere_trace.shape[0] - 1)]
    intersection_detected = False
    for i in range(len(arcs)):
        for j in reversed(range(len(arcs))):
            if (j <= (i + 1)) or ((i == 0) and (j == (len(arcs) - 1))):
                continue
            else:
                if intersects(arcs[i][0], arcs[i][1], arcs[j][0], arcs[j][1]):
                    intersection_detected = True
                    print(f'self-intersection at i={i}, j={j}')
                    break
        if intersection_detected:
            break
    # print(f"Computed intesections in {(time.time() - t0)} seconds")
    return intersection_detected


def get_trajectory_from_raster_image(filename, do_plotting=True):
    image = io.imread(filename)[:, :, 0]
    trajectory_points = np.zeros(shape=(image.shape[0], 2))
    for i in range(image.shape[0]):
        trajectory_points[i, 0] = i / image.shape[0] * 2 * np.pi  # assume that x dimension of path is 2*pi
        trajectory_points[i, 1] = np.argmin(image[i, :]) / image.shape[1] * np.pi - np.pi / 2
    trajectory_points[:, 1] -= trajectory_points[0, 1]  # make path relative to first point
    trajectory_points = trajectory_points[::5, :]  # decimation by a factor 5
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
    rotation_matrix = trimesh.transformations.rotation_matrix(angle=-1 * theta,
                                                              direction=axis_of_rotation,
                                                              point=[0, 0, 0])
    return rotation_matrix, theta


def rotation_to_previous_point(i, data):
    # make turn w.r.t. an apporopriate axis parallel to xy plane to get to the previous point.
    point = data[i]
    previous_point = data[i - 1]
    return rotation_from_point_to_point(point, previous_point)


def rotation_to_origin(index_in_trajectory, data, use_cache=True, recursive=True):
    if use_cache:
        global last_path
        global cached_rotations_to_origin
        if data.shape == last_path.shape:
            if np.isclose(data, last_path).all():
                if index_in_trajectory in cached_rotations_to_origin.keys():
                    return cached_rotations_to_origin[index_in_trajectory]

    if not recursive:
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
    elif recursive:
        theta_sum = 0
        if index_in_trajectory == 0:
            net_rotation_matrix = trimesh.transformations.identity_matrix()
        else:
            net_rotation_matrix, theta = rotation_to_previous_point(index_in_trajectory, data)
            net_rotation_matrix = trimesh.transformations.concatenate_matrices(
                rotation_to_origin(index_in_trajectory - 1,
                                   data, use_cache, recursive),
                net_rotation_matrix)
    # add to cache
    if use_cache:
        cache_have_same_path = False
        if data.shape == last_path.shape:
            if np.isclose(data, last_path).all():
                cache_have_same_path = True
                cached_rotations_to_origin[index_in_trajectory] = net_rotation_matrix
                logging.debug(f'Updated cache, index_in_trajectory = {index_in_trajectory}')
        if not cache_have_same_path:
            # clear cache
            logging.debug('Clearing cache.')
            cached_rotations_to_origin = dict()
            cached_rotations_to_origin[index_in_trajectory] = net_rotation_matrix
            last_path = np.copy(data)

    return net_rotation_matrix


def plot_mismatch_map_for_scale_tweaking(data0, N=30, M=30, kx_range=(0.1, 2), ky_range=(0.1, 2), vmin=0, vmax=np.pi,
                                         signed_angle=False):
    # sweeping parameter space for optimal match of the starting and ending orientation
    angles = np.zeros(shape=(N, M))
    xs = np.zeros_like(angles)
    ys = np.zeros_like(angles)
    for i, kx in enumerate(np.linspace(kx_range[0], kx_range[1], N)):
        for j, ky in enumerate(np.linspace(ky_range[0], ky_range[1], M)):
            data = np.copy(data0)
            data[:, 0] = data[:, 0] * kx
            data[:, 1] = data[:, 1] * ky  # +  kx * np.sin(data0[:, 0])
            rotation_of_entire_traj = trimesh.transformations.rotation_from_matrix(
                rotation_to_origin(data.shape[0] - 1, data))
            angle = rotation_of_entire_traj[0]
            xs[i, j] = kx
            ys[i, j] = ky
            angles[i, j] = angle

    print('Min angle = {0}'.format(np.min(np.abs(angles))))
    f3 = plt.figure(3)
    if signed_angle:
        plt.pcolormesh(xs, ys, angles, cmap='viridis', vmin=-vmax, vmax=vmax)
    else:
        plt.pcolormesh(xs, ys, np.abs(angles), cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()


def compute_shape(data0, kx, ky, folder_for_path, folder_for_meshes='cut_meshes', core_radius=1,
                  cut_size=10):
    data = np.copy(data0)
    data[:, 0] = data[:, 0] * kx
    data[:, 1] = data[:, 1] * ky
    # This code computes the positions and orientations of the boxes_for_cutting, and saves each box to a file.
    # These boxes are later loaded to 3dsmax and subtracted from a sphere
    rotation_of_entire_traj = trimesh.transformations.rotation_from_matrix(rotation_to_origin(data.shape[0] - 1, data))
    # print(rotation_of_entire_traj)
    angle = rotation_of_entire_traj[0]
    # print('Angle: {0}'.format(angle))

    np.save(folder_for_path + '/path_data', data)
    base_box = trimesh.creation.box(extents=[cut_size * core_radius, cut_size * core_radius, cut_size * core_radius],
                                    transform=trimesh.transformations.translation_matrix(
                                        [0, 0, -core_radius - 1 * cut_size * core_radius / 2]))
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


def plot_sphere(r0, line_radius, sphere_opacity=.8):
    sphere = mlab.points3d(0, 0, 0, scale_mode='none',
                           scale_factor=2 * r0,
                           color=(1, 1, 1),
                           resolution=100,
                           opacity=sphere_opacity,
                           name='Earth')
    sphere.actor.property.frontface_culling = True

    phi = np.linspace(0, 2 * np.pi, 50)
    r = r0 + line_radius
    for theta in np.linspace(0, np.pi, 7)[:-1]:
        mlab.plot3d(
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi), tube_radius=line_radius)

    theta = np.linspace(0, 2 * np.pi, 50)
    for phi in np.linspace(0, np.pi, 7)[:-1]:
        mlab.plot3d(
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi) * np.ones_like(theta), tube_radius=line_radius)


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
        mlab.figure(size=(1024, 768), \
                    bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
        # tube_radius=0.05
        tube_radius = 0.01

        plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
        # # plot a simple sphere
        # phi, theta = np.mgrid[0:np.pi:31j, 0:2 * np.pi:31j]
        # r = 0.95
        # x = r * np.sin(phi) * np.cos(theta)
        # y = r * np.sin(phi) * np.sin(theta)
        # z = r * np.cos(phi)
        # mlab.mesh(x, y, z, color=(0.7, 0.7, 0.7), opacity=0.736, representation='surface') #representation='wireframe'
        # plot the trace
        l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(0, 0, 1),
                        tube_radius=tube_radius)
        mlab.show()
    return sphere_trace


def trace_on_sphere_nonocontact_point(data0, kx, ky, core_radius=1, do_plot=False, startpoint=[0, 0, -1]):
    data = np.copy(data0)
    data[:, 0] = data[:, 0] * kx
    data[:, 1] = data[:, 1] * ky  # +  kx * np.sin(data0[:, 0]/2)
    point_at_plane = trimesh.PointCloud([startpoint])
    sphere_trace = []
    for i, point in enumerate(data):
        point_at_plane_copy = point_at_plane.copy()
        point_at_plane_copy.apply_transform(rotation_to_origin(i, data))
        sphere_trace.append(np.array(point_at_plane_copy.vertices[0]))
    sphere_trace = np.array(sphere_trace)
    if do_plot:
        mlab.figure(size=(1024, 768), \
                    bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
        # tube_radius=0.05
        tube_radius = 0.01

        plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
        # # plot a simple sphere
        # phi, theta = np.mgrid[0:np.pi:31j, 0:2 * np.pi:31j]
        # r = 0.95
        # x = r * np.sin(phi) * np.cos(theta)
        # y = r * np.sin(phi) * np.sin(theta)
        # z = r * np.cos(phi)
        # mlab.mesh(x, y, z, color=(0.7, 0.7, 0.7), opacity=0.736, representation='surface') #representation='wireframe'
        # plot the trace
        l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(0, 0, 1),
                        tube_radius=tube_radius)
        mlab.show()
    return sphere_trace


def path_from_trace(sphere_trace, core_radius=1):
    sphere_trace_cloud = trimesh.PointCloud(sphere_trace)
    translation_vectors = []
    position_vectors = [np.array([0, 0])]
    vector_downward = np.array([0, 0, -core_radius])
    for i in range(sphere_trace.shape[0] - 1):
        # make sure that current (i-th) point is the contact point and therefore coincides with the
        #   downward vector.
        assert np.isclose(vector_downward, sphere_trace[i]).all()
        to_next_point_of_trace = sphere_trace[i + 1] - vector_downward

        # find the vector of translation
        theta = np.arccos(-sphere_trace[i + 1, 2] / core_radius)
        arc_length = theta * core_radius
        # Here the vector to_next_point_of_trace[:-1] is the xy-projection of to_next_point_of_trace vector
        #   We normalize it and then multiply by arc length to get translation vector.
        translation_vector = arc_length * to_next_point_of_trace[:-1] / np.linalg.norm(to_next_point_of_trace[:-1])
        translation_vectors.append(np.copy(translation_vector))
        position_vectors.append(position_vectors[-1] + translation_vector)

        # Rotate the cloud of points containing the trace. This correspods to a roll of the sphere
        #   from this point of the trace to the next point of the trace.
        # After this roll, the next point of the tract will be the contact point (its vector will be
        # directly downlward, equal to [0, 0, -core_radius]
        # Axis of rotation lies in the xy plane and is perpendicular to the vector to_next_point_of_trace
        axis_of_rotation = [to_next_point_of_trace[1], -to_next_point_of_trace[0], 0]
        rotation_matrix = trimesh.transformations.rotation_matrix(angle=-1 * theta,
                                                                  direction=axis_of_rotation,
                                                                  point=[0, 0, 0])
        sphere_trace_cloud.apply_transform(rotation_matrix)
        sphere_trace = np.array(sphere_trace_cloud.vertices)

    translation_vectors = np.array(translation_vectors)
    position_vectors = np.array(position_vectors)
    return position_vectors


def plot_three_path_periods(input_path, savetofile=False, plot_midpoints=False):
    figtraj = plt.figure(10, figsize=(10, 5))
    dataxlen = np.max(input_path[:, 0])

    def plot_periods(data, linestyle, linewidth):
        plt.plot(data[:, 0], data[:, 1], color='black', alpha=1, linestyle=linestyle, linewidth=linewidth)
        plt.plot(dataxlen + data[:, 0], data[:, 1], color='black', alpha=1, linestyle=linestyle, linewidth=linewidth)
        plt.plot(2 * dataxlen + data[:, 0], data[:, 1], color='black', alpha=1, linestyle=linestyle,
                 linewidth=linewidth)
        plt.plot(3 * dataxlen + data[:, 0], data[:, 1], color='black', alpha=1, linestyle=linestyle,
                 linewidth=linewidth)

    # plot_periods(data, '--', linewidth=0.5)
    plot_periods(input_path, '-', linewidth=1)
    # plot_periods(projection_centers, '-', linewidth=1)

    for shift in dataxlen * np.arange(3):
        plt.scatter(shift + input_path[-1, 0], input_path[-1, 1], s=35, color='black')
    # plt.scatter(dataxlen + input_path[-1, 0], input_path[-1, 1], s=35, color='black')
    if plot_midpoints:
        midpoint_index = int(round(input_path.shape[0] / 2))
        for shift in dataxlen * np.arange(4):
            plt.scatter(shift + input_path[midpoint_index, 0], input_path[midpoint_index, 1], s=35, facecolors='white',
                        edgecolors='black')
    plt.axis('equal')
    if savetofile:
        figtraj.savefig(f'{savetofile}.png', dpi=300)
        figtraj.savefig(f'{savetofile}.eps')
    plt.show()


def bridge_two_points_by_arc(point1, point2, npoints=10):
    '''points are 3d vectors from center of unit sphere to sphere surface'''
    # make sure that the lengths of input vectors are equal to unity
    for point in [point1, point2]:
        assert np.isclose(np.linalg.norm(point), 1)
    # Find the single turn that brings point 1 into point 2 and split this turn into many small turns.
    # Each small turn gives a new intermediate point. Total number of small turns is equal to npoints.
    axis_of_rotation = np.cross(point1, point2)
    sum_theta = np.arccos(np.dot(point1, point2))
    thetas = np.linspace(0, sum_theta, npoints)
    points_of_bridge = []
    for theta in thetas:
        point1_trimesh = trimesh.PointCloud([point1])
        rotation_matrix = trimesh.transformations.rotation_matrix(angle=theta,
                                                                  direction=axis_of_rotation,
                                                                  point=[0, 0, 0])
        point1_trimesh.apply_transform(rotation_matrix)
        point_here = np.array(point1_trimesh.vertices[0])
        points_of_bridge.append(point_here)
    return np.array(points_of_bridge)


def filter_backward_declination(declination_angle, input_path, maximum_angle_from_vertical=np.pi / 180 * 80):
    if declination_angle > np.pi:
        declination_angle = declination_angle - 2 * np.pi
    tangent_backward = input_path[0] - input_path[1]
    angle0 = signed_angle_between_2d_vectors(np.array([-1, 0]), tangent_backward)

    candidate_direction_backward = rotate_2d(tangent_backward, declination_angle)
    a = signed_angle_between_2d_vectors(tangent_backward, candidate_direction_backward)
    assert np.isclose(a, declination_angle)

    # if angle exceeds the highest allowed angle with vertical direction, use the highest allowed angle instead
    angle_from_vertical = signed_angle_between_2d_vectors(np.array([-1, 0]), candidate_direction_backward)
    if np.abs(angle_from_vertical) >= maximum_angle_from_vertical:
        declination_angle = maximum_angle_from_vertical * np.sign(angle_from_vertical) - angle0

    # this is to prevent exactly zero declination angles, lest it would cause degenerate backward small arc
    if declination_angle == 0:
        declination_angle = declination_angle + 1e-4
    return declination_angle


def filter_forward_declination(declination_angle, input_path, maximum_angle_from_vertical=np.pi / 180 * 80):
    if declination_angle > np.pi:
        declination_angle = declination_angle - 2 * np.pi
    tangent_forward = input_path[-1] - input_path[-2]
    angle0 = signed_angle_between_2d_vectors(np.array([1, 0]), tangent_forward)

    candidate_direction_forward = rotate_2d(tangent_forward, declination_angle)
    a = signed_angle_between_2d_vectors(tangent_forward, candidate_direction_forward)
    assert np.isclose(a, declination_angle)

    # if angle exceeds the highest allowed angle with vertical direction, use the highest allowed angle instead
    angle_from_vertical = signed_angle_between_2d_vectors(np.array([1, 0]), candidate_direction_forward)
    if np.abs(angle_from_vertical) >= maximum_angle_from_vertical:
        declination_angle = maximum_angle_from_vertical * np.sign(angle_from_vertical) - angle0

    # this is to prevent exactly zero declination angles: lest this would cause degenerate forward small arc
    if declination_angle == 0:
        declination_angle = declination_angle + 1e-4
    return declination_angle


def make_corner_bridge_candidate(input_declination_angle, input_path, npoints, do_plot=True):
    # Overall plan:
    # 1. forward declined section
    #       1.1. Filter forward declination angle -- make sure it's not too deflected from the downward direction
    #           (gravity projection)
    #       1.2. Make an forward arc
    # 2. Backward declined section
    #       2.1. Filter backward declination angle
    #       2.2. Check that backward arc and forward arc are in the same hemosphere. If not, multiply the backward angle
    #            by (-1) and use the new backward angle instead.
    # 3. Find the intersection of backward and forward arc. There are two intersection. You need the one that is in the
    #    same hemisphere as the infinitely small starting sections of that arc

    sphere_trace = trace_on_sphere(input_path, kx=1, ky=1)
    # forward arc
    axis_at_last_point = np.cross(sphere_trace[-2], sphere_trace[-1])
    forward_declination_angle = filter_forward_declination(input_declination_angle, input_path)
    forward_arc_axis = rotate_3d_vector(axis_at_last_point, sphere_trace[-1], forward_declination_angle)
    small_angle = np.pi / 18
    small_forward_arc = rotate_3d_vector(sphere_trace[-1], forward_arc_axis, small_angle)

    # if do_plot:
    #     core_radius = 1
    #     mlab.figure(size=(1024, 768), \
    #                 bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    #     tube_radius = 0.01
    #     plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
    #     l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(0, 0, 1),
    #                     tube_radius=tube_radius)
    # for point_here in [small_forward_arc]:
    #     mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.1, color=(0, 1, 0))

    def get_backward_arc_axis(input_declination_angle, input_path, sphere_trace):
        axis_at_first_point = np.cross(sphere_trace[0], sphere_trace[1])
        backward_declination_angle = filter_backward_declination(input_declination_angle, input_path)
        # print('Backward_an')
        backward_arc_axis = rotate_3d_vector(axis_at_first_point, sphere_trace[0], backward_declination_angle)
        return backward_arc_axis

    backward_arc_axis = get_backward_arc_axis(input_declination_angle, input_path, sphere_trace)
    small_backward_arc = rotate_3d_vector(sphere_trace[0], backward_arc_axis, -1 * small_angle)
    # check whether the forward and backward small arcs are on the same hemisphere with respect to the plane
    # passing through the sphere center and the line connecting the last point and first point
    reference_plane_normal = np.cross(sphere_trace[-1], sphere_trace[0])
    forward_sign = np.dot(small_forward_arc - sphere_trace[-1], reference_plane_normal)
    backward_sign = np.dot(small_backward_arc - sphere_trace[0], reference_plane_normal)
    # if they are not on the same side, then use opposite declination for backward arc
    if forward_sign * backward_sign < 0:
        backward_arc_axis = get_backward_arc_axis(-1 * input_declination_angle, input_path, sphere_trace)
        small_backward_arc = rotate_3d_vector(sphere_trace[0], backward_arc_axis, -1 * small_angle)

    # if do_plot:
    #     for point_here in [small_backward_arc]:
    #         mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.1, color=(1, 0, 0))

    # find intersection between the forward arc and the backward arc; intersection of interest lies in the same
    # hemisphere as the small arcs
    intersection = np.cross(forward_arc_axis, backward_arc_axis)
    intersection = intersection / np.linalg.norm(intersection)
    if np.dot(intersection, reference_plane_normal) * forward_sign < 0:
        intersection = -1 * intersection

    # if do_plot:
    #     for point_here in [intersection]:
    #         mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.1, color=(1, 1, 0))

    forward_full_arc = bridge_two_points_by_arc(sphere_trace[-1], intersection, npoints=npoints)
    backward_full_arc = bridge_two_points_by_arc(intersection, sphere_trace[0], npoints=npoints)
    if do_plot:
        tube_radius = 0.01
        for curve in [forward_full_arc, backward_full_arc]:
            mlab.plot3d(curve[:, 0], curve[:, 1], curve[:, 2], color=(0, 1, 0),
                        tube_radius=tube_radius)

    full_bridge = np.concatenate((forward_full_arc[1:], backward_full_arc[1:]), axis=0)
    trace_width_bridge = np.concatenate((sphere_trace, full_bridge), axis=0)
    input_path_with_bridge = path_from_trace(trace_width_bridge)
    return input_path_with_bridge


def mismatch_angle_for_path(input_path, recursive=False, use_cache=False):
    rotation_of_entire_traj = trimesh.transformations.rotation_from_matrix(
        rotation_to_origin(input_path.shape[0] - 1, input_path, recursive=recursive, use_cache=use_cache))
    angle = rotation_of_entire_traj[0]
    return angle


def mismatch_angle_for_bridge(declination_angle, input_path, npoints=30):
    path_with_bridge = make_corner_bridge_candidate(declination_angle, input_path, npoints=npoints, do_plot=False)
    angle = mismatch_angle_for_path(path_with_bridge)
    return angle


def find_best_bridge(input_path, npoints=30, do_plot=True):
    declination_angles = np.linspace(-np.pi * 0.75, np.pi * 0.75, 13)
    mismatches = []
    for i, declination_angle in enumerate(declination_angles):
        mismatches.append(mismatch_angle_for_bridge(declination_angle, input_path, npoints=30))
        print(f'Preliminary screening, step {i} completed')
    initial_guess = declination_angles[np.argmin(np.abs(np.array(mismatches)))]
    print(f'Initial guess: {initial_guess}')
    if do_plot:
        plt.plot(declination_angles, mismatches, 'o-')
        plt.show()

    def left_hand_side(x):  # the function whose root we want to find
        return np.array([mismatch_angle_for_bridge(s, input_path, npoints=30) for s in x])

    best_declination = fsolve(left_hand_side, initial_guess, maxfev=20)
    print(f'Best declination: {best_declination}')
    print(f'Best mismatch: {left_hand_side(best_declination)}')
    return best_declination[0]


def mismatch_angle_for_smooth_bridge(declination_angle, input_path, npoints=30, return_error_messages=True,
                                     min_curvature_radius=0.2):
    path_with_bridge, is_successful = make_smooth_bridge_candidate(declination_angle, input_path, npoints=npoints,
                                                                   do_plot=False,
                                                                   min_curvature_radius=min_curvature_radius)
    angle = mismatch_angle_for_path(path_with_bridge)
    if return_error_messages:
        return angle, is_successful
    else:
        return angle


def find_best_smooth_bridge(input_path, npoints=30, do_plot=True, max_declination=np.pi / 180 * 80,
                            min_curvature_radius=0.2):
    declination_angles = np.linspace(-max_declination, max_declination, 20)
    mismatches = []
    for i, declination_angle in enumerate(declination_angles):
        mismatches.append(mismatch_angle_for_smooth_bridge(declination_angle, input_path, npoints=npoints,
                                                           min_curvature_radius=min_curvature_radius))
        logging.debug(f'Preliminary screening, step {i} completed')
    mismatches = np.array(mismatches)
    # use split by mask here and find roots in each subsection
    mask_here = mismatches[:, 1]
    declination_fragments = split_by_mask(declination_angles, mask_here)
    mismatches_fragments = split_by_mask(mismatches[:, 0], mask_here)
    found_sign_change = False
    for i, mismatches_fragment in enumerate(mismatches_fragments):
        if np.max(mismatches_fragment) >= 0 and np.min(mismatches_fragment) <= 0:
            minangle = np.min(declination_fragments[i])
            maxangle = np.max(declination_fragments[i])
            linear_interpolator_function = interpolate.interp1d(declination_fragments[i], mismatches_fragment)
            found_sign_change = True
            break
    if not found_sign_change:  # this means failure of the entire endeavour
        return False
    else:
        initial_guess = brentq(linear_interpolator_function, a=minangle, b=maxangle)
        position = np.argmax(declination_angles > initial_guess)
        maxangle = declination_angles[position]
        minangle = declination_angles[position - 1]
        logging.debug(f'Sign-changing interval: from {minangle} to {maxangle}')
        # initial_guess = declination_angles[np.argmin(np.abs(np.array(mismatches)))]
        logging.debug(f'Initial guess: {initial_guess}')
        if do_plot:
            # mlab.show()
            plt.plot(declination_angles, mismatches, 'o-')
            plt.show()

        def left_hand_side(x):  # the function whose root we want to find
            logging.debug(f'Sampling function at x={x}')
            return mismatch_angle_for_smooth_bridge(x, input_path, npoints=npoints, return_error_messages=False,
                                                    min_curvature_radius=min_curvature_radius)

        best_declination = brentq(left_hand_side, a=minangle, b=maxangle, maxiter=20, xtol=0.001, rtol=0.004)
        logging.debug(f'Best declination: {best_declination}')
        logging.debug(f'Best mismatch: {left_hand_side(best_declination)}')
        return best_declination


def make_smooth_bridge_candidate(input_declination_angle, input_path, npoints, min_curvature_radius=0.2,
                                 do_plot=True, mlab_show=False, make_animation=False,
                                 default_forward_angle='downward',
                                 default_backward_angle='downward'):
    # forward smooth deflection section is a semicircle made by slowly rotating tangent with a given curvature radius
    #   until the total accumulated angle (in plane) is equal to the input deflection angle
    # Backward deflection section is created similarly.
    # Deflection sections are extended by geodesic sections. These geodesics are then joined by smooth semicircle,
    #   instead o the direct intesections (as it is in the "corner bridge" implementation).
    sphere_trace = trace_on_sphere(input_path, kx=1, ky=1)
    # forward arc
    axis_at_last_point = np.cross(sphere_trace[-2], sphere_trace[-1])
    # find angle between the direction at last point and the direction to downward
    if default_forward_angle == 'downward':
        tangent_forward = input_path[-1] - input_path[-2]
        default_forward_angle = signed_angle_between_2d_vectors(np.array([1, 0]), tangent_forward)
    if default_backward_angle == 'downward':
        tangent_backward = input_path[0] - input_path[1]
        default_backward_angle = signed_angle_between_2d_vectors(np.array([-1, 0]), tangent_backward)

    # implemennt option where default direction is connecting the end and beginning point
    if default_forward_angle == 'directbridge':
        axis_of_direct_bridge = np.cross(sphere_trace[-1], sphere_trace[0])
        sign_here = np.sign(np.dot(sphere_trace[-1],
                                   np.cross(axis_at_last_point, axis_of_direct_bridge)))
        default_forward_angle = -1 * sign_here * unsigned_angle_between_vectors(axis_at_last_point,
                                                                                axis_of_direct_bridge)

    # implemennt option where default direction is connecting the end and beginning point
    if default_backward_angle == 'directbridge':
        axis_at_first_point = np.cross(sphere_trace[1], sphere_trace[0])
        axis_of_direct_bridge = np.cross(sphere_trace[0], sphere_trace[-1])
        sign_here = np.sign(np.dot(sphere_trace[0],
                                   np.cross(axis_at_first_point, axis_of_direct_bridge)))
        default_backward_angle = sign_here * unsigned_angle_between_vectors(axis_at_first_point, axis_of_direct_bridge)

    forward_declination_angle = filter_forward_declination(input_declination_angle - default_forward_angle, input_path)
    print(
        f'Forward angle:  raw={input_declination_angle}, plusdef={input_declination_angle - default_forward_angle}, filtered={forward_declination_angle}')
    turn_angle_increment = forward_declination_angle / npoints
    geodesic_length_of_single_step = np.abs(min_curvature_radius * forward_declination_angle / npoints)
    point_here = np.copy(sphere_trace[-1])
    forward_arc_points = [point_here]
    for i in range(npoints):
        next_axis = rotate_3d_vector(axis_at_last_point, point_here, -turn_angle_increment)
        next_point = rotate_3d_vector(point_here, next_axis, geodesic_length_of_single_step)
        forward_arc_points.append(next_point)
        axis_at_last_point = np.copy(next_axis)
        point_here = np.copy(next_point)
    forward_arc_points = np.array(forward_arc_points)

    def get_backward_arc(input_declination_angle, input_path, sphere_trace):
        axis_at_last_point = np.cross(sphere_trace[0], sphere_trace[1])
        backward_declination_angle = filter_backward_declination(input_declination_angle - default_backward_angle,
                                                                 input_path)
        print(
            f'Backward angle: raw={input_declination_angle},  plusdef={input_declination_angle - default_backward_angle}, filtered={backward_declination_angle}')
        turn_angle_increment = backward_declination_angle / npoints
        geodesic_length_of_single_step = np.abs(min_curvature_radius * backward_declination_angle / npoints)
        point_here = np.copy(sphere_trace[0])
        backward_arc_points = [point_here]
        for i in range(npoints):
            next_axis = rotate_3d_vector(axis_at_last_point, point_here, -1 * turn_angle_increment)
            next_point = rotate_3d_vector(point_here, next_axis, -1 * geodesic_length_of_single_step)
            backward_arc_points.append(next_point)
            axis_at_last_point = np.copy(next_axis)
            point_here = np.copy(next_point)
        backward_arc_points = np.array(backward_arc_points)
        return backward_arc_points

    backward_arc_points = get_backward_arc(input_declination_angle, input_path, sphere_trace)

    # # make sure that they are on the same side -- better version
    reference_plane_normal1 = np.cross(backward_arc_points[-2], forward_arc_points[-2])
    forward_sign1 = np.dot(forward_arc_points[-1] - forward_arc_points[-2], reference_plane_normal1)
    backward_sign1 = np.dot(backward_arc_points[-1] - backward_arc_points[-2], reference_plane_normal1)

    # # make sure that they are on the same side -- inaccurate version
    # reference_plane_normal2 = np.cross(sphere_trace[-1], sphere_trace[0])
    # forward_sign2 = np.dot(forward_arc_points[-1] - sphere_trace[-1], reference_plane_normal2)
    # backward_sign2 = np.dot(backward_arc_points[-1] - sphere_trace[0], reference_plane_normal2)

    # if they are not on the same side, then use opposite declination for backward arc
    if (forward_sign1 * backward_sign1 < 0):  # or (forward_sign2 * backward_sign2 < 0):
        backward_arc_points = get_backward_arc(-1 * input_declination_angle, input_path, sphere_trace)

        reference_plane_normal1 = np.cross(backward_arc_points[-2], forward_arc_points[-2])
        forward_sign1 = np.dot(forward_arc_points[-1] - forward_arc_points[-2], reference_plane_normal1)
        backward_sign1 = np.dot(backward_arc_points[-1] - backward_arc_points[-2], reference_plane_normal1)

        # reference_plane_normal2 = np.cross(sphere_trace[-1], sphere_trace[0])
        # forward_sign2 = np.dot(forward_arc_points[-1] - sphere_trace[-1], reference_plane_normal2)
        # backward_sign2 = np.dot(backward_arc_points[-1] - sphere_trace[0], reference_plane_normal2)
    if (forward_sign1 * backward_sign1 < 0):  # or (forward_sign2 * backward_sign2 < 0):
        # if still not on same side even despite the sign flip
        print('Deflections are never on the same side. Escaping.')
        res = input_path
        is_successful = False
    else:
        pd = pairwise_distances(forward_arc_points, backward_arc_points)
        if np.min(pd) < 2 * geodesic_length_of_single_step:
            print('Intersection of forward and backward arcs. Escaping.')
            res = input_path
            is_successful = False
        else:
            # Find the intersection point of the straight segments
            forward_straight_section_axis = np.cross(forward_arc_points[-2], forward_arc_points[-1])
            backward_straight_section_axis = np.cross(backward_arc_points[-2], backward_arc_points[-1])
            intersection = np.cross(forward_straight_section_axis, backward_straight_section_axis)
            intersection = intersection / np.linalg.norm(intersection)
            # make sure that the intersection is on the proper side of the plane passing through the line connecting
            #   two ends of input path and (0, 0, 0)
            sign_marker = 1
            if np.dot(intersection, reference_plane_normal1) * forward_sign1 < 0:
                intersection = -1 * intersection
                sign_marker = -1
            geodesic_length_from_intersection_to_forward_arc = unsigned_angle_between_vectors(intersection,
                                                                                              forward_arc_points[-1])
            geodesic_length_from_intersection_to_backward_arc = unsigned_angle_between_vectors(intersection,
                                                                                               backward_arc_points[-1])
            angle_at_intersection = unsigned_angle_between_vectors(forward_straight_section_axis,
                                                                   backward_straight_section_axis)
            # if angle_at_intersection > np.pi/2:
            #     angle_at_intersection = np.pi - angle_at_intersection
            geodesic_length_from_intersection_to_tangent_of_main_arc = min_curvature_radius / np.tan(
                angle_at_intersection / 2)
            forward_straight_section_length = geodesic_length_from_intersection_to_forward_arc - \
                                              geodesic_length_from_intersection_to_tangent_of_main_arc
            backward_straight_section_length = geodesic_length_from_intersection_to_backward_arc - \
                                               geodesic_length_from_intersection_to_tangent_of_main_arc
            if (forward_straight_section_length <= 0) or (backward_straight_section_length <= 0):
                print('Impossible to make main arc: intersection too close. Escaping')
                res = input_path
                is_successful = False
            else:
                forward_straight_section_points = bridge_two_points_by_arc(forward_arc_points[-1],
                                                                           rotate_3d_vector(forward_arc_points[-1],
                                                                                            forward_straight_section_axis,
                                                                                            forward_straight_section_length),
                                                                           npoints=npoints)
                backward_straight_section_points = bridge_two_points_by_arc(backward_arc_points[-1],
                                                                            rotate_3d_vector(backward_arc_points[-1],
                                                                                             backward_straight_section_axis,
                                                                                             backward_straight_section_length),
                                                                            npoints=npoints)

                # main arc, version 2
                main_arc_radius = np.sqrt(1 / (1 + (1 / min_curvature_radius) ** 2))
                # Find center of main arc circle
                aa = np.cross(forward_straight_section_points[-1], forward_straight_section_axis)
                bb = np.cross(backward_straight_section_points[-1], backward_straight_section_axis)
                main_arc_center = sign_marker * np.cross(aa, bb)
                main_arc_center = main_arc_center / np.linalg.norm(main_arc_center) * np.sqrt(1 - main_arc_radius ** 2)

                def make_main_arc():
                    start = forward_straight_section_points[-1] - main_arc_center
                    end = backward_straight_section_points[-1] - main_arc_center
                    startlen = np.linalg.norm(start)
                    endlen = np.linalg.norm(end)
                    # assert np.isclose(np.linalg.norm(start), main_arc_radius)
                    # assert np.isclose(np.linalg.norm(end), main_arc_radius)
                    full_turn_angle = -1 * sign_marker * unsigned_angle_between_vectors(start, end)
                    full_turn_angle = -1 * sign_marker * np.arccos(
                        np.dot(start, end) / np.linalg.norm(start) / np.linalg.norm(end))
                    main_arc_points = []
                    thetas = np.linspace(0, full_turn_angle, npoints)
                    for theta in thetas:
                        main_arc_points.append(
                            main_arc_center + rotate_3d_vector(start, main_arc_center / np.linalg.norm(main_arc_center),
                                                               theta))
                    return np.array(main_arc_points)

                main_arc_points = make_main_arc()

                backward_straight_section_points = backward_straight_section_points[::-1]
                backward_arc_points = backward_arc_points[::-1]
                if True:
                    core_radius = 1
                    mfig = mlab.figure(size=(1024, 768), bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
                    tube_radius = 0.01
                    arccolor = tuple(np.array([44, 160, 44]) / 255)
                    arccolor = (1, 0, 0)
                    plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)

                    l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2],
                                    color=tuple(np.array([31, 119, 180]) / 255),
                                    tube_radius=tube_radius, opacity=0.5)
                    for piece_of_bridge in [forward_arc_points, backward_arc_points]:
                        p = mlab.plot3d(piece_of_bridge[:, 0], piece_of_bridge[:, 1], piece_of_bridge[:, 2],
                                        color=arccolor,
                                        tube_radius=tube_radius, opacity=0.5)
                    for piece_of_bridge in [forward_straight_section_points, backward_straight_section_points]:
                        p = mlab.plot3d(piece_of_bridge[:, 0], piece_of_bridge[:, 1], piece_of_bridge[:, 2],
                                        color=tuple(np.array([255, 127, 14]) / 255),
                                        tube_radius=tube_radius, opacity=0.5)
                    for piece_of_bridge in [main_arc_points]:
                        p = mlab.plot3d(piece_of_bridge[:, 0], piece_of_bridge[:, 1], piece_of_bridge[:, 2],
                                        color=arccolor,
                                        tube_radius=tube_radius, opacity=0.5)
                    # for point_here in [backward_arc_points[0]]:
                    #     mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.05, color=(1, 1, 0))
                    mlab.view(elevation=120, azimuth=180, roll=-90)
                    mlab.savefig('tests/figures/{0:.2f}.png'.format(input_declination_angle))
                    if make_animation:
                        mlab.view(elevation=150)
                        for frame_id, azimuth in enumerate(np.linspace(0, 359, 60)):
                            mlab.view(azimuth=azimuth)
                            # camera_radius = 4
                            # mfig.scene.camera.position = [camera_radius*np.cos(azimuth), camera_radius*np.sin(azimuth), -2.30]
                            # print(mfig.actors)
                            # mfig.actors.actor.rotate_y(5)
                            mlab.savefig('tests/figures/frames/{0:08d}.png'.format(frame_id))
                    if mlab_show:
                        mlab.show()
                    else:
                        mlab.close()
                trace_with_bridge = np.concatenate((sphere_trace,
                                                    forward_arc_points[1:],
                                                    forward_straight_section_points[1:],
                                                    main_arc_points[1:],
                                                    backward_straight_section_points[1:],
                                                    backward_arc_points[1:-1]),
                                                   axis=0)

                # check for self-intersections
                if spherical_trace_is_self_intersecting(trace_with_bridge):
                    res = input_path
                    is_successful = False
                    print('Self-intersection of whole trace_with_bridge. Escaping.')
                else:
                    res = path_from_trace(trace_with_bridge)
                    is_successful = True
    return res, is_successful


def plot_bridged_path(path, savetofilename=False, npoints=30, netscale=1, linewidth=5):
    fig, ax = plt.subplots(figsize=(12, 2))
    alphabridge = 0.3
    bridgelen = npoints * 5 - 5
    dxs = [0,
           - path[-1, 0],
           path[-1, 0],
           2 * path[-1, 0]]
    dys = [0,
           - path[-1, 1] + path[0, 1],
           - path[0, 1] + path[-1, 1],
           - path[0, 1] + 2 * path[-1, 1]]
    for k in range(len(dxs)):
        dx = dxs[k]
        dy = dys[k]
        if k == 0:
            alpha = 1
        else:
            alpha = 1
        plt.plot(path[:, 0] + dx,
                 path[:, 1] + dy, '-', alpha=alpha, color='black', linewidth=1, zorder=10)
        plt.plot(path[-(bridgelen):, 0] + dx,
                 path[-(bridgelen):, 1] + dy, '-', alpha=alphabridge, color='C1', linewidth=linewidth)
        plt.plot(path[-(bridgelen):-(bridgelen) + npoints - 1, 0] + dx,
                 path[-(bridgelen):-(bridgelen) + npoints - 1, 1] + dy, '-', alpha=alphabridge, color='red',
                 linewidth=linewidth)
        plt.plot(path[-(bridgelen) + npoints * 2 - 2:-(bridgelen) + npoints * 3 - 3, 0] + dx,
                 path[-(bridgelen) + npoints * 2 - 2:-(bridgelen) + npoints * 3 - 3, 1] + dy, '-', alpha=alphabridge,
                 color='red',
                 linewidth=linewidth)
        plt.plot(path[-(npoints - 1):, 0] + dx,
                 path[-(npoints - 1):, 1] + dy, '-', alpha=alphabridge, color='red',
                 linewidth=linewidth)
    plt.scatter([path[0, 0], path[-1, 0], path[0, 0] + 2 * path[-1, 0]], [path[0, 1], path[-1, 1], 2 * path[-1, 1]],
                s=35, alpha=0.8, color='black', zorder=100)
    # plt.scatter([path[0, 0], path[-1, 0]], [path[0, 1], path[-1, 1]], s=35, alpha=0.8, color='black', zorder=100)

    plt.axis('equal')
    plt.xlim(-8, -8 + 35 * netscale)
    ax.axis('off')
    if savetofilename:
        fig.savefig(savetofilename, dpi=300)
    plt.show()


def make_random_path(Npath=150, amplitude=2, x_span_in_2pis=0.8, seed=1, make_ends_horizontal=False,
                     start_from_zero=True,
                     end_with_zero=False, savgom_window_1=31, savgol_window_2=7):
    np.random.seed(seed)
    xs = np.linspace(0, 2 * np.pi * x_span_in_2pis, Npath)
    ys = np.random.rand(Npath)
    ys = savgol_filter(amplitude * ys, savgom_window_1, 3)
    ys = savgol_filter(ys, savgol_window_2, 1)
    if start_from_zero:
        ys = ys - ys[0]
    if end_with_zero:
        ys = ys - xs * (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    if make_ends_horizontal == 'both':
        ys[1] = ys[0]
        ys[-1] = ys[-2]
    elif make_ends_horizontal == 'last':
        ys[-1] = ys[-2]
    elif make_ends_horizontal == 'first':
        ys[1] = ys[0]

    return np.stack((xs, ys)).T


def blend_two_paths(path1, path2, fraction_of_path1):
    assert np.all(path1.shape == path2.shape)
    assert np.all(path1[:, 0] == path2[:, 0])
    assert fraction_of_path1 <= 1
    assert fraction_of_path1 >= 0
    result = np.copy(path1)
    result[:, 1] = fraction_of_path1 * path1[:, 1] + (1 - fraction_of_path1) * path2[:, 1]
    return result


def get_end_to_end_distance(input_path, uniform_scale_factor):
    sphere_trace = trace_on_sphere(input_path, kx=uniform_scale_factor, ky=uniform_scale_factor)
    return np.linalg.norm(sphere_trace[0] - sphere_trace[-1])


def get_scale_that_minimizes_end_to_end(input_path, minimal_scale=0.1):
    # find the scale factor that gives minimal end-to-end distance
    # the initial guess is such that length in x axis is 2*pi
    initial_x_length = np.abs(input_path[-1, 0] - input_path[0, 0])
    initial_guess = 2 * np.pi / initial_x_length
    print(initial_guess)

    def func(x):
        print(x)
        return [get_end_to_end_distance(input_path, s) for s in x]

    bounds = [[minimal_scale, np.inf]]
    solution = minimize(func, initial_guess, bounds=bounds)
    print(solution)
    return solution.x


def minimize_mismatch_by_scaling(input_path_0, scale_range=(0.8, 1.2)):
    scale_max = scale_range[1]
    scale_min = scale_range[0]
    # if the sign of mismatch angle is same at the ends of the region -- there is no solution
    if mismatch_angle_for_path(input_path_0 * scale_max) * mismatch_angle_for_path(input_path_0 * scale_min) > 0:
        logging.info('Sign of mismatch is the same on both sides of the interval.')
        logging.info(f'Mismatch at max scale = {mismatch_angle_for_path(input_path_0 * scale_min)}')
        logging.info(f'Mismatch at min scale = {mismatch_angle_for_path(input_path_0 * scale_max)}')
        return False

    def left_hand_side(x):  # the function whose root we want to find
        logging.debug(f'Sampling function at x={x}')
        return mismatch_angle_for_path(input_path_0 * x)

    best_scale = brentq(left_hand_side, a=scale_min, b=scale_max, maxiter=80, xtol=0.00001, rtol=0.00005)
    logging.debug(f'Minimized mismatch angle = {left_hand_side(best_scale)}')
    return best_scale


def double_the_path(input_path_0, do_plot=False, do_sort=True):
    # input_path_0 = input_path
    input_path_1 = np.copy(input_path_0)
    # input_path_1[:,0] = 2*input_path_1[-1,0]-input_path_1[:,0]
    input_path_1[:, 0] = input_path_1[-1, 0] + input_path_1[:, 0]
    # input_path_1[:,1] = -1*input_path_1[:,1]

    # input_path_1 = np.concatenate((input_path_0, np.flipud))
    if do_plot:
        plt.plot(input_path_0[:, 0], input_path_0[:, 1], '-', color='C2')  # , label='Asymmetric')
        plt.plot(input_path_1[:, 0], input_path_1[:, 1], '-', color='C0')
        plt.axis('equal')
        # plt.legend(loc='upper left')
        plt.show()

    input_path_0 = np.concatenate((input_path_0, sort_path(input_path_1)[1:, ]), axis=0)
    if do_sort:
        input_path_0 = sort_path(input_path_0)
    if do_plot:
        plt.plot(input_path_0[:, 0], input_path_0[:, 1], '-o', alpha=0.5)
        plt.axis('equal')
        plt.show()

    return sort_path(input_path_0)


def multiply_the_path(input_path_0, m, do_plot=False, do_sort=True):
    input_path_to_append = np.copy(input_path_0)
    multiplied_path = np.copy(input_path_0)
    for i in range(m - 1):
        input_path_to_append[:, 0] = (i + 1) * input_path_0[-1, 0] + input_path_0[:, 0]
        # input_path_to_append[:,1] = -1*input_path_to_append[:,1]
        multiplied_path = np.concatenate((multiplied_path, sort_path(input_path_to_append)[1:, ]), axis=0)
    if do_sort:
        multiplied_path = sort_path(multiplied_path)
    return sort_path(multiplied_path)


## This old implementation is wrong by a integer number of 2*pi
def get_gb_area_deprecated(input_path):
    '''This function does not take into account the possibly changing rotation index of the spherical trace.
    It has to be accounted for in the downstream code.'''
    sphere_trace = trace_on_sphere(input_path, kx=1, ky=1, do_plot=False)

    # get total change of angle
    extended_trace = np.vstack((sphere_trace, sphere_trace[0, :], sphere_trace[1, :]))

    # rotation angles on the extended path
    def rotation_angle_from_point_to_point(i):
        # from ith point to (i+1)th point of the extended path
        if i <= input_path.shape[0] - 2:
            flat_vector_to_previous_point = input_path[i + 1] - input_path[i]
            rotation_angle = np.linalg.norm(flat_vector_to_previous_point)
        elif i == (input_path.shape[0] - 1):
            rotation_angle = np.arccos(np.dot(sphere_trace[0], sphere_trace[-1]))
        elif i == (input_path.shape[0]):
            flat_vector_to_previous_point = input_path[1] - input_path[0]
            rotation_angle = np.linalg.norm(flat_vector_to_previous_point)
        return rotation_angle

    # angle change between consecutive arcs
    angles = []
    sum_angle = 0
    for i in range(extended_trace.shape[0] - 2):
        first_point = extended_trace[i, :]
        central_point = extended_trace[i + 1, :]
        last_point = extended_trace[i + 2, :]
        # These axes must be corrected. Right now they produce wrong rotation index, but correct area up to a multiple of 2*pi.
        # In fact, it should be impossible to get areas just from spherical trace. Angle of each rotation must be additionally known.
        # If closest angles is alpha. Real angle is 2*pi +- alpha. Depending on plus and minus, the direction of turn must be chosen.
        # First, find the actual angles of these two rotations
        first_arc_axis = np.cross(first_point, central_point)
        second_arc_axis = np.cross(central_point, last_point)
        first_arc_axis = first_arc_axis / np.linalg.norm(first_arc_axis)
        second_arc_axis = second_arc_axis / np.linalg.norm(second_arc_axis)
        signed_sine = np.dot(np.cross(first_arc_axis, second_arc_axis),
                             central_point)
        signed_cosine = np.dot(first_arc_axis, second_arc_axis)
        angle = np.arctan2(signed_sine, signed_cosine)
        angles.append(angle)
        sum_angle += angle
        gauss_bonnet_area = 2 * np.pi - sum_angle
    # print(angles)
    logging.debug(f'Sum angle = {sum_angle / np.pi} pi')
    logging.debug(f'Area = {gauss_bonnet_area / np.pi} pi')
    return gauss_bonnet_area


def plot_3d_vector_with_origin(vector, vector_origin, color, tube_radius=0.05):
    vector_end = vector + vector_origin
    points = np.vstack((vector_origin, vector_end))
    mlab.points3d(vector_end[0], vector_end[1], vector_end[2], scale_factor=0.1, color=color)
    mlab.plot3d(points[:, 0],
                points[:, 1],
                points[:, 2], color=color, tube_radius=tube_radius / 5, opacity=0.7)


def get_gb_area(input_path, flat_path_change_of_direction='auto', do_plot=False, return_arc_normal=False):
    '''This function does not take into account the possibly changing rotation index of the spherical trace.
    It has to be accounted for in the downstream code.'''
    sphere_trace = trace_on_sphere(input_path, kx=1, ky=1, do_plot=False)

    # Change of direction of the flat path:
    if flat_path_change_of_direction == 'auto':
        flat_path_change_of_direction = np.sum(
            np.array([signed_angle_between_2d_vectors(input_path[i + 2] - input_path[i + 1],
                                                      input_path[i + 1] - input_path[i])
                      for i in range(input_path.shape[0] - 2)
                      ]))

    # Change of direction due to 2 angles formed by great arc connecting the last and first point

    # direction from point 0 to point 1, expressed as normal of the respective arc on the sphere_trace
    if do_plot:
        tube_radius = 0.05
        core_radius = 1
        plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
        l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(0, 0, 1),
                        tube_radius=tube_radius, opacity=0.3)
        colors = [(1, 0, 0), (0, 1, 0)]
        for i, point_here in enumerate([sphere_trace[0], sphere_trace[-1]]):
            mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.05, color=colors[i])
        # mlab.axes()

    path_start_direction_vector_flat = input_path[1] - input_path[0]
    path_start_direction_vector_flat /= np.linalg.norm(path_start_direction_vector_flat)
    path_start_direction_vector = np.array([path_start_direction_vector_flat[0],
                                            path_start_direction_vector_flat[1],
                                            0])
    path_start_arc_normal = np.cross(np.array([0, 0, -1]), path_start_direction_vector)
    path_start_arc_normal /= np.linalg.norm(path_start_arc_normal)

    # direction from point (-2) to point (-1), expressed as normal of the respective arc on the sphere_trace
    path_end_direction_vector_flat = input_path[-1] - input_path[-2]
    path_end_direction_vector_flat /= np.linalg.norm(path_end_direction_vector_flat)

    # Convert to trimesh point cloud and apply reverse rolling to origin
    point_at_plane = trimesh.PointCloud([[path_end_direction_vector_flat[0],
                                          path_end_direction_vector_flat[1],
                                          -1]])
    point_at_plane.apply_transform(rotation_to_origin(input_path.shape[0] - 1, input_path))
    path_end_direction_vector = point_at_plane.vertices[0] - sphere_trace[-1, :]

    # compute the normal of that rotation arc
    path_end_arc_normal = np.cross(sphere_trace[-1, :], path_end_direction_vector)
    path_end_arc_normal /= np.linalg.norm(path_end_arc_normal)

    # Now calculate the change of direction due to the arc connecting the trace ends
    normal_of_arc_connecting_trace_ends = np.cross(sphere_trace[-1], sphere_trace[0])
    normal_of_arc_connecting_trace_ends /= np.linalg.norm(normal_of_arc_connecting_trace_ends)

    def get_signed_change_of_direction_at_point(first_arc_axis, second_arc_axis, central_point):
        unsigned_sine = np.cross(first_arc_axis, second_arc_axis)
        signed_sine = np.linalg.norm(unsigned_sine) * np.sign(np.dot(unsigned_sine,
                                                                     central_point))
        signed_cosine = np.dot(first_arc_axis, second_arc_axis)
        return np.arctan2(signed_sine, signed_cosine)

    # change of direction computed from the flat path angles
    net_change_of_direction = flat_path_change_of_direction
    logging.debug(f'Flat path change of direction = {net_change_of_direction / np.pi} pi')

    # change of direction from end of trace to the connecting arc
    net_change_of_direction += get_signed_change_of_direction_at_point(path_end_arc_normal,
                                                                       normal_of_arc_connecting_trace_ends,
                                                                       sphere_trace[-1])
    logging.debug(f'Net change of direction path with first angle to arc = {net_change_of_direction / np.pi} pi')

    # change of direction from connecting arc to the start of trace
    net_change_of_direction += get_signed_change_of_direction_at_point(normal_of_arc_connecting_trace_ends,
                                                                       path_start_arc_normal,
                                                                       sphere_trace[0])

    gauss_bonnet_area = 2 * np.pi - net_change_of_direction
    # print(angles)
    logging.debug(f'Net change of direction = {net_change_of_direction / np.pi} pi')
    logging.debug(f'Area = {gauss_bonnet_area / np.pi} pi')

    if do_plot:
        plot_3d_vector_with_origin(vector=path_start_direction_vector, vector_origin=np.array([0, 0, -1]),
                                   color=(0, 0, 1))
        plot_3d_vector_with_origin(vector=path_start_arc_normal, vector_origin=np.array([0, 0, -1]), color=(0, 1, 0))
        plot_3d_vector_with_origin(vector=path_end_direction_vector, vector_origin=sphere_trace[-1, :], color=(0, 0, 1))
        plot_3d_vector_with_origin(vector=path_end_arc_normal, vector_origin=sphere_trace[-1, :], color=(0, 1, 0))
        plot_3d_vector_with_origin(vector=normal_of_arc_connecting_trace_ends, vector_origin=np.array([0, 0, 0]),
                                   color=(1, 1, 0))
        mlab.show()

    if return_arc_normal:
        end_to_end_distance = np.linalg.norm(sphere_trace[0] - sphere_trace[-1])
        return gauss_bonnet_area, normal_of_arc_connecting_trace_ends, end_to_end_distance
    else:
        return gauss_bonnet_area


def gb_areas_for_all_scales(input_path, minscale=0.01, maxscale=2, nframes=100, exclude_legitimate_discont=False,
                            adaptive_sampling=True, diff_thresh=2 * np.pi * 0.1, max_number_of_subdivisions=15):
    '''This function takes into account the possibly changing rotation index of the spherical trace.'''
    gauss_bonnet_areas = []
    connecting_arc_axes = []
    end_to_end_distances = []
    sweeped_scales = np.linspace(minscale, maxscale, nframes)

    flat_path_change_of_direction = np.sum(
        np.array([signed_angle_between_2d_vectors(input_path[i + 2] - input_path[i + 1],
                                                  input_path[i + 1] - input_path[i])
                  for i in range(input_path.shape[0] - 2)
                  ]))

    for frame_id, scale in enumerate(tqdm(sweeped_scales, desc='Computing oriented (Gauss-Bonnet) areas')):
        logging.debug(f'Computing GB_area for scale {scale}')
        input_path_scaled = input_path * scale
        gb_area_here, arc_axis, end_to_end = get_gb_area(input_path_scaled,
                                                         flat_path_change_of_direction,
                                                         return_arc_normal=True)
        gauss_bonnet_areas.append(gb_area_here)
        connecting_arc_axes.append(arc_axis)
        end_to_end_distances.append(end_to_end)

    end_to_end_distances = np.array(end_to_end_distances)

    if adaptive_sampling:
        for subdivision_iteration in range(max_number_of_subdivisions):
            logging.debug(f'Subvidision iteration: {subdivision_iteration}')
            area_diff = np.diff(gauss_bonnet_areas)
            if np.max(area_diff) < diff_thresh:
                break
            insert_before_indices = []
            insert_scales = []
            insert_areas = []
            insert_axes = []
            insert_ends = []
            for i, area in enumerate(gauss_bonnet_areas[:-1]):
                if np.abs(area_diff[i]) > diff_thresh:
                    insert_before_indices.append(i + 1)
                    new_scale_here = (sweeped_scales[i] + sweeped_scales[i + 1]) / 2
                    logging.debug(f'Sampling at new scale {new_scale_here}')
                    insert_scales.append(new_scale_here)
                    if not exclude_legitimate_discont:
                        gb_area_here, arc_axis, end_to_end = get_gb_area(input_path * new_scale_here,
                                                                         flat_path_change_of_direction,
                                                                         return_arc_normal=True)
                    else:
                        gb_area_here = get_gb_area(input_path * new_scale_here,
                                                   flat_path_change_of_direction,
                                                   return_arc_normal=False)
                    insert_areas.append(gb_area_here)
                    insert_axes.append(arc_axis)
                    insert_ends.append(end_to_end)

            sweeped_scales = np.insert(sweeped_scales, insert_before_indices, insert_scales)
            gauss_bonnet_areas = np.insert(gauss_bonnet_areas, insert_before_indices, insert_areas)
            if not exclude_legitimate_discont:
                end_to_end_distances = np.insert(end_to_end_distances, insert_before_indices, insert_ends)
                acc = 0
                for i in range(len(insert_axes)):
                    connecting_arc_axes.insert(insert_before_indices[i] + acc, insert_axes[i])
                    acc += 1

    gb_areas = np.array(gauss_bonnet_areas)
    connecting_arc_axes = tuple(connecting_arc_axes)
    # compensation for integer number of 2*pi due to rotation index of the curve
    gb_area_zero = round(gb_areas[0] / np.pi) * np.pi
    gb_areas -= gb_area_zero
    logging.info(f'Initial gauss-bonnet area is {gb_area_zero / np.pi} pi')

    # correct for changes of rotation index I upon scaling. Upon +1 or -1 change of I, the integral of geodesic curvature
    # (total change of direction) increments or decrements by 2*pi
    additional_rotation_indices = np.zeros_like(gb_areas)
    additional_rotation_index_here = 0
    threshold_for_ind = 2 * np.pi * 0.75
    for i in range(1, gb_areas.shape[0]):
        diff_here = gb_areas[i] - gb_areas[i - 1]
        if np.abs(diff_here) > threshold_for_ind:
            if exclude_legitimate_discont and \
                    ((np.dot(connecting_arc_axes[i], connecting_arc_axes[i - 1]) < 0) and (
                            end_to_end_distances[i] > 1.4) and (end_to_end_distances[i - 1] > 1.4)):
                # if start and end points of trace are antipodal and the arc axis has turned very much,
                # then this change of area is real and the rotation index is unchanged
                # By "very much" we (rather arbitrarily) mean by more than 90 degrees. Turn by more than 90 degrees is equibalent
                #  to having a negative dot product of previous and current axis vectors
                logging.info(f'Legitimate discontinuity of area is found at scale {sweeped_scales[i]}')
            else:
                additional_rotation_index_here += np.round(diff_here / (2 * np.pi))
        additional_rotation_indices[i] = additional_rotation_index_here
    gb_areas -= 2 * np.pi * additional_rotation_indices

    # # Plot additional rotation indices for debugging
    # plt.plot(sweeped_scales, additional_rotation_indices, 'o-')
    # plt.show()

    # plt.plot(sweeped_scales, connecting_arc_axes, 'o-')
    # plt.show()

    return sweeped_scales, gb_areas


def length_of_the_path(input_path_0):
    return np.sum(np.sqrt((np.diff(input_path_0[:, 0]) ** 2 + np.diff(input_path_0[:, 1]) ** 2)))


def cumsum_half_length_along_the_path(input_path_0):
    x = input_path_0[:, 0]
    y = input_path_0[:, 1]
    length_along_the_path = np.cumsum(np.sqrt((np.diff(x) ** 2 + np.diff(y) ** 2)))
    length_along_the_path = np.insert(length_along_the_path, 0, 0)
    length_along_the_path = np.remainder(length_along_the_path, np.max(length_along_the_path) / 2)
    length_along_the_path /= np.max(length_along_the_path)
    return length_along_the_path


def cumsum_full_length_along_the_path(input_path_0):
    x = input_path_0[:, 0]
    y = input_path_0[:, 1]
    length_along_the_path = np.cumsum(np.sqrt((np.diff(x) ** 2 + np.diff(y) ** 2)))
    length_along_the_path = np.insert(length_along_the_path, 0, 0)
    length_along_the_path = np.remainder(length_along_the_path, np.max(length_along_the_path))
    length_along_the_path /= np.max(length_along_the_path)
    return length_along_the_path


def upsample_path(input_path, by_factor=10, kind='linear'):
    old_indices = np.arange(input_path.shape[0])
    max_index = input_path.shape[0] - 1
    new_indices = np.arange(0, max_index, 1 / by_factor)
    new_indices = np.append(new_indices, max_index)
    new_xs = interpolate.interp1d(old_indices, input_path[:, 0], kind=kind)(new_indices)
    new_ys = interpolate.interp1d(old_indices, input_path[:, 1], kind=kind)(new_indices)
    return np.stack((new_xs, new_ys)).T


def plot_flat_path_with_color(input_path, half_of_input_path, axs, linewidth=1, alpha=1,
                              plot_single_period=False):
    '''plotting with color along the line'''
    length_from_start_to_here = cumsum_half_length_along_the_path(input_path)

    if not plot_single_period:
        x = input_path[:, 0]
        y = input_path[:, 1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Coloring the curve
        norm = plt.Normalize(length_from_start_to_here.min(), length_from_start_to_here.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(length_from_start_to_here)
        lc.set_linewidth(linewidth)
        lc.set_alpha(alpha)
        line = axs.add_collection(lc)

        # black dots at middle and ends of the path
        for point in [half_of_input_path[0], half_of_input_path[-1], input_path[-1]]:
            axs.scatter(point[0], point[1], color='black', s=10)

        plt.axis('equal')
    else:
        half_index = half_of_input_path.shape[0]
        x = input_path[:half_index, 0]
        y = input_path[:half_index, 1]
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Coloring the curve
        norm = plt.Normalize(length_from_start_to_here.min(), length_from_start_to_here.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(length_from_start_to_here[:half_index])
        lc.set_linewidth(linewidth)
        lc.set_alpha(alpha)
        line = axs.add_collection(lc)

        # black dots at middle and ends of the path
        for point in [half_of_input_path[0], half_of_input_path[-1]]:
            axs.scatter(point[0], point[1], color='black', s=10)

        plt.axis('equal')


def plot_spherical_trace_with_color_along_the_trace(input_path, input_path_half, scale, plotting_upsample_factor=1,
                                                    sphere_opacity=.8, plot_endpoints=False, endpoint_radius=0.1):
    length_from_start_to_here = cumsum_half_length_along_the_path(input_path)
    sphere_trace = trace_on_sphere(upsample_path(scale * input_path,
                                                 by_factor=plotting_upsample_factor), kx=1, ky=1)
    logging.debug('Mlab plot begins...')
    core_radius = 1
    tube_radius = 0.01
    last_index = sphere_trace.shape[0] // 2
    if USED_3D_PLOTTING_PACKAGE == 'mayavi':
        mfig = mlab.figure(size=(1024, 1024), \
                           bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
        plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4, sphere_opacity=sphere_opacity)
        mlab.plot3d(sphere_trace[:, 0],
                    sphere_trace[:, 1],
                    sphere_trace[:, 2],
                    length_from_start_to_here, colormap='viridis',
                    tube_radius=tube_radius)
        for point_here in [sphere_trace[-1], sphere_trace[0]]:
            mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=endpoint_radius, color=(0, 0, 0))
        return mfig
    elif USED_3D_PLOTTING_PACKAGE == 'plotly':
        fig = go.Figure(data=go.Scatter3d(
            x=sphere_trace[:, 0], y=sphere_trace[:, 1], z=sphere_trace[:, 2],
            marker=dict(
                size=0,
                color=length_from_start_to_here,
                colorscale='Viridis',
            ),
            line=dict(
                color=length_from_start_to_here,
                colorscale='Viridis',
                width=5
            )
        ))

        fig.update_layout(
            width=800,
            height=700,
            autosize=False,
            scene=dict(
                camera=dict(
                    up=dict(
                        x=0,
                        y=0,
                        z=1
                    ),
                    eye=dict(
                        x=0,
                        y=1.0707,
                        z=1,
                    )
                ),
                aspectratio=dict(x=1, y=1, z=1),
                aspectmode='manual'
            ),
        )

        fig.show()
