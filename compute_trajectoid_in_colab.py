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
from tqdm import tqdm
import logging
import plotly.express as px
import plotly.graph_objects as go


logging.basicConfig(level=logging.INFO)
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


def get_trajectory_from_raster_image(filename, do_plotting=True, resample_to=200):
    image = io.imread(filename)[:, :, 0].T
    image = np.fliplr(image)
    # trajectory_points = np.zeros(shape=(image.shape[0], 2))
    xs = []
    ys = []
    for i in range(image.shape[0]):
        # if all pixels are white then don't add point here
        if np.all(image[i, :] == 255):
            continue
        xs.append(i / image.shape[0] * 2 * np.pi)  # assume that x dimension of path is 2*pi
        # ys.append(np.argmin(image[i, :]) / image.shape[0] * 2 * np.pi)
        ys.append(np.mean(np.argwhere(image[i, :] != 255)) / image.shape[0] * 2 * np.pi)
    # trajectory_points[:, 1] -= trajectory_points[0, 1]  # make path relative to first point
    # trajectory_points = trajectory_points[::5, :]  # decimation by a factor 5
    # print('Decimated to {0} elements'.format(trajectory_points.shape[0]))
    # xs = trajectory_points[:, 0]
    # ys = trajectory_points[:, 1]
    xs = np.array(xs)
    ys = np.array(ys)

    # resample to 200 points
    if resample_to is not None:
        xs_new = np.linspace(xs[0], xs[-1], resample_to)
        ys = np.interp(xs_new, xs, ys)
        xs = xs_new

    ys = ys - ys[0]
    ys = ys - xs * (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    trajectory_points = np.stack((xs, ys)).T

    if do_plotting:
        # print(trajectory_points[0, 1])
        # print(trajectory_points[0, 1] - trajectory_points[-1, 1])
        # plt.imshow(image.T)
        plt.plot(trajectory_points[:, 0], trajectory_points[:, 1])
        plt.axis('equal')
        plt.show()
    return trajectory_points


def get_trajectory_from_csv(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data


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
        # print('Saving box for cutting: {0}'.format(i))
        box.export('{0}/test_{1}.stl'.format(folder_for_meshes, i))


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
    # plt.show()


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


def mismatch_angle_for_path(input_path, recursive=False, use_cache=False):
    rotation_of_entire_traj = trimesh.transformations.rotation_from_matrix(
        rotation_to_origin(input_path.shape[0] - 1, input_path, recursive=recursive, use_cache=use_cache))
    angle = rotation_of_entire_traj[0]
    return angle


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
