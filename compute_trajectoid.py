import numpy as np
from numpy.linalg import norm as lnorm
import trimesh
import matplotlib.pyplot as plt
from skimage import io
from mayavi import mlab
from math import atan2

def signed_angle_between_2d_vectors(vector1, vector2):
    """Calculate the signed angle between two 2-dimensional vectors using the atan2 formula.
    The angle is positive if rotation from vector1 to vector2 is counterclockwise, and negative
    of the rotation is clockwise. Angle is in radians.

    This is more numerically stable for angles close to 0 or pi than the acos() formula.
    """
    # make sure that vectors are 2d
    assert vector1.shape == (2, )
    assert vector2.shape == (2, )
    # Convert to 3D for making cross product
    vector1_ = np.append(vector1, 0)
    vector2_ = np.append(vector2, 0)
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

# def rotate_2d_by_matrix(vector, theta):
#     c, s = np.cos(theta), np.sin(theta)
#     R = np.array(((c, -s), (s, c)))
#     return

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

def plot_sphere(r0, line_radius):
    sphere = mlab.points3d(0, 0, 0, scale_mode='none',
                           scale_factor=2*r0,
                           color=(1, 1, 1),
                           resolution=100,
                           opacity=.8,
                           name='Earth')
    sphere.actor.property.frontface_culling = True

    phi = np.linspace(0, 2*np.pi, 50)
    r = r0 + line_radius
    for theta in np.linspace(0, np.pi, 7)[:-1]:
        mlab.plot3d(
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi), tube_radius=line_radius)

    theta = np.linspace(0, 2*np.pi, 50)
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

        plot_sphere(r = core_radius - tube_radius, line_radius = tube_radius/4)
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
    for i in range(sphere_trace.shape[0]-1):
        # make sure that current (i-th) point is the contact point and therefore coincides with the
        #   downward vector.
        assert np.isclose(vector_downward, sphere_trace[i]).all()
        to_next_point_of_trace = sphere_trace[i+1] - vector_downward

        # find the vector of translation
        theta = np.arccos(-sphere_trace[i + 1, 2] / core_radius)
        arc_length = theta*core_radius
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
        rotation_matrix = trimesh.transformations.rotation_matrix(angle=-1*theta,
                                                direction=axis_of_rotation,
                                                point=[0, 0, 0])
        sphere_trace_cloud.apply_transform(rotation_matrix)
        sphere_trace = np.array(sphere_trace_cloud.vertices)

    translation_vectors = np.array(translation_vectors)
    position_vectors = np.array(position_vectors)
    return position_vectors

def bridge_two_points_by_arc(point1, point2, npoints = 10):
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

def filter_backward_declination(declination_angle, input_path, maximum_angle_from_vertical = np.pi/180*80):
    tangent_backward = input_path[0] - input_path[1]
    angle0 = signed_angle_between_2d_vectors(np.array([-1, 0]), tangent_backward)

    candidate_direction_backward = rotate_2d(tangent_backward, declination_angle)
    a = signed_angle_between_2d_vectors(tangent_backward, candidate_direction_backward)
    assert np.isclose(a, declination_angle)

    # if angle exceeds the highest allowed angle with vertical direction, use the highest allowed angle instead
    angle_from_vertical = signed_angle_between_2d_vectors(np.array([-1, 0]), candidate_direction_backward)
    if np.abs(angle_from_vertical) >= maximum_angle_from_vertical:
        declination_angle = maximum_angle_from_vertical * np.sign(angle_from_vertical) - angle0
    return declination_angle

def filter_forward_declination(declination_angle, input_path, maximum_angle_from_vertical = np.pi/180*80):
    tangent_forward = input_path[-1] - input_path[-2]
    angle0 = signed_angle_between_2d_vectors(np.array([1, 0]), tangent_forward)

    candidate_direction_forward = rotate_2d(tangent_forward, declination_angle)
    a = signed_angle_between_2d_vectors(tangent_forward, candidate_direction_forward)
    assert np.isclose(a, declination_angle)

    # if angle exceeds the highest allowed angle with vertical direction, use the highest allowed angle instead
    angle_from_vertical = signed_angle_between_2d_vectors(np.array([1, 0]), candidate_direction_forward)
    if np.abs(angle_from_vertical) >= maximum_angle_from_vertical:
        declination_angle = maximum_angle_from_vertical * np.sign(angle_from_vertical) - angle0
    return declination_angle

def make_corner_bridge_candidate(input_declination_angle, input_path):
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
    #    same hemisphere as the infinitely small starting sections of that arc.
    return True