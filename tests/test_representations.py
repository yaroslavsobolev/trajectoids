import matplotlib.pyplot as plt
import numpy as np

from compute_trajectoid import *

def test_path_from_trace():
    '''we convert path to spherical trace and back, then make sure we got the same path back'''
    # Make a path that is just a single zigzag like so: /\
    # TODO: Test more complex input paths as well for completeness
    Npath = 400
    factor = 0.2
    xs = np.linspace(0, 2*np.pi, Npath)
    ys = xs*factor
    middle = int(round(Npath/2))
    ys[middle:] = factor*np.flip(xs)[middle:]
    input_path = np.stack((xs, ys)).T
    # compute a spherical trace corresponding to flat path
    sphere_trace = trace_on_sphere(input_path, kx=1, ky=1)
    # compute a flat path corresponding to spherical trace
    path_reconstructed_from_trace = path_from_trace(sphere_trace, core_radius=1)
    # compare to the input path
    assert np.isclose(path_reconstructed_from_trace, input_path).all()

    ## optional plotting to debug if something goes wrong
    # plt.scatter(xs, ys, alpha=0.5, color='C0')
    # plt.plot(path_reconstructed_from_trace[:, 0], path_reconstructed_from_trace[:, 1], '-', alpha=0.5, color='C1')
    # plt.show()

# test the bridging mode
def test_single_arc_bridging(do_plot=False):
    true_bridge_points = np.load('tests/bridge_test_points.npy')
    point1 = np.array([0,np.sin(np.pi/4),np.cos(np.pi/4)])
    point2 = np.array([0,-1,0])
    bridge_points = bridge_two_points_by_arc(point1, point2)
    assert np.isclose(bridge_points, true_bridge_points).all()
    if do_plot:
        plot_sphere(1, line_radius=0.01)
        mlab.points3d(point1[0], point1[1], point1[2], scale_factor=0.2, color=(1,0,0))
        mlab.points3d(point2[0], point2[1], point2[2], scale_factor=0.2, color=(0,1,0))
        for point_here in bridge_points:
            mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.1, color=(0,0,1))
        mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.1, color=(0, 1, 0))
        mlab.show()

# def test_angular_bridging():
# test angular bridging

# make_angle_bridge(declination_from_tangents, max_declination_from_gravity)
def test_angle_between_vectors():
    assert np.isclose(signed_angle_between_2d_vectors(np.array([1, 0]), np.array([-0.5, -1])),
                      -2.0344439357957027)

# Make a path that is just a single zigzag like so: /\
# TODO: Test more complex input paths as well for completeness
Npath = 400
factor = 1
factor2 = 0.8
xs = np.linspace(0, 2*np.pi*factor2, Npath)
ys = xs*factor
middle = int(round(Npath/2))
ys[middle:] = factor*np.flip(xs)[middle:]
input_path = np.stack((xs, ys)).T

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

for declination_angle in np.linspace(-np.pi, np.pi, 100):
    plt.scatter(declination_angle, filter_backward_declination(declination_angle, input_path),
                alpha=0.5)
plt.axvline(x=signed_angle_between_2d_vectors(np.array([-1, 0]), input_path[0] - input_path[1]))
# for v in [tangent_backward, candidate_direction_backward, np.array([-1, 0])]:
#     w = v/np.linalg.norm(v)
#     plt.scatter(w[0], w[1])
plt.axis('equal')
plt.show()
print(1)


