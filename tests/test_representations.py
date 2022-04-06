import matplotlib.pyplot as plt
import numpy as np

from compute_trajectoid import *

def make_zigzag_path(Npath, factor, factor2):
    '''Make a path that is just a single zigzag like so: /\ '''
    xs = np.linspace(0, 2 * np.pi * factor2, Npath)
    ys = xs * factor
    middle = int(round(Npath / 2))
    ys[middle:] = factor * np.flip(xs)[middle:]
    return np.stack((xs, ys)).T

def test_path_from_trace():
    '''we convert path to spherical trace and back, then make sure we got the same path back'''
    # Make a path that is just a single zigzag like so: /\
    # TODO: Test more complex input paths as well for completeness
    input_path = make_zigzag_path(Npath = 400, factor = 0.2, factor2=1)
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

def test_filtering_of_backward_declination_angles(do_plot=False):
    # Make a path that is just a single zigzag like so: /\
    # TODO: Test more complex input paths as well for completeness
    input_path = make_zigzag_path(Npath = 400, factor = 1, factor2=0.8)
    declination_angles = np.linspace(-np.pi, np.pi, 100)
    filtered_declination_angles = np.array([filter_backward_declination(declination_angle, input_path,
                                                                        maximum_angle_from_vertical=np.pi / 180 * 80
                                                                        )
                                               for declination_angle in declination_angles])
    # np.save('test/true_backward_declination_angles.npy', filtered_declination_angles)
    assert np.isclose(filtered_declination_angles, np.load('tests/true_backward_declination_angles.npy')).all()
    if do_plot:
        plt.scatter(declination_angles, filtered_declination_angles)
        plt.axvline(x=-1*signed_angle_between_2d_vectors(np.array([-1, 0]), input_path[0] - input_path[1]))
        plt.axis('equal')
        plt.show()

def test_filtering_of_backward_declination_angles(do_plot=False):
    input_path = make_zigzag_path(Npath = 400, factor = 0.2, factor2=0.8)
    declination_angles = np.linspace(-np.pi, np.pi, 100)
    filtered_declination_angles = np.array([filter_backward_declination(declination_angle, input_path)
                                            for declination_angle in declination_angles])
    # np.save('test/true_backward_declination_angles.npy', filtered_declination_angles)
    if do_plot:
        plt.scatter(declination_angles, filtered_declination_angles)
        maximum_angle_from_vertical = np.pi/180*80
        plt.axvline(x=-1 * signed_angle_between_2d_vectors(np.array([-1, 0]), input_path[0] - input_path[1]))
        plt.axvline(x=maximum_angle_from_vertical, color='green')
        plt.axvline(x=-1*maximum_angle_from_vertical, color='green')
        plt.axis('equal')
        plt.show()
