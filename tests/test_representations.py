import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

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
    input_path = make_zigzag_path(Npath=400, factor=0.2, factor2=1)
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
    point1 = np.array([0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    point2 = np.array([0, -1, 0])
    bridge_points = bridge_two_points_by_arc(point1, point2)
    assert np.isclose(bridge_points, true_bridge_points).all()
    if do_plot:
        plot_sphere(1, line_radius=0.01)
        mlab.points3d(point1[0], point1[1], point1[2], scale_factor=0.2, color=(1, 0, 0))
        mlab.points3d(point2[0], point2[1], point2[2], scale_factor=0.2, color=(0, 1, 0))
        for point_here in bridge_points:
            mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.1, color=(0, 0, 1))
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


## ================================================


def test_corner_bridge(do_plot = True):
    # input_path_0 = make_zigzag_path(Npath = 150, factor = 0.2, factor2=0.8)

    Npath = 150
    factor = 4
    factor2 = 0.7
    np.random.seed(0)
    xs = np.linspace(0, 2 * np.pi * factor2, Npath)
    ys = np.random.rand(Npath)
    ys = savgol_filter(factor * ys, 31, 3)
    ys = savgol_filter(ys, 7, 1)
    ys[1] = ys[0]
    ys[-1] = ys[-2]
    input_path_0 = np.stack((xs, ys)).T

    sphere_trace = trace_on_sphere(input_path_0, kx=1, ky=1)
    if do_plot:
        core_radius = 1
        mlab.figure(size=(1024, 768), \
                    bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
        tube_radius = 0.01
        plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
        l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(0, 0, 1),
                        tube_radius=tube_radius)
        mlab.show()

    npoints = 30
    netscale = 1
    input_path = netscale*input_path_0
    best_declination_angle = find_best_bridge(input_path, npoints=npoints)
    path_with_bridge = make_corner_bridge_candidate(best_declination_angle, input_path, npoints=npoints, do_plot=False)
    # plot results
    fig, ax = plt.subplots(figsize=(8, 2))
    path = path_with_bridge
    plt.plot(path[:, 0], path[:, 1], '-', alpha=1, color='black', linewidth=2)
    plt.plot(path[:-(2*npoints-2), 0], path[:-(2*npoints-2), 1], '-', alpha=1, color='C1', linewidth=2)
    plt.plot(path[:, 0] - path[-1, 0], path[:, 1] - path[-1, 1] + path[0, 1], '-', alpha=0.3, color='black', linewidth=2)
    plt.plot(path[:-(2*npoints-2), 0] - path[-1, 0], path[:-(2*npoints-2), 1] - path[-1, 1] + path[0, 1], '-', alpha=0.3, color='C1', linewidth=2)
    plt.plot(path[:, 0] + path[-1, 0], path[:, 1] - path[0, 1] + path[-1, 1], '-', alpha=0.3, color='black', linewidth=2)
    plt.plot(path[:-(2*npoints-2), 0] + path[-1, 0], path[:-(2*npoints-2), 1] - path[0, 1] + path[-1, 1], '-', alpha=0.3, color='C1', linewidth=2)
    plt.scatter([path[0, 0], path[-1, 0]], [path[0, 1], path[-1, 1]], alpha=1, color='black')

    plt.axis('equal')
    ax.axis('off')
    plt.show()


    # declination_angles = np.linspace(-np.pi, np.pi, 100)
    # declination_angle = np.pi/2
    # full_bridge = make_corner_bridge_candidate(declination_angle, input_path, npoints=30)

    # declination_angles = np.linspace(-np.pi/2, np.pi/2, 13)
    #
    # mismatches = []
    # print('start')
    # for declination_angle in declination_angles:
    #     mismatches.append(mismatch_angle_for_bridge(declination_angle, input_path, npoints=30))
    #     print('done')
    #
    # plt.plot(declination_angles, mismatches, 'o-')
    # plt.show()

    if do_plot:
        mlab.show()

def test_single_smooth_bridge():
    do_plot = True
    input_path = make_zigzag_path(Npath = 150, factor = 1, factor2=0.6)
    # Npath = 150
    # factor = 4
    # factor2 = 0.7
    # np.random.seed(0)
    # xs = np.linspace(0, 2 * np.pi * factor2, Npath)
    # ys = np.random.rand(Npath)
    # ys = savgol_filter(factor * ys, 31, 3)
    # ys = savgol_filter(ys, 7, 1)
    # ys[1] = ys[0]
    # ys[-1] = ys[-2]
    # input_path_0 = np.stack((xs, ys)).T

    sphere_trace = trace_on_sphere(input_path, kx=1, ky=1)

    # THIS GENERATES CRAZY INTERSECTION OF INITIAL ARCS
    # input_path = make_zigzag_path(Npath = 150, factor = 0.02, factor2=0.94)
    # input_declination_angle = 1*np.pi/2*0.3
    # res = make_smooth_bridge_candidate(input_declination_angle, input_path, npoints=150, min_curvature_radius = 0.5, do_plot = True)

    input_declination_angle = 0.1
    res = make_smooth_bridge_candidate(input_declination_angle, input_path, npoints=150, min_curvature_radius = 0.2, do_plot = True)

# if do_plot:
#     core_radius = 1
#     mlab.figure(size=(1024, 768), \
#                 bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
#     tube_radius = 0.005
#     plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
#     l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(0, 0, 1),
#                     tube_radius=tube_radius, opacity=0.5)
#     for piece_of_bridge in [forward_arc_points, backward_arc_points]:
#         p = mlab.plot3d(piece_of_bridge[:, 0], piece_of_bridge[:, 1], piece_of_bridge[:, 2], color=(0, 1, 0),
#                         tube_radius=tube_radius, opacity=0.5)
#     for piece_of_bridge in [forward_straight_section_points, backward_straight_section_points]:
#         p = mlab.plot3d(piece_of_bridge[:, 0], piece_of_bridge[:, 1], piece_of_bridge[:, 2], color=(1, 0, 0),
#                         tube_radius=tube_radius, opacity=0.5)
#     for piece_of_bridge in [main_arc_points]:
#         p = mlab.plot3d(piece_of_bridge[:, 0], piece_of_bridge[:, 1], piece_of_bridge[:, 2], color=(0, 1, 0),
#                         tube_radius=tube_radius, opacity=0.5)
#     for point_here in [main_arc_center]:
#         mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.1, color=(1, 1, 0))
#     mlab.show()

do_plot = True
# input_path_0 = make_zigzag_path(Npath = 150, factor = 0.1, factor2=0.8)


# ############## start of almost-failing example
# optimal_netscale = 1.15
# Npath = 150
# factor = 2
# # factor2 = 0.8
# factor2 = 0.8
# np.random.seed(0)
# xs = np.linspace(0, 2 * np.pi * factor2, Npath)
# ys = np.random.rand(Npath)
# ys = savgol_filter(factor*ys, 31, 3)
# ys = savgol_filter(ys, 7, 1)
# ys[1] = ys[0]
# ys[-1] = ys[-2]
# ys = ys - ys[0]
# # ys = ys - xs * ys[-1]/(xs[-1])
# input_path_0 =  np.stack((xs, ys)).T
# ####### ================ end of almost-failing parameters

def make_random_path(Npath = 150, factor = 2, factor2 = 0.8, seed=1, make_ends_horizontal=False, start_from_zero=True):
    np.random.seed(seed)
    xs = np.linspace(0, 2 * np.pi * factor2, Npath)
    ys = np.random.rand(Npath)
    ys = savgol_filter(factor*ys, 31, 3)
    ys = savgol_filter(ys, 7, 1)
    if start_from_zero:
        ys = ys - ys[0]
    if make_ends_horizontal:
        ys[1] = ys[0]
        ys[-1] = ys[-2]
    return np.stack((xs, ys)).T

def blend_two_paths(path1, path2, fraction_of_path1):
    assert np.all(path1.shape == path2.shape)
    assert np.all(path1[:,0] == path2[:,0])
    assert fraction_of_path1 <= 1
    assert fraction_of_path1 >= 0
    result = np.copy(path1)
    result[:,1] = fraction_of_path1 * path1[:, 1] + (1-fraction_of_path1) * path2[:, 1]
    return result



def get_end_to_end_distance(input_path, uniform_scale_factor):
    sphere_trace = trace_on_sphere(input_path, kx=uniform_scale_factor, ky=uniform_scale_factor)
    return np.linalg.norm(sphere_trace[0]-sphere_trace[-1])

def get_scale_that_minimizes_end_to_end(input_path, minimal_scale=0.1):
# find the scale factor that gives minimal end-to-end distance

# the initial guess is such that length in x axis is 2*pi
    initial_x_length = np.abs(input_path[-1, 0] - input_path[0, 0])
    initial_guess = 2*np.pi/initial_x_length
    print(initial_guess)
    def func(x):
        print(x)
        return [get_end_to_end_distance(input_path, s) for s in x]
    bounds = [[minimal_scale, np.inf]]
    solution = minimize(func, initial_guess, bounds=bounds)
    print(solution)
    return solution.x

input_path_0 = blend_two_paths(make_random_path(seed=0), make_random_path(seed=1), fraction_of_path1=0.5)

npoints = 30
netscale = 1
input_path = input_path_0

# optimal_netscale = get_scale_that_minimizes_end_to_end(input_path)

# input_path = optimal_netscale * input_path_0

optimal_netscale = 1.15
input_path = optimal_netscale * input_path_0

sphere_trace = trace_on_sphere(input_path, kx=1, ky=1)

if do_plot:
    core_radius = 1
    mlab.figure(size=(1024, 768), \
                bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    tube_radius = 0.01
    plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
    l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(0, 0, 1),
                    tube_radius=tube_radius)
    mlab.show()


### SINGLE ANGLE TEST
declination_angle = 1.3885089773333037
path_with_bridge, is_successful = make_smooth_bridge_candidate(declination_angle, input_path, npoints=npoints, do_plot=True,
                                                               make_animation=True, mlab_show=True)
print(f'Is successful: {is_successful}')
# plot results
fig, ax = plt.subplots(figsize=(8, 2))
path = path_with_bridge
bridgelen = npoints * 5 - 5
plt.plot(path[:, 0], path[:, 1], '-', alpha=1, color='black', linewidth=2)
# plt.show()

plt.plot(path[:-(bridgelen), 0], path[:-(bridgelen), 1], '-', alpha=1, color='C1', linewidth=2)
plt.plot(path[:, 0] - path[-1, 0], path[:, 1] - path[-1, 1] + path[0, 1], '-', alpha=0.3, color='black', linewidth=2)
plt.plot(path[:-(bridgelen), 0] - path[-1, 0], path[:-(bridgelen), 1] - path[-1, 1] + path[0, 1], '-', alpha=0.3,
         color='C1', linewidth=2)
plt.plot(path[:, 0] + path[-1, 0], path[:, 1] - path[0, 1] + path[-1, 1], '-', alpha=0.3, color='black', linewidth=2)
plt.plot(path[:-(bridgelen), 0] + path[-1, 0], path[:-(bridgelen), 1] - path[0, 1] + path[-1, 1], '-', alpha=0.3,
         color='C1', linewidth=2)
plt.scatter([path[0, 0], path[-1, 0]], [path[0, 1], path[-1, 1]], alpha=1, color='black')

plt.axis('equal')
ax.axis('off')
plt.show()

# mfig = mlab.figure(size=(1024, 768), bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
# core_radius = 1
# tube_radius = 0.01
# plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)

## BEST ANGLE TEST
for netscale in np.linspace(optimal_netscale, 0, 20):
    print(f'Netscale={netscale}')
    input_path = netscale * input_path_0
    best_declination_angle = find_best_smooth_bridge(input_path, npoints=npoints)
    if best_declination_angle:
        path_with_bridge, is_successful = make_smooth_bridge_candidate(best_declination_angle, input_path, npoints=npoints, do_plot=True,
                                                                       mlab_show = True, make_animation=True)
        print(f'Scale = {netscale}')
        # plot results
        fig, ax = plt.subplots(figsize=(8, 2))
        path = path_with_bridge
        bridgelen = npoints*5-5
        dxs = [0,
               - path[-1, 0],
                path[-1, 0]]
        dys = [0,
               - path[-1, 1] + path[0, 1],
               - path[0, 1] + path[-1, 1]]
        for k in range(len(dxs)):
            dx = dxs[k]
            dy = dys[k]
            plt.plot(path[:, 0]+dx,
                     path[:, 1]+dy, '-', alpha=1, color='C1', linewidth=2)
            plt.plot(path[:-(bridgelen), 0]+dx,
                     path[:-(bridgelen), 1]+dy, '-', alpha=1, color='C0', linewidth=2)
            plt.plot(path[-(bridgelen):-(bridgelen)+npoints-1, 0]+dx,
                     path[-(bridgelen):-(bridgelen)+npoints-1, 1]+dy, '-', alpha=1, color='red', linewidth=2)
            plt.plot(path[-(bridgelen) + npoints*2 - 2:-(bridgelen) + npoints*3 - 3, 0]+dx,
                     path[-(bridgelen) + npoints*2 - 2:-(bridgelen) + npoints*3 - 3, 1]+dy, '-', alpha=1, color='red',
                     linewidth=2)
            plt.plot(path[-(npoints-1):, 0]+dx,
                     path[-(npoints-1):, 1]+dy, '-', alpha=1, color='red',
                     linewidth=2)
        plt.scatter([path[0, 0], path[-1, 0]], [path[0, 1], path[-1, 1]], s=10, alpha=0.8, color='black', zorder=100)

        plt.axis('equal')
        plt.xlim(-8, -8+25*netscale)
        ax.axis('off')
        fig.savefig(f'tests/figures/2d-path_netscale{netscale}.png', dpi=300)
        plt.show()
        break
    else:
        print('No solution found.')
