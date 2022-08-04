import numpy as np

from compute_trajectoid import *
# import matplotlib.pyplot as plt
# import numpy as np
# import mayavi
from scipy.interpolate import interp1d
# import plotly.express as px
from matplotlib.gridspec import GridSpec
import time

def double_the_path_nosort(input_path_0, do_plot=False):
    # input_path_0 = input_path
    input_path_1 = np.copy(input_path_0)
    # input_path_1[:,0] = 2*input_path_1[-1,0]-input_path_1[:,0]
    input_path_1[:, 0] = input_path_1[-1, 0] + input_path_1[:, 0]
    # input_path_1[:,1] = -1*input_path_1[:,1]

    # input_path_1 = np.concatenate((input_path_0, np.flipud))
    if do_plot:
        plt.plot(input_path_0[:, 0], input_path_0[:, 1], '-', color='C2')#, label='Asymmetric')
        plt.plot(input_path_1[:, 0], input_path_1[:, 1], '-', color='C0')
        plt.axis('equal')
        # plt.legend(loc='upper left')
        plt.show()

    input_path_0 = np.concatenate((input_path_0, input_path_1[1:, ]), axis=0)
    # if do_sort:
    #     input_path_0 = sort_path(input_path_0)
    if do_plot:
        plt.plot(input_path_0[:, 0], input_path_0[:, 1], '-o', alpha=0.5)
        plt.axis('equal')
        plt.show()

    return input_path_0

def make_path_nonuniform(xlen, r, Npath = 400):
    # factor = 0.2
    xs = np.linspace(0, xlen, Npath)
    r0 = xlen/2
    ys = []
    for x in xs:
        if x <= r0-r or x >= r0 + r:
            y = 0
        else:
            y = np.sqrt(r**2 - (x-r0)**2)
        ys.append(y)
    ys = np.array(ys)
    input_path = np.stack((xs, ys)).T
    return input_path

def make_path(xlen, r, shift=0.25, Npath = 400, do_double=True):
    # first linear section
    step_size = xlen/Npath
    overall_xs = np.linspace(0, xlen/2 - r - shift, int(round((xlen/2 - r - shift)/step_size)))
    overall_ys = np.zeros_like(overall_xs)

    # semicirle section
    nsteps_in_theta = int(round(np.pi*r/step_size))
    thetas = np.linspace(np.pi, 0, nsteps_in_theta)
    xs = r*np.cos(thetas) + xlen/2 - shift
    ys = r*np.sin(thetas)
    overall_xs = np.concatenate((overall_xs[:-1], xs))
    overall_ys = np.concatenate((overall_ys[:-1], ys))

    # second linear section
    xs = np.linspace(xlen/2 + r - shift, xlen, int(round((xlen/2 - r + shift)/step_size)))
    ys = np.zeros_like(xs)
    overall_xs = np.concatenate((overall_xs, xs[1:]))
    overall_ys = np.concatenate((overall_ys, ys[1:]))

    input_path = np.stack((overall_xs, overall_ys)).T

    if do_double:
        input_path = double_the_path(input_path)

    return input_path

def align_view(scene):
    scene.scene.camera.position = [0.8338505129447937, -4.514338837405451, -4.8515133455799955]
    scene.scene.camera.focal_point = [0.00025303918109570445, 0.0, -0.007502121504843806]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.04727430775215174, -0.7266741891255103, 0.6853537500337594]
    scene.scene.camera.clipping_range = [3.573025799931812, 10.592960367304393]
    scene.scene.camera.compute_view_plane_normal()

def mismatches_for_all_scales(input_path, minscale=0.01, maxscale=2, nframes = 100, verbose=False):
    mismatch_angles = []
    sweeped_scales = np.linspace(minscale, maxscale, nframes)
    for frame_id, scale in enumerate(tqdm(sweeped_scales, desc='Computing mismatch for all scales')):
        logging.debug(f'Computing mismatch for scale {scale}')
        input_path_single_section = input_path * scale
        mismatch_angles.append(mismatch_angle_for_path(input_path_single_section, recursive=False, use_cache=False))
    return sweeped_scales, np.array(mismatch_angles)

def make_brownian_path(Npath = 150, seed=0, travel_length=0.1, end_with_zero=True):
    np.random.seed(seed)
    angles = np.random.random_sample(Npath)*2*np.pi
    xs = np.cumsum(np.cos(angles)*travel_length)
    xs = xs - xs[0]
    ys = np.cumsum(np.sin(angles)*travel_length)
    ys = ys - ys[0]
    if end_with_zero:
        ys = ys - xs*(ys[-1] - ys[0])/(xs[-1] - xs[0])
    return np.stack((xs, ys)).T

def make_archimedes_spiral(turns, rate_parameter, npoints, noise_amplitude, end_with_zero=True, seed=0):
    np.random.seed(seed)
    phi_max = turns * 2 * np.pi
    c1 = phi_max / np.sqrt(2*npoints)
    phis = c1 * np.sqrt(2 * np.arange(0, npoints))
    noise = noise_amplitude * (2 * np.random.random_sample(npoints) - 1)
    rs = phis * rate_parameter + noise
    xs = rs * np.cos(phis)# + noise_amplitude * np.cos(angles_for_noise)
    ys = rs * np.sin(phis)# + noise_amplitude * np.sin(angles_for_noise)
    xs = xs - xs[0]
    ys = ys - ys[0]
    if end_with_zero:
        ys = ys - xs*(ys[-1] - ys[0])/(xs[-1] - xs[0])
    plt.plot(xs, ys)
    return np.stack((xs, ys)).T

def make_narrow(npoints, shift=0.05, noise_amplitude=0, end_with_zero=True, seed=0, upsample_by=3):
    np.random.seed(seed)
    noise = noise_amplitude * (2 * np.random.random_sample(npoints) - 1)
    xs = np.linspace(0, 2, int(round(npoints/2)))
    ys = np.zeros_like(xs)
    ys[1:-1] += -1*shift
    xs2 = np.linspace(2, 0, int(round(npoints/2)))
    ys2 = np.zeros_like(xs2)
    ys2[1:-1] += shift
    xs = np.concatenate((xs[:-1], xs2))
    ys = np.concatenate((ys[:-1], ys2))
    xs = xs - xs[0]
    ys = ys - ys[0]

    upsampled_path = upsample_path(np.stack((xs, ys)).T, by_factor=upsample_by)
    xs = upsampled_path[:, 0]
    ys = upsampled_path[:, 1]

    spline_xs = np.array([0, 0.5, 1.2, 1.5, 2])
    spline_ys = np.array([0, 0.5, 0.3, -0.3, 0])
    s = interpolate.InterpolatedUnivariateSpline(spline_xs, spline_ys)
    ys = ys + s(xs)
    xs = xs - xs[0]
    ys = ys - ys[0]
    # if end_with_zero:
    #     ys = ys - xs*(ys[-1] - ys[0])/(xs[-1] - xs[0])
    plt.plot(xs, ys)
    return np.stack((xs, ys)).T

def make_sine(npoints, shift=0.05, end_with_zero=True, seed=0):
    freq = np.sqrt(2)*3
    np.random.seed(seed)
    xs = np.linspace(0, 2*np.pi, npoints)
    # ys = np.sin(1/(xs + 0.001))
    # ys = (5+5/(xs+1)**2)*np.sin(xs*freq)
    ys = 20*np.sin(xs/2)

    # xs2 = np.linspace(2*np.pi, 0, npoints)
    # ys2 = np.sin(xs2*freq)*(1-shift)
    # xs = np.concatenate((xs[:-1], xs2))
    # ys = np.concatenate((ys[:-1], ys2))

    xs = xs - xs[0]
    ys = ys - ys[0]

    # spline_xs = np.array([0, 0.5, 1.2, 1.5, 2])
    # spline_ys = np.array([0, 0.5, 0.3, -0.3, 0])
    # s = interpolate.InterpolatedUnivariateSpline(spline_xs, spline_ys)
    # ys = ys + s(xs)
    if end_with_zero:
        ys = ys - xs*(ys[-1] - ys[0])/(xs[-1] - xs[0])
    plt.plot(xs, ys)
    return np.stack((xs, ys)).T

def add_interval(startpoint, angle, length, Ns=30):
    xs = startpoint[0] + np.linspace(0, length * np.cos(angle), Ns)
    ys = startpoint[1] + np.linspace(0, length * np.sin(angle), Ns)
    return np.stack((xs, ys)).T


def make_zigzag(a, Ns=15):
    angles = [-np.pi * 3 / 8, np.pi * 3 / 8]  # , -np.pi/4, np.pi/4, -np.pi/4, np.pi/4]
    lengths = [a, np.pi / 2]  # , np.pi/2, a, np.pi/2, np.pi/2]
    input_path = np.array([[0, 0]])
    tips = [[0, 0]]
    for i, angle in enumerate(angles):
        startpoint = input_path[-1, :]
        new_section = add_interval(startpoint, angle, lengths[i], Ns=Ns)
        input_path = np.concatenate((input_path, new_section[1:]), axis=0)
        tips.append(new_section[-1])
    return input_path, np.array(tips)


def make_zigzag2(a):
    angles = [-np.pi * 3 / 8, np.pi * 3 / 8]  # , -np.pi/4, np.pi/4, -np.pi/4, np.pi/4]
    lengths = [a, np.pi / 2]  # , np.pi/2, a, np.pi/2, np.pi/2]
    input_path = np.array([[0, 0]])
    tips = [[0, 0]]
    for i, angle in enumerate(angles):
        startpoint = input_path[-1, :]
        new_section = add_interval(startpoint, angle, lengths[i], Ns=10)
        input_path = np.concatenate((input_path, new_section[1:]), axis=0)
        tips.append(new_section[-1])
    input_path[:, 1] = input_path[:, 1] + 0.15 * np.cos(input_path[:, 0] / input_path[-1, 0] * 4 * np.pi)
    input_path[:, 1] = input_path[:, 1] - input_path[0, 1]
    return input_path, np.array(tips)


def make_zigzag_tapered(zigzag_edge_length_without_taper=np.pi / 2,
                        zigzag_corner_angle=np.pi / 4,
                        taper_ratio=0.3, Ns=3):
    distance_from_taper_start_to_default_corner = taper_ratio * zigzag_edge_length_without_taper / 2
    taper_length = 2 * distance_from_taper_start_to_default_corner * np.sin(zigzag_corner_angle / 2)
    input_path = np.array([[0, 0]])
    angles = [0,
              -1 * (np.pi / 2 - zigzag_corner_angle / 2),
              0,
              np.pi / 2 - zigzag_corner_angle / 2,
              0]  # , -np.pi/4, np.pi/4, -np.pi/4, np.pi/4]
    lengths = [taper_length / 2,
               zigzag_edge_length_without_taper - 2 * distance_from_taper_start_to_default_corner,
               taper_length,
               zigzag_edge_length_without_taper - 2 * distance_from_taper_start_to_default_corner,
               taper_length / 2]
    tips = [[0, 0]]
    for i, angle in enumerate(angles):
        startpoint = input_path[-1, :]
        new_section = add_interval(startpoint, angle, lengths[i], Ns=Ns)
        input_path = np.concatenate((input_path, new_section[1:]), axis=0)
        tips.append(new_section[-1])
    input_path[:, 1] = input_path[:, 1] - input_path[0, 1]
    return input_path, np.array(tips)


def make_zigzag_kinked(zigzag_edge_length_without_kink=np.pi / 2,
                        zigzag_corner_angle=np.pi / 4,
                        kink_angle_1=0.1, kink_angle_2=0.4, Ns=3):
    input_path = np.array([[0, 0]])
    gamma = np.pi - kink_angle_1 - kink_angle_2
    length_of_segment_1 = zigzag_edge_length_without_kink / np.sin(gamma) * np.sin(kink_angle_2)
    length_of_segment_2 = zigzag_edge_length_without_kink / np.sin(gamma) * np.sin(kink_angle_1)
    angles = [-1 * (np.pi / 2 - zigzag_corner_angle / 2) - kink_angle_1,
              -1 * (np.pi / 2 - zigzag_corner_angle / 2) + kink_angle_2,
              np.pi / 2 - zigzag_corner_angle / 2 + kink_angle_2,
              np.pi / 2 - zigzag_corner_angle / 2 - kink_angle_1]
    lengths = [length_of_segment_1, length_of_segment_2, length_of_segment_2, length_of_segment_1]
    tips = [[0, 0]]
    for i, angle in enumerate(angles):
        startpoint = input_path[-1, :]
        new_section = add_interval(startpoint, angle, lengths[i], Ns=Ns)
        input_path = np.concatenate((input_path, new_section[1:]), axis=0)
        tips.append(new_section[-1])
    input_path[:, 1] = input_path[:, 1] - input_path[0, 1]
    return input_path, np.array(tips)

def make_zigzag_kinked_asymm(zigzag_edge_length_without_kink=np.pi / 2,
                        zigzag_corner_angle=np.pi / 4,
                        kink_angle_1=0.2, kink_angle_2=0.8, Ns=3, asymmetry=0.6):
    kink_angle_1_b = kink_angle_1 * (1 + asymmetry)
    kink_angle_2_b = kink_angle_2 * (1 + asymmetry)

    input_path = np.array([[0, 0]])
    gamma = np.pi - kink_angle_1 - kink_angle_2
    length_of_segment_1 = zigzag_edge_length_without_kink / np.sin(gamma) * np.sin(kink_angle_2)
    length_of_segment_2 = zigzag_edge_length_without_kink / np.sin(gamma) * np.sin(kink_angle_1)

    gamma_b = np.pi - kink_angle_1_b - kink_angle_2_b
    length_of_segment_1_b = zigzag_edge_length_without_kink / np.sin(gamma_b) * np.sin(kink_angle_2_b)
    length_of_segment_2_b = zigzag_edge_length_without_kink / np.sin(gamma_b) * np.sin(kink_angle_1_b)

    angles = [-1 * (np.pi / 2 - zigzag_corner_angle / 2) - kink_angle_1,
              -1 * (np.pi / 2 - zigzag_corner_angle / 2) + kink_angle_2,
              np.pi / 2 - zigzag_corner_angle / 2 + kink_angle_2_b,
              np.pi / 2 - zigzag_corner_angle / 2 - kink_angle_1_b]
    lengths = [length_of_segment_1, length_of_segment_2, length_of_segment_2_b, length_of_segment_1_b]
    tips = [[0, 0]]
    for i, angle in enumerate(angles):
        startpoint = input_path[-1, :]
        new_section = add_interval(startpoint, angle, lengths[i], Ns=Ns)
        input_path = np.concatenate((input_path, new_section[1:]), axis=0)
        tips.append(new_section[-1])
    input_path[:, 1] = input_path[:, 1] - input_path[0, 1]
    return input_path, np.array(tips)

def add_arc(center, start_angle, end_angle, radius, npoints):
    angles = np.linspace(start_angle, end_angle, npoints)
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    return center + np.stack((xs, ys)).T

def make_zigzag_with_smoothed_corner(zigzag_edge_length_without_smoothing=np.pi / 2,
                        zigzag_corner_angle=np.pi / 4,
                        radius_of_curvature=0.02,
                                     Ns=3, halfarc_segments_number=15):
    '''Makes an zigzag whose corners are smoothed by a given radius of curvature.
    It consists of two straight lines and three arcs.'''

    # add first flat arc segment
    arc_angle_span = np.pi/2 - zigzag_corner_angle/2
    center = np.array([0,
                       -1 * radius_of_curvature / np.cos(arc_angle_span)])
    input_path = add_arc(center, start_angle=np.pi/2, end_angle=np.pi/2 - arc_angle_span,
                         radius=radius_of_curvature, npoints=halfarc_segments_number)

    # add first straight segment
    length_of_straight_segment = zigzag_edge_length_without_smoothing - 2 * radius_of_curvature * np.tan(arc_angle_span)
    startpoint = input_path[-1, :]
    new_section = add_interval(startpoint, angle=-1 * (np.pi / 2 - zigzag_corner_angle / 2),
                               length=length_of_straight_segment, Ns=Ns)
    input_path = np.concatenate((input_path, new_section[1:]), axis=0)

    # Add middle arc
    center = np.array([zigzag_edge_length_without_smoothing * np.sin(zigzag_corner_angle / 2),
                       -1 * zigzag_edge_length_without_smoothing * np.cos(zigzag_corner_angle / 2) +
                                                                        radius_of_curvature / np.cos(arc_angle_span)])
    new_section = add_arc(center, start_angle=-np.pi/2 - arc_angle_span, end_angle=-np.pi/2 + arc_angle_span,
                         radius=radius_of_curvature, npoints=halfarc_segments_number * 2)
    input_path = np.concatenate((input_path, new_section[1:]), axis=0)

    # add first straight segment
    length_of_straight_segment = zigzag_edge_length_without_smoothing - 2 * radius_of_curvature * np.tan(arc_angle_span)
    startpoint = input_path[-1, :]
    new_section = add_interval(startpoint, angle=np.pi / 2 - zigzag_corner_angle / 2,
                               length=length_of_straight_segment, Ns=Ns)
    input_path = np.concatenate((input_path, new_section[1:]), axis=0)

    # add last flat arc segment
    arc_angle_span = np.pi/2 - zigzag_corner_angle/2
    center = np.array([zigzag_edge_length_without_smoothing * np.sin(zigzag_corner_angle / 2) * 2,
                       -1 * radius_of_curvature / np.cos(arc_angle_span)])
    new_section = add_arc(center, start_angle=np.pi/2 + arc_angle_span, end_angle=np.pi/2,
                         radius=radius_of_curvature, npoints=halfarc_segments_number)
    input_path = np.concatenate((input_path, new_section[1:]), axis=0)

    # input_path = np.array([[0, 0]])

    # angles = [-1 * (np.pi / 2 - zigzag_corner_angle / 2),
    #           np.pi / 2 - zigzag_corner_angle / 2,]
    # lengths = [length_of_straight_segment] * 2
    # for i, angle in enumerate(angles):
    #     startpoint = input_path[-1, :]
    #     new_section = add_interval(startpoint, angle, lengths[i], Ns=Ns)
    #     input_path = np.concatenate((input_path, new_section[1:]), axis=0)
    #
    # input_path[:, 1] = input_path[:, 1] - input_path[0, 1]
    return input_path

# input_path = make_zigzag_with_smoothed_corner()
# plt.plot(input_path[:, 0], input_path[:, 1], 'o-', alpha=0.3)
# plt.axis('equal')
# plt.show()

# def make_zigzag2(a):
#     beta = np.pi/8
#     alpha = np.pi/16
#     angles = [alpha, alpha + np.pi - beta, alpha + np.pi - beta - alpha - np.pi, 0]#, -np.pi/4, np.pi/4]
#     lengths = [a, np.pi/2, np.pi/2, np.pi/2]#, np.pi/2]
#     input_path = np.array([[0, 0]])
#     tips = [[0, 0]]
#     for i, angle in enumerate(angles):
#         startpoint = input_path[-1,:]
#         new_section = add_interval(startpoint, angle, lengths[i])
#         input_path = np.concatenate((input_path, new_section[1:]), axis=0)
#         tips.append(new_section[-1])
#     xs = input_path[:, 0]
#     ys = input_path[:, 1]
#     return input_path, np.array(tips)

def plot_mismatches_vs_scale(ax, input_path, sweeped_scales, mismatch_angles, mark_one_scale, scale_to_mark):
    ## This code injects the sampled point at the solution_scale. Otherwise the root can be not at one of sampled points
    if mark_one_scale:
        ii = np.searchsorted(sweeped_scales, scale_to_mark)
        sweeped_scales = np.insert(sweeped_scales, ii, scale_to_mark)
        # value_at_scale = 0
        value_at_scale_to_mark = mismatch_angle_for_path(input_path * scale_to_mark, recursive=False, use_cache=False)
        mismatch_angles = np.insert(mismatch_angles, ii, value_at_scale_to_mark)
    ax.plot(sweeped_scales, np.abs(mismatch_angles)/np.pi*180)
    # ax.plot(sweeped_scales, mismatch_angles / np.pi * 180)
    ax.axhline(y=0, color='black')
    if mark_one_scale:
        value_at_scale_to_mark = interp1d(sweeped_scales, mismatch_angles)(scale_to_mark)
        ax.scatter([scale_to_mark], [180/np.pi*np.abs(value_at_scale_to_mark)], s=20, color='red')
    ax.set_yticks(np.arange(0, 181, 20))
    ax.set_ylim(-5, 181)
    ax.set_ylabel('Mismatch angle (deg.)\nbetween initial and\nfinal orientations after\npassing two periods')


def plot_gb_areas(ax, sweeped_scales, gb_areas, mark_one_scale, scale_to_mark):

    ## This code injects the sampled point at the solution_scale. Otherwise the root can be not at one of sampled points
    # ii = np.searchsorted(sweeped_scales, solution_scale)
    # sweeped_scales = np.insert(sweeped_scales, ii, solution_scale)
    # mismatch_angles = np.insert(mismatch_angles, ii, np.pi * np.sign(interp1d(sweeped_scales, gb_areas)(solution_scale)))

    ax.plot(sweeped_scales, gb_areas)
    ax.axhline(y=np.pi, color='black', alpha=0.5)
    ax.axhline(y=0, color='black', alpha=0.3)
    ax.axhline(y=-1 * np.pi, color='black', alpha=0.5)
    if mark_one_scale:
        # ax.scatter([solution_scale], [np.pi * np.sign(interp1d(sweeped_scales, gb_areas)(solution_scale))], s=20, color='red')
        value_at_scale_to_mark = interp1d(sweeped_scales, gb_areas)(scale_to_mark)
        ax.scatter([scale_to_mark], [value_at_scale_to_mark], s=20, color='red')
    ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
    ax.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
    ax.set_ylim(-np.pi * 2 * 1.01, np.pi * 2 * 1.01)
    ax.set_ylabel('Spherical area $S(\sigma)$\n enclosed by the \nfirst period\'s trace')
    ax.set_xlabel('Path\'s scale factor $\sigma$ for fixed ball radius $r=1$\n(or $1/r$ for fixed $\sigma=1$)')


def select_path_by_path_type(path_parameter, path_type):
    if path_type == 'brownian':
        input_path_single_section = make_brownian_path(seed=0, Npath=150, travel_length=0.1)
        input_path_single_section = upsample_path(input_path_single_section, by_factor=5)
    elif path_type == 'spiral':
        input_path_single_section = make_archimedes_spiral(turns=5,
                                                           rate_parameter=0.1,
                                                           npoints=150, noise_amplitude=0.2)
        input_path_single_section = upsample_path(input_path_single_section, by_factor=5)
    elif path_type == 'narrow':
        # # this one worked with best_scale = 79.35082181975892,
        # input_path_single_section = make_narrow(npoints=150)
        input_path_single_section = make_narrow(npoints=150)
        # input_path_single_section = upsample_path(input_path_single_section, by_factor=20)
    elif path_type == 'sine':
        input_path_single_section = make_sine(npoints=800)
    elif path_type == 'brownian-smooth':
        input_path_single_section = make_brownian_path(seed=0, Npath=150, travel_length=0.1)
        input_path_single_section = upsample_path(input_path_single_section, by_factor=20, kind='cubic')
    elif path_type == 'zigzag':
        input_path_single_section, tips = make_zigzag(np.pi / 2)
    elif path_type == 'zigzag_2':
        input_path_single_section, tips = make_zigzag2(np.pi / 2)
    elif path_type == 'zigzag_tapered':
        input_path_single_section, tips = make_zigzag_tapered(taper_ratio=path_parameter)
    elif path_type == 'zigzag_kinked':
        input_path_single_section, tips = make_zigzag_kinked(kink_angle_1=path_parameter, kink_angle_2=path_parameter*4)
    elif path_type == 'zigzag_kinked_asymmetric':
        input_path_single_section, tips = make_zigzag_kinked_asymm(asymmetry=path_parameter)
    elif path_type == 'zigzag_smoothed':
        input_path_single_section = make_zigzag_with_smoothed_corner(radius_of_curvature=path_parameter)
    return input_path_single_section


def test_trajectoid_existence(path_type='brownian', path_for_figs='examples/brownian_path_1/figures',
                              forced_best_scale = False,
                              nframes=300,
                              minscale=0.02,
                              maxscale = 26,
                              figsizefactor = 0.85,
                              circle_center=[0, 0],
                              circlealpha=1, plot_solution=True,
                              range_for_searching_the_roots='auto',
                              do_plot = True,
                              path_parameter=0.1,
                              path_for_united_fig=False,
                              fig_title='Path parameter: ',
                              trace_upsample_factor = 100,
                              path_linewidth=1,
                              path_alpha=1,
                              plot_single_period=False
                              ):
    """
    Tests existence of two-period trajectoid for a given path. It will also plot the mismatch angle and the
    Gauss-Bonnet area enclosed by the first period and great arc connecting its ends. Mismatch angles and areas will be
    evaluated for a number 'nframes' values scale factor in the interval between 'minscale' and 'maxscale'

    :param path_type: String. Chooses the input path to be tested. So far, can take values "brownian", "spiral", "narrow",
                        "sine", "brownian-smooth", "zigzag", "zigzag2".
    :param path_for_figs: String (path). Path to folder where the output plots will be saved.
    :param forced_best_scale: Bool "False" or Float. As a float value, controls the value of scale factor for which the first colored
                              spherical trace will be plotted. It must be between the minscale and maxscale.
                              Red dot on the plots will also be plotted at this scale factor. If 'False', the plotting will
                              instead be done for the true "best scale" found by true minimization on the previous
                              launch of this script.
    :param nframes: Integer. Number of values of scale factor at which enclosed areas and angular
                    mismatch will be evaluated.
    :param minscale: Float. Minimum value of scale factor to be plotted.
    :param maxscale: Float. Maximum value of scale factor to be plotted.
    :param figsizefactor: Scales the output images. This is only useful for making proper font sizes in the final plots
                          in the paper
    :param circle_center: Tuple of two floats. Location of circle illustrating the size of the rolling sphere relative to the path.
    :param circlealpha: Float. Alpha (opacity) of circle illustrating the size of the rolling sphere relative to the path.
    :param plot_solution: Bool. Whether to plot the red dot (at 'forced_best_scale') on the plots of area and mismatch angle.
    :param range_for_searching_the_roots: String "auto" or a tuple of two float values. The best scale minimizing the mismatch angle will be automatically searched
                                          in this range by a root-finding algorithm. If set to "auto", the first crossing of pi by enclosed are
                                          will be used as right end of range, and the previous point as the lft end of range.
    :param do_plot: Boolean. Whether to plot some of the plots.
    :param path_parameter: Optional parameter that controls features of some types of the path.
    """
    # input_path_single_section = make_random_path(seed=1, amplitude=3, make_ends_horizontal='both', end_with_zero=True)
    input_path_single_section = select_path_by_path_type(path_parameter, path_type)
    input_path_0 = double_the_path_nosort(input_path_single_section, do_plot=False)

    if forced_best_scale:
        # Plot flat path with color along the path
        fig, axs = plt.subplots()
        # plot_flat_path_with_color(input_path_0, input_path_single_section, axs)
        plot_flat_path_with_color(upsample_path(input_path_0, by_factor=trace_upsample_factor),
                                  upsample_path(input_path_single_section, by_factor=trace_upsample_factor),
                                  axs, linewidth=path_linewidth, alpha=path_alpha,
                                  plot_single_period=plot_single_period)
        # plot circle showing relative diameter of the sphere
        if forced_best_scale:
            circle_rad = 1 / forced_best_scale
        else:
            circle_rad = 1
        if plot_solution:
            circle1 = plt.Circle((circle_center[0], circle_center[1]),
                                 circle_rad, fill=False, linewidth=2, edgecolor='C1', alpha=circlealpha)
            axs.add_patch(circle1)
        ## Add colorbar for figures in the paper
        # cbar = fig.colorbar(line, ax=axs)
        # cbar.set_label('Distance along the period')
        fig.savefig(path_for_figs + f'/{path_type}_path.png', dpi=300)
        plt.show()

        plot_spherical_trace_with_color_along_the_trace(upsample_path(input_path_0, by_factor=trace_upsample_factor),
                                                        upsample_path(input_path_single_section, by_factor=trace_upsample_factor),
                                                        forced_best_scale)
        mlab.show()

    sweeped_scales, mismatch_angles = mismatches_for_all_scales(input_path_0, minscale=minscale, maxscale=maxscale, nframes=nframes)
    sweeped_scales, gb_areas = gb_areas_for_all_scales(input_path_single_section, minscale=minscale, maxscale=maxscale,
                                                       nframes=nframes)
    np.save(path_for_figs + '/sweeped_scales.npy', sweeped_scales)
    np.save(path_for_figs + '/gb_areas.npy', gb_areas)

    if not forced_best_scale:
        if range_for_searching_the_roots == 'auto':
            index_where_area_crosses_pi = np.argmax(np.abs(gb_areas) > np.pi)
            range_for_searching_the_roots = [sweeped_scales[index_where_area_crosses_pi-1],
                                             sweeped_scales[index_where_area_crosses_pi]]
            logging.debug(f'Auto range for roots: {range_for_searching_the_roots}')
        best_scale = minimize_mismatch_by_scaling(input_path_0, scale_range=range_for_searching_the_roots)
        logging.info(f'Best scale: {best_scale}')

        forced_best_scale = best_scale

    if not path_for_united_fig:
        fig, axarr = plt.subplots(2, 1, sharex=True, figsize=(7 * figsizefactor, 5 * figsizefactor))
        plot_mismatches_vs_scale(axarr[0], input_path_0, sweeped_scales, mismatch_angles,
                                 mark_one_scale=plot_solution,
                                 scale_to_mark=forced_best_scale)
        plot_gb_areas(axarr[1], sweeped_scales, gb_areas, mark_one_scale=plot_solution,
                      scale_to_mark=forced_best_scale)
        plt.tight_layout()
        fig.savefig(path_for_figs + '/angle-vs-scale.png', dpi=300)
        plt.show()

    else:
        if not best_scale:
            best_scale = 4
        fig = plt.figure(figsize=(8,8))
        fig.suptitle(fig_title + f'{path_parameter:.3f}')
        gs1 = GridSpec(3, 2, left=0.15, right=0.95, wspace=0.05, height_ratios=[2, 1, 1])
        ax_path = fig.add_subplot(gs1[0, 0])
        ax_trace = fig.add_subplot(gs1[0, 1])
        ax_angle = fig.add_subplot(gs1[1, :])
        ax_area = fig.add_subplot(gs1[2, :])

        # Plot flat path with color along the path
        plot_flat_path_with_color(upsample_path(input_path_0, by_factor=trace_upsample_factor),
                                  upsample_path(input_path_single_section, by_factor=trace_upsample_factor),
                                  ax_path)
        # plot circle showing relative diameter of the sphere
        circle_rad = 1 / best_scale
        if plot_solution:
            circle1 = plt.Circle((circle_center[0], circle_center[1]),
                                 circle_rad, fill=False, linewidth=2, edgecolor='C1', alpha=circlealpha)
            ax_path.add_patch(circle1)
        ## Add colorbar for figures in the paper
        # cbar = fig.colorbar(line, ax=axs)
        # cbar.set_label('Distance along the period')
        ax_path.set_aspect('equal', adjustable='datalim')
        ax_path.set_axis_off()

        plot_mismatches_vs_scale(ax_angle, input_path_0, sweeped_scales, mismatch_angles,
                                 mark_one_scale=plot_solution,
                                 scale_to_mark=best_scale)
        plot_gb_areas(ax_area, sweeped_scales, gb_areas, mark_one_scale=plot_solution,
                      scale_to_mark=best_scale)
        for ax in [ax_angle, ax_area]:
            ax.set_aspect('auto')
            ax.set_xlim(-1, maxscale)

        # Plot the 3D and show it in the matplotlib subplot
        mlab.options.offscreen = True
        mfig = plot_spherical_trace_with_color_along_the_trace(
            upsample_path(input_path_0, by_factor=trace_upsample_factor),
            upsample_path(input_path_single_section, by_factor=trace_upsample_factor), best_scale)
        # img = mlab.screenshot()
        f = mlab.gcf()
        f.scene._lift()
        cam = mfig.scene.camera
        cam.zoom(1.7)
        img = mlab.screenshot(figure=mfig, mode='rgba', antialiased=True)
        mlab.close()
        ax_trace.imshow(img)
        ax_trace.set_axis_off()

        fig.savefig(path_for_united_fig, dpi=300)
        # plt.show()

def animate_scale_sweep(path_type='brownian', path_for_frames='examples/brownian_path_1/figures/frames_scalesweep',
                        npoints=300, minscale=0.01, maxscale=26, circle_center=[0, 0],
                        circlealpha=1, plot_solution=True, range_for_searching_the_roots='auto', path_parameter=0.1,
                        nframes=10, indices_to_plot = [3, 7], spherical_trace_upsample_factor=100):
    """
    Tests existence of two-period trajectoid for a given path. It will also plot the mismatch angle and the
    Gauss-Bonnet area enclosed by the first period and great arc connecting its ends. Mismatch angles and areas will be
    evaluated for a number 'nframes' values scale factor in the interval between 'minscale' and 'maxscale'

    :param path_type: String. Chooses the input path to be tested. So far, can take values "brownian", "spiral", "narrow",
                        "sine", "brownian-smooth", "zigzag", "zigzag2".
    :param path_for_frames: String (path). Path to folder where the output plots will be saved.
    :param npoints: Integer. Number of values of scale factor at which enclosed areas and angular
                    mismatch will be evaluated.
    :param minscale: Float. Minimum value of scale factor to be plotted.
    :param maxscale: Float. Maximum value of scale factor to be plotted.
    :param circle_center: Tuple of two floats. Location of circle illustrating the size of the rolling sphere relative to the path.
    :param circlealpha: Float. Alpha (opacity) of circle illustrating the size of the rolling sphere relative to the path.
    :param plot_solution: Bool. Whether to plot the red dot (at 'forced_best_scale') on the plots of area and mismatch angle.
    :param range_for_searching_the_roots: String "auto" or a tuple of two float values. The best scale minimizing the mismatch angle will be automatically searched
                                          in this range by a root-finding algorithm. If set to "auto", the first crossing of pi by enclosed are
                                          will be used as right end of range, and the previous point as the lft end of range.
    :param do_plot: Boolean. Whether to plot some of the plots.
    :param path_parameter: Optional parameter that controls features of some types of the path.
    :param indices_to_plot: list of two integers. Indices of the path nodes that will be marked by blue and green
                            points on the flat plot and on the spherical trace
    :param spherical_trace_upsample_factor: Integer. The path will be upsampled by this factor before plotting its trace
                                            on the sphere.
    """
    # input_path_single_section = make_random_path(seed=1, amplitude=3, make_ends_horizontal='both', end_with_zero=True)
    input_path_single_section = select_path_by_path_type(path_parameter, path_type)
    input_path_0 = double_the_path_nosort(input_path_single_section, do_plot=False)

    sweeped_scales, mismatch_angles = mismatches_for_all_scales(input_path_0, minscale=minscale, maxscale=maxscale, nframes=npoints)
    sweeped_scales, gb_areas = gb_areas_for_all_scales(input_path_single_section, minscale=minscale, maxscale=maxscale,
                                                       nframes=npoints)

    if maxscale == 'best':
        if range_for_searching_the_roots == 'auto':
            index_where_area_crosses_pi = np.argmax(np.abs(gb_areas) > np.pi)
            range_for_searching_the_roots = [sweeped_scales[index_where_area_crosses_pi-1],
                                             sweeped_scales[index_where_area_crosses_pi]]
            print(f'Auto range for roots: {range_for_searching_the_roots}')
        best_scale = minimize_mismatch_by_scaling(input_path_0, scale_range=range_for_searching_the_roots)
        print(f'Best scale: {best_scale}')
        maxscale = best_scale

    list_of_scales_to_plot = np.linspace(0.01, maxscale, nframes)
    t0 = time.time()
    mlab.options.offscreen = True
    for frame_id, scale_to_plot in enumerate(list_of_scales_to_plot):
        print(f'Animation frame {frame_id}, scale {scale_to_plot:.3f}, ETA: {(time.time()-t0)/(frame_id + 1)*(nframes - frame_id)/60:.1f} min')
        path_for_united_fig = path_for_frames + f'/{frame_id:06d}.png'
        fig = plt.figure(figsize=(8,8))
        fig.suptitle(f'Scale factor: {scale_to_plot:.3f}')
        gs1 = GridSpec(3, 2, left=0.15, right=0.95, wspace=0.05, height_ratios=[2, 1, 1])
        ax_path = fig.add_subplot(gs1[0, 0])
        ax_trace = fig.add_subplot(gs1[0, 1])
        ax_angle = fig.add_subplot(gs1[1, :])
        ax_area = fig.add_subplot(gs1[2, :])

        # Plot flat path with color along the path
        plot_flat_path_with_color(input_path_0, input_path_single_section, ax_path)
        # plot certain points
        certain_point_colors = ['blue', 'lime']
                           # int(round(input_path_single_section.shape[0] * 0.25)),
                           # int(round(input_path_single_section.shape[0] * 0.75))]
        ax_path.scatter(input_path_single_section[indices_to_plot, 0], input_path_single_section[indices_to_plot, 1],
                        color=certain_point_colors, s=30)
        # plot circle showing relative diameter of the sphere
        circle_rad = 1 / scale_to_plot
        if plot_solution:
            circle1 = plt.Circle((circle_center[0], circle_center[1]),
                                 circle_rad, fill=False, linewidth=2, edgecolor='C1', alpha=circlealpha)
            ax_path.add_patch(circle1)
        ## Add colorbar for figures in the paper
        # cbar = fig.colorbar(line, ax=axs)
        # cbar.set_label('Distance along the period')
        ax_path.set_aspect('equal', adjustable='datalim')
        ax_path.set_axis_off()

        plot_mismatches_vs_scale(ax_angle, input_path_0, sweeped_scales, mismatch_angles,
                                 mark_one_scale=plot_solution,
                                 scale_to_mark=scale_to_plot)
        plot_gb_areas(ax_area, sweeped_scales, gb_areas, mark_one_scale=plot_solution,
                      scale_to_mark=scale_to_plot)
        for ax in [ax_angle, ax_area]:
            ax.set_aspect('auto')
            ax.set_xlim(-1, maxscale)

        # Plot the 3D and show it in the matplotlib subplot
        mfig = plot_spherical_trace_with_color_along_the_trace(
            upsample_path(input_path_0, by_factor=spherical_trace_upsample_factor),
            upsample_path(input_path_single_section, by_factor=spherical_trace_upsample_factor), scale_to_plot,
            sphere_opacity=0.6)
        # plot certain points
        point_radius = 0.1
        sphere_trace_single_section = trace_on_sphere(scale_to_plot * input_path_single_section, kx=1, ky=1)
        colors_of_trace_points = [(0, 0, 1), (0, 1, 0)]
        for i, index_to_plot in enumerate(indices_to_plot):
            point_here = sphere_trace_single_section[index_to_plot]
            mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=point_radius, color=colors_of_trace_points[i])
        sphere_trace_full = trace_on_sphere(scale_to_plot * input_path_0, kx=1, ky=1)
        for point_here in [sphere_trace_full[-1], sphere_trace_full[0]]:
            mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=point_radius, color=(0, 0, 0))

        # img = mlab.screenshot()
        f = mlab.gcf()
        f.scene._lift()
        cam = mfig.scene.camera
        cam.zoom(1.7)
        img = mlab.screenshot(figure=mfig, mode='rgba', antialiased=True)
        mlab.show()
        mlab.close(all=True)
        ax_trace.imshow(img)
        ax_trace.set_axis_off()

        fig.savefig(path_for_united_fig, dpi=300)
        # plt.show()
        plt.close(fig)

if __name__ == '__main__':
    # UNCOMMENT NEEDED PARTS BELOW TO TEST TWO-PERIOD TRAJECTOID EXISTENCE FOR VARIOUS PATHS

    # test_trajectoid_existence(path_type='brownian', path_for_figs='examples/brownian_path_1/figures',
    #                           forced_best_scale = 24.810359103416662,
    #                           nframes=300,
    #                           maxscale=26,
    #                           figsizefactor=0.85,
    #                           circle_center=[-1.7, -0.6],
    #                           circlealpha=1,
    #                           range_for_searching_the_roots=(24.7, 24.9)
    #                           )
    #
    # test_trajectoid_existence(path_type='spiral', path_for_figs='examples/spiral_path_1/figures',
    #                           forced_best_scale = 0.2588162519798698,
    #                           nframes=30,
    #                           maxscale=0.4,
    #                           figsizefactor=0.85,
    #                           circle_center=[0, 0],
    #                           circlealpha=0.5,
    #                           range_for_searching_the_roots=(0.25, 0.27)
    #                           )
    #
    # test_trajectoid_existence(path_type='narrow', path_for_figs='examples/narrow_1/figures',
    #                           forced_best_scale = 79.83304727542121, #79.35082181975892,
    #                           nframes=900, #900
    #                           maxscale=85,
    #                           figsizefactor=0.85,
    #                           circle_center=[0.75, 0],
    #                           circlealpha=1,
    #                           range_for_searching_the_roots=(79.6, 80)
    #                           )
    #
    # test_trajectoid_existence(path_type='sine', path_for_figs='examples/sine_existence_1/figures',
    #                           forced_best_scale = 14.906282559862566, #79.35082181975892,
    #                           nframes=2000, #900
    #                           maxscale=16,
    #                           figsizefactor=0.85,
    #                           circle_center=[0.75, 0],
    #                           circlealpha=1,
    #                           range_for_searching_the_roots=(14.902, 14.912)
    #                           )
    #
    # test_trajectoid_existence(path_type='brownian-smooth', path_for_figs='examples/brownian_path_2/figures',
    #                           forced_best_scale = 27.136794417872355,
    #                           nframes=300,
    #                           maxscale=28,
    #                           figsizefactor=0.85,
    #                           circle_center=[-1.7, -0.6],
    #                           circlealpha=1,
    #                           range_for_searching_the_roots=(27.0, 27.5)
    #                           )

    # for main-text figure
    test_trajectoid_existence(path_type='brownian-smooth', path_for_figs='examples/brownian_path_2/figures',
                              forced_best_scale = 27.136794417872355,
                              nframes=300,
                              maxscale=28,
                              figsizefactor=0.85,
                              circle_center=[-1.7, -0.6],
                              circlealpha=1,
                              range_for_searching_the_roots=(27.0, 27.5),
                              trace_upsample_factor=1
                              # path_linewidth=2,
                              # path_alpha=0.8,
                              # plot_single_period=True
                              )

    # test_trajectoid_existence(path_type='zigzag', path_for_figs='examples/zigzag_1/figures',
    #                           forced_best_scale = 4,
    #                           nframes=150,
    #                           maxscale=15,
    #                           figsizefactor=0.85,
    #                           circle_center=[0.6, -0.3],
    #                           circlealpha=1,
    #                           range_for_searching_the_roots=(3.9, 4.1)
    #                           )

    # test_trajectoid_existence(path_type='zigzag_2', path_for_figs='examples/zigzag_2/figures',
    #                           forced_best_scale = 43,
    #                           nframes=4000,
    #                           maxscale=250,
    #                           figsizefactor=0.85,
    #                           circle_center=[-1.7, -0.6],
    #                           circlealpha=1,
    #                           plot_solution=False,
    #                           range_for_searching_the_roots=(63, 64)
    #                           )

    # test_trajectoid_existence(path_type='zigzag_tapered', path_for_figs='examples/zigzag_tapered/figures',
    #                           forced_best_scale = 10.805702204321273,# 4.240589475501186,
    #                           nframes=500,
    #                           maxscale=11.23,
    #                           figsizefactor=0.85,
    #                           circle_center=[-1.7, -0.6],
    #                           circlealpha=1,
    #                           plot_solution=True,
    #                           range_for_searching_the_roots=(10.6, 10.9),
    #                           path_parameter=0.3
    #                           )

    # test_trajectoid_existence(path_type='zigzag_tapered', path_for_figs='examples/zigzag_tapered/figures',
    #                           forced_best_scale=26.61504628916999, #46.777252049239166, #10.805702204321273,  # 4.240589475501186,
    #                           nframes=2000,
    #                           maxscale=28,#70,
    #                           figsizefactor=0.85,
    #                           circle_center=[1.2, -0.8],
    #                           circlealpha=1,
    #                           plot_solution=True,
    #                           range_for_searching_the_roots='auto',  #(10.6, 10.9),
    #                           path_parameter=0.1
    #                           )

    # test_trajectoid_existence(path_type='zigzag_kinked', path_for_figs='examples/zigzag_kinked/figures',
    #                           forced_best_scale=19.6017, #46.777252049239166, #10.805702204321273,  # 4.240589475501186,
    #                           nframes=2000,
    #                           maxscale=40,#70,
    #                           figsizefactor=0.85,
    #                           circle_center=[-1.7, -0.6],
    #                           circlealpha=1,
    #                           plot_solution=True,
    #                           range_for_searching_the_roots='auto',  #(10.6, 10.9),
    #                           path_parameter=0.1
    #                           )

    # test_trajectoid_existence(path_type='zigzag_kinked_asymmetric', path_for_figs='examples/zigzag_kinked_asymmetric/figures',
    #                           forced_best_scale=21.012723684811053, #46.777252049239166, #10.805702204321273,  # 4.240589475501186,
    #                           nframes=2000,
    #                           maxscale=28,#70,
    #                           figsizefactor=0.85,
    #                           circle_center=[1.05, -0.95],
    #                           circlealpha=1,
    #                           plot_solution=True,
    #                           range_for_searching_the_roots='auto',  #(10.6, 10.9),
    #                           path_parameter=0.370
    #                           )

    # test_trajectoid_existence(path_type='zigzag_tapered', path_for_figs='examples/zigzag_tapered/figures_continuity',
    #                           forced_best_scale=11, #46.777252049239166, #10.805702204321273,  # 4.240589475501186,
    #                           nframes=4000,
    #                           maxscale=15,#70,
    #                           figsizefactor=0.85,
    #                           circle_center=[1.2, -0.8],
    #                           circlealpha=1,
    #                           plot_solution=True,
    #                           range_for_searching_the_roots='auto',  #(10.6, 10.9),
    #                           path_parameter=0.246# 0.2464
    #                           )

    # # Animating the path parameter sweep
    # power_here = 3
    # for frame_id, path_parameter in enumerate(np.linspace((0.05)**(1/power_here), (0.3)**(1/power_here), 80)**power_here):
    #     test_trajectoid_existence(path_type='zigzag_tapered', path_for_figs='examples/zigzag_tapered/figures',
    #                               forced_best_scale=False, #46.777252049239166, #10.805702204321273,  # 4.240589475501186,
    #                               nframes=2000,
    #                               maxscale=60,#70,
    #                               figsizefactor=0.85,
    #                               circle_center=[1.3, -0.8],
    #                               circlealpha=1,
    #                               plot_solution=True,
    #                               range_for_searching_the_roots='auto',  #(10.6, 10.9),
    #                               path_parameter=path_parameter,
    #                               path_for_united_fig=f'examples/zigzag_tapered/figures/frames_paramsweep/{frame_id:06d}.png',
    #                               fig_title='Taper fraction: '
    #                               )

    # power_here=1
    # for frame_id, path_parameter in enumerate(np.linspace((0.01)**(1/power_here), (0.4)**(1/power_here), 80)**power_here):
    #     test_trajectoid_existence(path_type='zigzag_kinked', path_for_figs='examples/zigzag_kinked/figures',
    #                               forced_best_scale=False, #46.777252049239166, #10.805702204321273,  # 4.240589475501186,
    #                               nframes=2000,
    #                               maxscale=112,#70,
    #                               figsizefactor=0.85,
    #                               circle_center=[1.3, -0.8],
    #                               circlealpha=1,
    #                               plot_solution=False,
    #                               range_for_searching_the_roots=(1, 1.1),
    #                               path_parameter=path_parameter,
    #                               path_for_united_fig=f'examples/zigzag_kinked/figures/frames_paramsweep/{frame_id:06d}.png',
    #                               fig_title='Declination of first kink arm, rad: '
    #                               )

    # # Animating the path parameter sweep
    # power_here = 3
    # for frame_id, path_parameter in enumerate(np.linspace((0.08)**(1/power_here), (0.6)**(1/power_here), 80)**power_here):
    #     test_trajectoid_existence(path_type='zigzag_kinked_asymmetric', path_for_figs='examples/zigzag_kinked_asymmetric/figures',
    #                               forced_best_scale=False, #46.777252049239166, #10.805702204321273,  # 4.240589475501186,
    #                               nframes=6000,
    #                               maxscale=120,#70,
    #                               figsizefactor=0.85,
    #                               circle_center=[1.3, -0.8],
    #                               circlealpha=1,
    #                               plot_solution=True,
    #                               range_for_searching_the_roots='auto',  #(10.6, 10.9),
    #                               path_parameter=path_parameter,
    #                               path_for_united_fig=f'examples/zigzag_kinked_asymmetric/figures/frames_paramsweep/{frame_id:06d}.png',
    #                               fig_title='Asymmetry: ',
    #                               trace_upsample_factor=300
    #                               )

    # # Animating the path parameter sweep
    # power_here = 3
    # for frame_id, path_parameter in enumerate(np.linspace((0.03)**(1/power_here), (0.25)**(1/power_here), 80)**power_here):
    #     test_trajectoid_existence(path_type='zigzag_smoothed', path_for_figs='examples/zigzag_smoothed/figures',
    #                               forced_best_scale=False, #46.777252049239166, #10.805702204321273,  # 4.240589475501186,
    #                               nframes=800,
    #                               maxscale=45,#70,
    #                               figsizefactor=0.85,
    #                               circle_center=[1.202, -0.95],
    #                               circlealpha=1,
    #                               plot_solution=True,
    #                               range_for_searching_the_roots='auto',  #(10.6, 10.9),
    #                               path_parameter=path_parameter,
    #                               path_for_united_fig=f'examples/zigzag_smoothed/figures/frames_paramsweep/{frame_id:06d}.png',
    #                               fig_title='Corner curvature radius: '
    #                               )

    ### Scale sweep animations
    # animate_scale_sweep(path_type='zigzag_tapered', path_for_frames='examples/zigzag_tapered/figures/frames_scalesweep',
    #                     npoints=2000, maxscale='best', figsizefactor=0.85, circle_center=[1.3, -0.8], circlealpha=1,
    #                     plot_solution=True, range_for_searching_the_roots='auto', path_parameter=0.1,
    #                     nframes=200, indices_to_plot = [3, 7])

    # animate_scale_sweep(path_type='zigzag', path_for_frames='examples/zigzag_1/figures/frames_scalesweep',
    #                     npoints=700, maxscale=10, figsizefactor=0.85, circle_center=[1.3, -0.8], circlealpha=1,
    #                     plot_solution=True, range_for_searching_the_roots='auto', path_parameter=0.1,
    #                     nframes=200, indices_to_plot = [7, 21], spherical_trace_upsample_factor=10)