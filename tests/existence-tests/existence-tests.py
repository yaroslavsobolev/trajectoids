import matplotlib.pyplot as plt
import numpy as np
import mayavi
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from compute_trajectoid import *

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

def mismatches_for_all_scales(input_path, minscale=0.01, maxscale=2, nframes = 100):
    mismatch_angles = []
    sweeped_scales = np.linspace(minscale, maxscale, nframes)
    for frame_id, scale in enumerate(sweeped_scales):
        print(f'Computing mismatch for scale {scale}')
        input_path_single_section = input_path * scale
        mismatch_angles.append(mismatch_angle_for_path(input_path_single_section, recursive=False, use_cache=False))
    return sweeped_scales, np.array(mismatch_angles)

def gb_areas_for_all_scales(input_path, minscale=0.01, maxscale=2, nframes = 100):
    '''This function takes into account the possibly changing rotation index of the spherical trace.'''
    gauss_bonnet_areas = []
    sweeped_scales = np.linspace(minscale, maxscale, nframes)
    for frame_id, scale in enumerate(sweeped_scales):
        print(f'Computing GB_area for scale {scale}')
        input_path_scaled = input_path * scale
        gauss_bonnet_areas.append(get_gb_area(input_path_scaled))

    gb_areas = np.array(gauss_bonnet_areas)
    # compensation for integer number of 2*pi due to rotation index of the curve
    gb_area_zero = round(gb_areas[0]/np.pi) * np.pi
    gb_areas -= gb_area_zero

    # correct for changes of rotation index I upon scaling. Upon +1 or -1 change of I, the integral of geodesic curvature
    # (total change of direction) increments or decrements by 2*pi
    additional_rotation_indices = np.zeros_like(gb_areas)
    additional_rotation_index_here = 0
    threshold_for_ind = 2*np.pi*0.75
    for i in range(1, gb_areas.shape[0]):
        diff_here = gb_areas[i] - gb_areas[i-1]
        if np.abs(diff_here) > threshold_for_ind:
            additional_rotation_index_here += np.round(diff_here/(2*np.pi))
        additional_rotation_indices[i] = additional_rotation_index_here
    gb_areas -= 2 * np.pi * additional_rotation_indices

    # # Plot additional rotation indices for debugging
    # plt.plot(sweeped_scales, additional_rotation_indices, 'o-')
    # plt.show()

    return sweeped_scales, gb_areas

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

def upsample_path(input_path, by_factor=10, kind='linear'):
    old_indices = np.arange(input_path.shape[0])
    max_index = input_path.shape[0]-1
    new_indices = np.arange(0, max_index, 1/by_factor)
    new_indices = np.append(new_indices, max_index)
    new_xs = interp1d(old_indices, input_path[:, 0], kind=kind)(new_indices)
    new_ys = interp1d(old_indices, input_path[:, 1], kind=kind)(new_indices)
    return np.stack((new_xs, new_ys)).T

def add_interval(startpoint, angle, length, Ns = 30):
    xs = startpoint[0] + np.linspace(0, length * np.cos(angle), Ns)
    ys = startpoint[1] + np.linspace(0, length * np.sin(angle), Ns)
    return np.stack((xs, ys)).T

def make_zigzag(a):
    angles = [-np.pi*3/8, np.pi*3/8]#, -np.pi/4, np.pi/4, -np.pi/4, np.pi/4]
    lengths = [a, np.pi/2]#, np.pi/2, a, np.pi/2, np.pi/2]
    input_path = np.array([[0, 0]])
    tips = [[0, 0]]
    for i, angle in enumerate(angles):
        startpoint = input_path[-1,:]
        new_section = add_interval(startpoint, angle, lengths[i])
        input_path = np.concatenate((input_path, new_section[1:]), axis=0)
        tips.append(new_section[-1])
    return input_path, np.array(tips)

def make_zigzag2(a):
    angles = [-np.pi*3/8, np.pi*3/8]#, -np.pi/4, np.pi/4, -np.pi/4, np.pi/4]
    lengths = [a, np.pi/2]#, np.pi/2, a, np.pi/2, np.pi/2]
    input_path = np.array([[0, 0]])
    tips = [[0, 0]]
    for i, angle in enumerate(angles):
        startpoint = input_path[-1,:]
        new_section = add_interval(startpoint, angle, lengths[i], Ns=10)
        input_path = np.concatenate((input_path, new_section[1:]), axis=0)
        tips.append(new_section[-1])
    input_path[:,1] = input_path[:,1] + 0.15*np.cos(input_path[:,0] / input_path[-1,0] * 4 * np.pi)
    input_path[:,1] = input_path[:,1] - input_path[0,1]
    return input_path, np.array(tips)

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

def length_along_the_path(input_path_0):
    x = input_path_0[:, 0]
    y = input_path_0[:, 1]
    length_from_start_to_here = np.cumsum( np.sqrt((np.diff(x)**2 + np.diff(y)**2)) )
    length_from_start_to_here = np.insert(length_from_start_to_here, 0, 0)
    length_from_start_to_here = np.remainder(length_from_start_to_here, np.max(length_from_start_to_here)/2)
    length_from_start_to_here /= np.max(length_from_start_to_here)
    return length_from_start_to_here

def plot_flat_path_with_color(input_path_0, input_path_single_section, axs):
    '''plotting with color along the line'''
    length_from_start_to_here = length_along_the_path(input_path_0)
    x = input_path_0[:, 0]
    y = input_path_0[:, 1]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Coloring the curve
    norm = plt.Normalize(length_from_start_to_here.min(), length_from_start_to_here.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(length_from_start_to_here)
    lc.set_linewidth(1)
    line = axs.add_collection(lc)

    #black dots at middle and ends of the path
    for point in [input_path_single_section[0], input_path_single_section[-1], input_path_0[-1]]:
        axs.scatter(point[0], point[1], color='black', s=10)

    plt.axis('equal')

def plot_spherical_trace_with_color_along_the_trace(input_path_0, input_path_single_section, preliminary_best_scale,
                                                    verbose=False):
    length_from_start_to_here = length_along_the_path(input_path_0)
    plotting_upsample_factor = 1
    t0 = time.time()
    sphere_trace = trace_on_sphere(upsample_path(preliminary_best_scale * input_path_0, by_factor=plotting_upsample_factor),
                                   kx=1, ky=1)
    sphere_trace_single_section = trace_on_sphere(upsample_path(input_path_single_section * preliminary_best_scale,
                                                                by_factor=plotting_upsample_factor),
                                                  kx=1, ky=1)
    if verbose:
        print(f'Seconds passed: {time.time() - t0:.3f}')
        print('Mlab plot begins...')
    core_radius = 1
    last_index = sphere_trace.shape[0] // 2
    mlab.figure(size=(1024, 768), \
                bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    tube_radius = 0.01
    plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
    mlab.plot3d(sphere_trace[:, 0],
                sphere_trace[:, 1],
                sphere_trace[:, 2],
                length_from_start_to_here, colormap='viridis',
                tube_radius=tube_radius)

def plot_mismatches_vs_scale(ax, input_path_0, sweeped_scales, mismatch_angles, plot_solution, solution_scale):
    ## This code injects the sampled point at the solution_scale. Otherwise the root can be not at one of sampled points
    if plot_solution:
        ii = np.searchsorted(sweeped_scales, solution_scale)
        sweeped_scales = np.insert(sweeped_scales, ii, solution_scale)
        # value_at_scale = 0
        value_at_scale = mismatch_angle_for_path(input_path_0 * solution_scale, recursive=False, use_cache=False)
        mismatch_angles = np.insert(mismatch_angles, ii, value_at_scale)
    ax.plot(sweeped_scales, np.abs(mismatch_angles)/np.pi*180)
    # ax.plot(sweeped_scales, mismatch_angles / np.pi * 180)
    ax.axhline(y=0, color='black')
    if plot_solution:
        value_at_scale = interp1d(sweeped_scales, mismatch_angles)(solution_scale)
        ax.scatter([solution_scale], [value_at_scale], s=20, color='red')
    ax.set_yticks(np.arange(0, 181, 20))
    ax.set_ylim(-5, 181)
    ax.set_ylabel('Mismatch angle (deg.)\nbetween initial and\nfinal orientations after\npassing two periods')

def plot_gb_ares(ax, sweeped_scales, gb_areas, plot_solution, solution_scale):

    ## This code injects the sampled point at the solution_scale. Otherwise the root can be not at one of sampled points
    # ii = np.searchsorted(sweeped_scales, solution_scale)
    # sweeped_scales = np.insert(sweeped_scales, ii, solution_scale)
    # mismatch_angles = np.insert(mismatch_angles, ii, np.pi * np.sign(interp1d(sweeped_scales, gb_areas)(solution_scale)))

    ax.plot(sweeped_scales, gb_areas)
    ax.axhline(y=np.pi, color='black', alpha=0.5)
    ax.axhline(y=0, color='black', alpha=0.3)
    ax.axhline(y=-1 * np.pi, color='black', alpha=0.5)
    if plot_solution:
        # ax.scatter([solution_scale], [np.pi * np.sign(interp1d(sweeped_scales, gb_areas)(solution_scale))], s=20, color='red')
        value_at_scale = interp1d(sweeped_scales, gb_areas)(solution_scale)
        ax.scatter([solution_scale], [value_at_scale], s=20, color='red')
    ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
    ax.set_yticklabels(['-2π', '-π', '0', 'π', '2π'])
    ax.set_ylim(-np.pi * 2 * 1.01, np.pi * 2 * 1.01)
    ax.set_ylabel('Spherical area $S(\sigma)$\n enclosed by the \nfirst period\'s trace')
    ax.set_xlabel('Path\'s scale factor $\sigma$ for fixed ball radius $r=1$\n(or $1/r$ for fixed $\sigma=1$)')

def test_trajectoid_existence(path_type='brownian', path_for_figs='examples/brownian_path_1/figures',
                              preliminary_best_scale = 24.810359103416662,
                              nframes=300,
                              minscale=0.01,
                              maxscale = 26,
                              figsizefactor = 0.85,
                              circle_center=[-1.7, -0.6],
                              circlealpha=1, plot_solution=True,
                              range_for_searching_the_roots=(0, 4),
                              do_plot = True):
    """
    Tests existence of two-period trajectoid for a given path. It will also plot the mismatch angle and the
    Gauss-Bonnet area enclosed by the first period and great arc connecting its ends. Mismatch angles and areas will be
    evaluated for a number 'nframes' values scale factor in the interval between 'minscale' and 'maxscale'

    :param path_type: String. Chooses the input path to be tested. So far, can take values "brownian", "spiral", "narrow",
                        "sine", "brownian-smooth", "zigzag", "zigzag2".
    :param path_for_figs: String (path). Path to folder where the output plots will be saved.
    :param preliminary_best_scale: Float. The value of scale factor for which the first colored spherical trace will be plotted.
                               It must be between the minscale and maxscale.
                               Red dot on the plots will also be plotted at this scale factor. Basically, you should use
                               the true "best scale" that you found by true minimization on the previous launch of this script.
    :param nframes: Integer. Number of values of scale factor at which enclosed areas and angular
                    mismatch will be evaluated.
    :param minscale: Float. Minimum value of scale factor to be plotted.
    :param maxscale: Float. Maximum value of scale factor to be plotted.
    :param figsizefactor: Scales the output images. This is only useful for making proper font sizes in the final plots
                          in the paper
    :param circle_center: Tuple of two floats. Location of circle illustrating the size of the rolling sphere relative to the path.
    :param circlealpha: Float. Alpha (opacity) of circle illustrating the size of the rolling sphere relative to the path.
    :param plot_solution: Bool. Whether to plot the red dot (at 'preliminary_best_scale') on the plots of area and mismatch angle.
    :param range_for_searching_the_roots: Tuple of two float values. The best scale minimizing the mismatch angle will be automatically searched
                                          in this range by a root-finding algorithm.
    :param do_plot: Boolean. Whether to plot some of the plots.
    """
    # input_path_single_section = make_random_path(seed=1, amplitude=3, make_ends_horizontal='both', end_with_zero=True)
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

    input_path_0 = double_the_path_nosort(input_path_single_section, do_plot=False)

    # Plot flat path with color along the path
    fig, axs = plt.subplots()
    plot_flat_path_with_color(input_path_0, input_path_single_section, axs)
    # plot circle showing relative diameter of the sphere
    if plot_solution:
        circle1 = plt.Circle((circle_center[0], circle_center[1]),
                             1/preliminary_best_scale, fill=False, linewidth=2, edgecolor='C1', alpha=circlealpha)
        axs.add_patch(circle1)
    ## Add colorbar for figures in the paper
    # cbar = fig.colorbar(line, ax=axs)
    # cbar.set_label('Distance along the period')
    fig.savefig(path_for_figs + f'/{path_type}_path.png', dpi=300)
    plt.show()

    plot_spherical_trace_with_color_along_the_trace(input_path_0, input_path_single_section, preliminary_best_scale)
    mlab.show()

    sweeped_scales, mismatch_angles = mismatches_for_all_scales(input_path_0, minscale=minscale, maxscale=maxscale, nframes=nframes)
    sweeped_scales, gb_areas = gb_areas_for_all_scales(input_path_single_section, minscale=minscale, maxscale=maxscale, nframes=nframes)

    fig, axarr = plt.subplots(2, 1, sharex=True, figsize=(7*figsizefactor, 5*figsizefactor))
    plot_mismatches_vs_scale(axarr[0], input_path_0, sweeped_scales, mismatch_angles, plot_solution, preliminary_best_scale)
    plot_gb_ares(axarr[1], sweeped_scales, gb_areas, plot_solution, preliminary_best_scale)
    plt.tight_layout()
    fig.savefig(path_for_figs + '/angle-vs-scale.png', dpi=300)
    plt.show()

    best_scale = minimize_mismatch_by_scaling(input_path_0, scale_range=range_for_searching_the_roots)
    print(f'Best scale: {best_scale}')

    # # Plot the spherical trace for the true solution (best_scale found by the root finding)
    # input_path = best_scale * input_path_0
    # sphere_trace = trace_on_sphere(upsample_path(input_path, by_factor=10),
    #                                kx=1, ky=1)
    # sphere_trace_single_section = trace_on_sphere(upsample_path(input_path_single_section * best_scale, by_factor=10),
    #                                               kx=1, ky=1)
    # if do_plot:
    #     core_radius = 1
    #     mlab.figure(size=(1024, 768), \
    #                 bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    #     tube_radius = 0.01
    #     plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
    #     l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(0, 0, 1),
    #                     tube_radius=tube_radius)
    #     l = mlab.plot3d(sphere_trace_single_section[:, 0],
    #                     sphere_trace_single_section[:, 1],
    #                     sphere_trace_single_section[:, 2], color=(0, 1, 0),
    #                     tube_radius=tube_radius)
    #     mlab.show()

if __name__ == '__main__':
    # UNCOMMENT NEEDED PARTS BELOW TO TEST TWO-PERIOD TRAJECTOID EXISTENCE FOR VARIOUS PATHS

    # test_trajectoid_existence(path_type='brownian', path_for_figs='examples/brownian_path_1/figures',
    #                           preliminary_best_scale = 24.810359103416662,
    #                           nframes=300,
    #                           maxscale=26,
    #                           figsizefactor=0.85,
    #                           circle_center=[-1.7, -0.6],
    #                           circlealpha=1,
    #                           range_for_searching_the_roots=(24.7, 24.9)
    #                           )
    #
    # test_trajectoid_existence(path_type='spiral', path_for_figs='examples/spiral_path_1/figures',
    #                           preliminary_best_scale = 0.2588162519798698,
    #                           nframes=30,
    #                           maxscale=0.4,
    #                           figsizefactor=0.85,
    #                           circle_center=[0, 0],
    #                           circlealpha=0.5,
    #                           range_for_searching_the_roots=(0.25, 0.27)
    #                           )
    #
    # test_trajectoid_existence(path_type='narrow', path_for_figs='examples/narrow_1/figures',
    #                           preliminary_best_scale = 79.83304727542121, #79.35082181975892,
    #                           nframes=900, #900
    #                           maxscale=85,
    #                           figsizefactor=0.85,
    #                           circle_center=[0.75, 0],
    #                           circlealpha=1,
    #                           range_for_searching_the_roots=(79.6, 80)
    #                           )
    #
    # test_trajectoid_existence(path_type='sine', path_for_figs='examples/sine_existence_1/figures',
    #                           preliminary_best_scale = 14.906282559862566, #79.35082181975892,
    #                           nframes=1200, #900
    #                           maxscale=16,
    #                           figsizefactor=0.85,
    #                           circle_center=[0.75, 0],
    #                           circlealpha=1,
    #                           range_for_searching_the_roots=(14.902, 14.912)
    #                           )
    #
    # test_trajectoid_existence(path_type='brownian-smooth', path_for_figs='examples/brownian_path_2/figures',
    #                           preliminary_best_scale = 27.136794417872355,
    #                           nframes=300,
    #                           maxscale=28,
    #                           figsizefactor=0.85,
    #                           circle_center=[-1.7, -0.6],
    #                           circlealpha=1,
    #                           range_for_searching_the_roots=(27.0, 27.5)
    #                           )

    test_trajectoid_existence(path_type='zigzag', path_for_figs='examples/zigzag_1/figures',
                              preliminary_best_scale = 4,
                              nframes=150,
                              maxscale=15,
                              figsizefactor=0.85,
                              circle_center=[0.6, -0.3],
                              circlealpha=1,
                              range_for_searching_the_roots=(3.9, 4.1)
                              )

    # test_trajectoid_existence(path_type='zigzag_2', path_for_figs='examples/zigzag_2/figures',
    #                           preliminary_best_scale = 43,
    #                           nframes=4000,
    #                           maxscale=250,
    #                           figsizefactor=0.85,
    #                           circle_center=[-1.7, -0.6],
    #                           circlealpha=1,
    #                           plot_solution=False,
    #                           range_for_searching_the_roots=(63, 64)
    #                           )