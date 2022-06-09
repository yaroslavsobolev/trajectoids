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

def make_narrow(npoints, shift=0.05, noise_amplitude=0, end_with_zero=True, seed=0):
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

def upsample_path(input_path, by_factor=10):
    old_indices = np.arange(input_path.shape[0])
    max_index = input_path.shape[0]-1
    new_indices = np.arange(0, max_index, 1/by_factor)
    new_indices = np.append(new_indices, max_index)
    new_xs = interp1d(old_indices, input_path[:, 0])(new_indices)
    new_ys = interp1d(old_indices, input_path[:, 1])(new_indices)
    return np.stack((new_xs, new_ys)).T

def test_trajectoid_existence(path_type='brownian', path_for_figs='examples/brownian_path_1/figures',
                              best_scale = 24.810359103416662,
                              nframes=300,
                              maxscale = 26,
                              figsizefactor = 0.85,
                              circle_center=[-1.7, -0.6],
                              circlealpha=1
):
    target_folder='tests/existence-tests/path_folder'

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
        input_path_single_section = make_narrow(npoints=150)
        # input_path_single_section = upsample_path(input_path_single_section, by_factor=20)

    input_path_0 = double_the_path_nosort(input_path_single_section, do_plot=False)
    # plotting with color along the line
    x = input_path_0[:, 0]
    y = input_path_0[:, 1]
    length_from_start_to_here = np.cumsum( np.sqrt((np.diff(x)**2 + np.diff(y)**2)) )
    length_from_start_to_here = np.insert(length_from_start_to_here, 0, 0)
    length_from_start_to_here = np.remainder(length_from_start_to_here, np.max(length_from_start_to_here)/2)
    length_from_start_to_here /= np.max(length_from_start_to_here)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, axs = plt.subplots()
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(length_from_start_to_here.min(), length_from_start_to_here.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(length_from_start_to_here)
    lc.set_linewidth(1)
    line = axs.add_collection(lc)
    # fig.colorbar(line, ax=axs[0])
    for point in [input_path_single_section[0], input_path_single_section[-1], input_path_0[-1]]:
        plt.scatter(point[0], point[1], color='black', s=10)
    # plt.plot(input_path_single_section[:, 0], input_path_single_section[:, 1], '-', color=colors)  # , label='Asymmetric')
    # plt.plot(input_path_0[:, 0], input_path_0[:, 1], '-', color='C0')
    # plot circle showing relative diameter of the sphere
    circle1 = plt.Circle((circle_center[0], circle_center[1]), 1/best_scale, fill=False, linewidth=2, edgecolor='C1', alpha=circlealpha)
    axs.add_patch(circle1)
    plt.axis('equal')
    # cbar = fig.colorbar(line, ax=axs)
    # cbar.set_label('Distance along the period')
    fig.savefig(path_for_figs + f'/{path_type}_path.png', dpi=300)
    plt.show()

    # upsampled_path = upsample_path(input_path_0)
    # plt.plot(upsampled_path[:, 0], upsampled_path[:, 1], 'o-', alpha=0.5)
    # plt.show()
    #
    plotting_upsample_factor = 1
    t0 = time.time()
    sphere_trace = trace_on_sphere(upsample_path(best_scale * input_path_0, by_factor=plotting_upsample_factor),
                                   kx=1, ky=1)
    sphere_trace_single_section = trace_on_sphere(upsample_path(input_path_single_section * best_scale,
                                                                by_factor=plotting_upsample_factor),
                                                  kx=1, ky=1)
    print(f'Seconds passed: {time.time() - t0:.3f}')
    print('Mlab plot begins...')
    core_radius = 1
    last_index = sphere_trace.shape[0] // 2
    mlab.figure(size=(1024, 768), \
                bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    tube_radius = 0.01
    plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
    # l = mlab.plot3d(sphere_trace[last_index:, 0],
    #                 sphere_trace[last_index:, 1],
    #                 sphere_trace[last_index:, 2], color=(0, 0, 1),
    #                 tube_radius=tube_radius)
    # l = mlab.plot3d(sphere_trace_single_section[:, 0],
    #                 sphere_trace_single_section[:, 1],
    #                 sphere_trace_single_section[:, 2], color=(0, 1, 0),
    #                 tube_radius=tube_radius)
    l = mlab.plot3d(sphere_trace[:, 0],
                    sphere_trace[:, 1],
                    sphere_trace[:, 2],
                    length_from_start_to_here, colormap='viridis',
                    tube_radius=tube_radius)

    mlab.show()
    minscale=0.01
    fig, axarr = plt.subplots(2, 1, sharex=True, figsize=(7*figsizefactor, 5*figsizefactor))
    # sweeped_scales, mismatch_angles = mismatches_for_all_scales(input_path_single_section, minscale=24.5, maxscale=25.1)
    # sweeped_scales, mismatch_angles = mismatches_for_all_scales(input_path_0, minscale=24, maxscale=25)
    sweeped_scales, mismatch_angles = mismatches_for_all_scales(input_path_0, minscale=minscale, maxscale=maxscale, nframes=nframes)
    solution_scale = best_scale
    # ii = np.searchsorted(sweeped_scales, solution_scale)
    # sweeped_scales = np.insert(sweeped_scales, ii, solution_scale)
    # mismatch_angles = np.insert(mismatch_angles, ii, 0)
    ax = axarr[0]
    ax.plot(sweeped_scales, np.abs(mismatch_angles)/np.pi*180)
    # ax.plot(sweeped_scales, mismatch_angles / np.pi * 180)
    ax.axhline(y=0, color='black')
    ax.scatter([solution_scale], [0], s=20, color='red')
    ax.set_yticks(np.arange(0, 181, 20))
    ax.set_ylim(-5, 181)
    ax.set_ylabel('Mismatch angle\nbetween initial and\nfinal orientations, deg')
    # ax.set_xlabel('Path\'s scale factor S for fixed sphere radius\n(or inverse sphere radius for fixed scale factor)')
    # plt.plot(sweeped_scales, mismatch_angles)

    ax = axarr[1]
    sweeped_scales, mismatch_angles = mismatches_for_all_scales(input_path_single_section, minscale=minscale, maxscale=maxscale, nframes=nframes)
    solution_scale = best_scale
    # ii = np.searchsorted(sweeped_scales, solution_scale)
    # sweeped_scales = np.insert(sweeped_scales, ii, solution_scale)
    # mismatch_angles = np.insert(mismatch_angles, ii, np.pi)
    ax.plot(sweeped_scales, np.abs(mismatch_angles))
    # ax.axhline(y=0, color='black')
    ax.scatter([solution_scale], [np.pi], s=20, color='red')
    ax.set_yticks([0, np.pi/2, np.pi])
    ax.set_yticklabels(['0', 'π/2', 'π'])
    ax.set_ylim(-0.001, np.pi*1.01)
    ax.set_ylabel('Spherical area enclosed\n by the first period\'s trace')
    ax.set_xlabel('Path\'s scale factor S for fixed sphere radius\n(or inverse sphere radius for fixed scale factor)')
    plt.tight_layout()
    fig.savefig(path_for_figs + '/angle-vs-scale.png', dpi=300)
    plt.show()


    do_plot = True

    if path_type == 'brownian':
        best_scale = minimize_mismatch_by_scaling(input_path_0, scale_range=(24.7, 24.9))
    elif path_type == 'spiral':
        best_scale = minimize_mismatch_by_scaling(input_path_0, scale_range=(0.25, 0.27))
    elif path_type == 'narrow':
        best_scale = minimize_mismatch_by_scaling(input_path_0, scale_range=(79.3, 79.4))
    print(f'Best scale: {best_scale}')
    # Minimized mismatch angle = -2.6439127092433114e-05
    # Best scale: 0.6387022944333781

    input_path = best_scale * input_path_0
    plotting_upsample_factor = 1
    sphere_trace = trace_on_sphere(upsample_path(input_path, by_factor=10),
                                   kx=1, ky=1)
    sphere_trace_single_section = trace_on_sphere(upsample_path(input_path_single_section * best_scale, by_factor=10),
                                                  kx=1, ky=1)
    if do_plot:
        core_radius = 1
        mlab.figure(size=(1024, 768), \
                    bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
        tube_radius = 0.01
        plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
        l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(0, 0, 1),
                        tube_radius=tube_radius)
        l = mlab.plot3d(sphere_trace_single_section[:, 0],
                        sphere_trace_single_section[:, 1],
                        sphere_trace_single_section[:, 2], color=(0, 1, 0),
                        tube_radius=tube_radius)
        mlab.show()

if __name__ == '__main__':
    # test_trajectoid_existence(path_type='brownian', path_for_figs='examples/brownian_path_1/figures',
    #                           best_scale = 24.810359103416662,
    #                           nframes=300,
    #                           maxscale=26,
    #                           figsizefactor=0.85,
    #                           circle_center=[-1.7, -0.6],
    #                           circlealpha=1
    #                           )

    # test_trajectoid_existence(path_type='spiral', path_for_figs='examples/spiral_path_1/figures',
    #                           best_scale = 0.2588162519798698,
    #                           nframes=300,
    #                           maxscale=0.4,
    #                           figsizefactor=0.85,
    #                           circle_center=[0, 0],
    #                           circlealpha=0.5
    #                           )


    test_trajectoid_existence(path_type='narrow', path_for_figs='examples/narrow_1/figures',
                              best_scale = 79.35082181975892,
                              nframes=900,
                              maxscale=85,
                              figsizefactor=0.85,
                              circle_center=[0.75, 0],
                              circlealpha=1
                              )
# sweeped_scales, mismatch_angles = mismatches_for_all_scales()
# plt.plot(sweeped_scales, np.abs(mismatch_angles))
# plt.show()

# make_animation_of_rotational_symmetry()

# make_orbit_animation(folder_for_frames='examples/penannular_1/orbit_frames', elevation=60)

# for point_here in [sphere_trace_single_section[-1], sphe]:
#     mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.05, color=(1, 1, 0))
