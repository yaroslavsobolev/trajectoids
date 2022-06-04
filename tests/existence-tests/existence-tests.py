import matplotlib.pyplot as plt
import numpy as np
import mayavi
from scipy.interpolate import interp1d

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
        mismatch_angles.append(mismatch_angle_for_path(input_path_single_section))
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

def upsample_path(input_path, by_factor=10):
    old_indices = np.arange(input_path.shape[0])
    max_index = input_path.shape[0]-1
    new_indices = np.arange(0, max_index, 1/by_factor)
    new_indices = np.append(new_indices, max_index)
    new_xs = interp1d(old_indices, input_path[:, 0])(new_indices)
    new_ys = interp1d(old_indices, input_path[:, 1])(new_indices)
    return np.stack((new_xs, new_ys)).T

target_folder='tests/existence-tests/path_folder'

# input_path_single_section = make_random_path(seed=1, amplitude=3, make_ends_horizontal='both', end_with_zero=True)

input_path_single_section = make_brownian_path(seed=0, Npath=150, travel_length=0.1)

input_path_0 = double_the_path_nosort(input_path_single_section, do_plot=True)

# upsampled_path = upsample_path(input_path_0)
# plt.plot(upsampled_path[:, 0], upsampled_path[:, 1], 'o-', alpha=0.5)
# plt.show()
#
best_scale = 20
plotting_upsample_factor = 2
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
l = mlab.plot3d(sphere_trace[last_index:, 0],
                sphere_trace[last_index:, 1],
                sphere_trace[last_index:, 2], color=(0, 0, 1),
                tube_radius=tube_radius)
l = mlab.plot3d(sphere_trace_single_section[:, 0],
                sphere_trace_single_section[:, 1],
                sphere_trace_single_section[:, 2], color=(0, 1, 0),
                tube_radius=tube_radius)
mlab.show()


# sweeped_scales, mismatch_angles = mismatches_for_all_scales(input_path_single_section, minscale=24.5, maxscale=25.1)
sweeped_scales, mismatch_angles = mismatches_for_all_scales(input_path_0, minscale=24, maxscale=25)
plt.plot(sweeped_scales, np.abs(mismatch_angles))
plt.plot(sweeped_scales, mismatch_angles)
plt.show()


do_plot = True
npoints = 30

best_scale = minimize_mismatch_by_scaling(input_path_0, scale_range=(24.7, 24.9))
print(f'Best scale: {best_scale}')
# Minimized mismatch angle = -2.6439127092433114e-05
# Best scale: 0.6387022944333781

input_path = best_scale * input_path_0
sphere_trace = trace_on_sphere(input_path, kx=1, ky=1)
sphere_trace_single_section = trace_on_sphere(input_path_single_section * best_scale, kx=1, ky=1)
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

# sweeped_scales, mismatch_angles = mismatches_for_all_scales()
# plt.plot(sweeped_scales, np.abs(mismatch_angles))
# plt.show()

# make_animation_of_rotational_symmetry()

# make_orbit_animation(folder_for_frames='examples/penannular_1/orbit_frames', elevation=60)

# for point_here in [sphere_trace_single_section[-1], sphe]:
#     mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.05, color=(1, 1, 0))
