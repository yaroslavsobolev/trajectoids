import numpy as np

from compute_trajectoid import *
from scipy.interpolate import interp1d

target_folder='examples/little-prince-2'

input_path_0 = np.loadtxt(target_folder + '/input_path.txt')
input_path_0 = sort_path(input_path_0)

# upsampling
f = interp1d(input_path_0[:,0], input_path_0[:,1], kind='quadratic')
xs = []
minstep = 0.015
for i in range(input_path_0.shape[0]-1):
    distance_to_next_point = np.linalg.norm(input_path_0[i+1, :] - input_path_0[i, :])
    additional_xs = np.linspace(input_path_0[i, 0], input_path_0[i+1, 0], num=int(round(distance_to_next_point/minstep)))
    xs.extend(list(additional_xs)[:-1])
xs.append(input_path_0[-1,0])
xs = np.array(xs)
ys = f(xs)
# plt.plot(input_path_0[:,0], input_path_0[:, 1], 'o-')
input_path_0 = np.stack((xs, ys)).T
# plt.plot(input_path_0[:,0], input_path_0[:, 1], 'x-', alpha=1)
# plt.show()

input_path_0[:,0] = input_path_0[:,0] - input_path_0[0,0]
input_path_0[:,1] = input_path_0[:,1] - input_path_0[0,1]
input_path_0[:,1] = input_path_0[:,1] - input_path_0[:,0]*(input_path_0[-1,1] / (input_path_0[-1,0] - input_path_0[0,0]) )
input_path_single_section = np.copy(input_path_0)
# plt.plot(input_path_0[:,0], input_path_0[:, 1], 'x-', alpha=1)
# plt.show()

# ## ==== for tests of various symmetries
# input_path_1 = np.copy(input_path_0)
# input_path_1[:,0] = 2*input_path_1[-1,0]-input_path_1[:,0]
# # input_path_1[:,0] = input_path_1[-1,0] + input_path_1[:,0]
# input_path_1[:,1] = -1*input_path_1[:,1]
#
# input_path_1 = np.concatenate((input_path_0, np.flipud))
# plt.plot(input_path_0[:,0], input_path_0[:,1], '-o')
# plt.plot(input_path_1[:,0], input_path_1[:,1], '-o')
# plt.axis('equal')
# plt.show()
#
# input_path_0 = np.concatenate((input_path_0, sort_path(input_path_1)[1:,]), axis=0)
# input_path_0 = sort_path(input_path_0)
# plt.plot(input_path_0[:,0], input_path_0[:,1], '-o', alpha=0.5)
# plt.axis('equal')
# plt.show()

input_path_0 = double_the_path(input_path_0, do_plot=True)

do_plot = True
npoints = 30
min_curvature_radius = 0.4

# optimal_netscale = get_scale_that_minimizes_end_to_end(input_path_0)
# input_path = optimal_netscale * input_path_0

optimal_netscale = 1
input_path = optimal_netscale * input_path_0

# plt.plot(input_path[:,0], input_path[:,1], '-o')
# plt.axis('equal')
# plt.show()
#
sphere_trace = trace_on_sphere(input_path, kx=1, ky=1)
sphere_trace_single_section = trace_on_sphere(input_path_single_section * optimal_netscale, kx=1, ky=1)
if do_plot:
    core_radius = 1
    mlab.figure(size=(1024, 768), \
                bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    tube_radius = 0.01
    plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
    l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(1, 0, 1),
                    tube_radius=tube_radius)
    l = mlab.plot3d(sphere_trace_single_section[:, 0],
                    sphere_trace_single_section[:, 1],
                    sphere_trace_single_section[:, 2], color=(0, 1, 0),
                    tube_radius=tube_radius)
    mlab.show()


# plot_mismatch_map_for_scale_tweaking(input_path_0, kx_range=(0.8, 1.1), ky_range=(0.7, 1.1), vmax=0.1, signed_angle=True)

best_scale = minimize_mismatch_by_scaling(input_path_0, scale_range=(0.9, 1.05))

input_path = best_scale * input_path_0
sphere_trace = trace_on_sphere(input_path, kx=1, ky=1)
sphere_trace_single_section = trace_on_sphere(input_path_single_section * best_scale, kx=1, ky=1)
if do_plot:
    core_radius = 1
    mlab.figure(size=(1024, 768), \
                bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    tube_radius = 0.01
    plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
    l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(1, 0, 1),
                    tube_radius=tube_radius)
    l = mlab.plot3d(sphere_trace_single_section[:, 0],
                    sphere_trace_single_section[:, 1],
                    sphere_trace_single_section[:, 2], color=(0, 1, 0),
                    tube_radius=tube_radius)
    mlab.view(elevation=60)
    for frame_id, azimuth in enumerate(np.linspace(0, 359, 60)):
        mlab.view(azimuth=azimuth)
        # camera_radius = 4
        # mfig.scene.camera.position = [camera_radius*np.cos(azimuth), camera_radius*np.sin(azimuth), -2.30]
        # print(mfig.actors)
        # mfig.actors.actor.rotate_y(5)
        mlab.savefig('examples/little-prince-2/azimuth_animation/{0:08d}.png'.format(frame_id))
    mlab.show()

plot_three_path_periods(input_path, plot_midpoints=True, savetofile=target_folder + '/input_path')



def plot_three_path_periods_color(input_path, savetofile=False, plot_midpoints=False):
    figtraj = plt.figure(10, figsize=(10, 5))
    dataxlen = np.max(input_path[:, 0])

    def plot_wcolor(x, y, color, alpha, linestyle, linewidth):
        alpha=1
        midindex = int(round(x.shape[0]/2))
        plt.plot(x[:midindex], y[:midindex], color='limegreen', alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        plt.plot(x[midindex:], y[midindex:], color='magenta', alpha=alpha, linestyle=linestyle, linewidth=linewidth)

    def plot_periods(data, linestyle, linewidth):
        plot_wcolor(data[:, 0], data[:, 1], color='black', alpha=0.3, linestyle=linestyle, linewidth=linewidth)
        plot_wcolor(dataxlen + data[:, 0], data[:, 1], color='black', alpha=1, linestyle=linestyle, linewidth=linewidth)
        plot_wcolor(2 * dataxlen + data[:, 0], data[:, 1], color='black', alpha=0.3, linestyle=linestyle,
                 linewidth=linewidth)
        plot_wcolor(3 * dataxlen + data[:, 0], data[:, 1], color='black', alpha=0.3, linestyle=linestyle,
                 linewidth=linewidth)

    # plot_periods(data, '--', linewidth=0.5)
    plot_periods(input_path, '-', linewidth=1)
    # plot_periods(projection_centers, '-', linewidth=1)

    for shift in dataxlen * np.arange(3):
        plt.scatter(shift + input_path[-1, 0], input_path[-1, 1], s=35, color='black')
    # plt.scatter(dataxlen + input_path[-1, 0], input_path[-1, 1], s=35, color='black')
    if plot_midpoints:
        midpoint_index = int(round(input_path.shape[0]/2))
        for shift in dataxlen*np.arange(4):
            plt.scatter(shift + input_path[midpoint_index, 0], input_path[midpoint_index, 1], s=35, facecolors='white', edgecolors='black')
    plt.axis('equal')
    if savetofile:
        figtraj.savefig(f'{savetofile}.png', dpi=300)
        figtraj.savefig(f'{savetofile}.eps')
    plt.show()

plot_three_path_periods_color(input_path, plot_midpoints=False, savetofile=target_folder + '/input_path_wcolors_orig_nopoints')
# np.save(target_folder + '/folder_for_path/path_data.npy', input_path)
# np.savetxt(target_folder + '/folder_for_path/best_scale.txt', np.array([best_scale]))


# ## Make cut meshes for trajectoid
# input_path = np.load(target_folder + '/folder_for_path/path_data.npy')
# compute_shape(input_path, kx=1, ky=1,
#               folder_for_path=target_folder + '/folder_for_path',
#               folder_for_meshes=target_folder + '/cut_meshes',
#               core_radius=1, cut_size = 10)
