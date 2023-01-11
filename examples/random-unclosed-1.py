from compute_trajectoid import *

target_folder = 'examples/random_unclosed_1'
input_path_0 = make_random_path(seed=0, make_ends_horizontal=False, start_from_zero=True, end_with_zero=True, amplitude=3)

input_path_0 = upsample_path(input_path_0, by_factor=2, kind='linear')


do_plot = True

# optimal_netscale = get_scale_that_minimizes_end_to_end(input_path)
# input_path = optimal_netscale * input_path_0

optimal_netscale = 1.15
optimal_netscale = 1.2
input_path = optimal_netscale * input_path_0
# input_path_doubled = double_the_path(input_path)
tube_radius = 0.01
sphere_trace = trace_on_sphere(input_path, kx=1, ky=1, do_plot=False) * (1 + tube_radius*3)
# sphere_trace_doubled = trace_on_sphere(input_path_doubled, kx=1, ky=1)

if do_plot:
    core_radius = 1
    mlab.figure(size=(1024, 768), \
                bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4, sphere_opacity=0.4)
    l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=tuple(np.array([130, 130, 130])/255),# color=tuple(np.array([31,119,180])/255),
                    tube_radius=tube_radius*3)#, opacity=0.4)
    # first = 298
    # last = 330
    # l = mlab.plot3d(sphere_trace_doubled[first:last, 0], sphere_trace_doubled[first:last, 1], sphere_trace_doubled[first:last, 2], color=tuple(np.array([31,119,180])/255),
    #                 tube_radius=tube_radius, opacity=0.1)
    mlab.show()

    figtraj = plt.figure(10, figsize=(10, 5))
    dataxlen = np.max(input_path[:, 0])

savetofile=target_folder + '/input_path_illustration'
linewidth=3
def plot_periods(data, linestyle, linewidth):
    plt.plot(data[:, 0], data[:, 1], color='black', alpha=0.3, linestyle=linestyle, linewidth=linewidth)
    plt.plot(dataxlen + data[:, 0], data[:, 1], color='black', alpha=1, linestyle=linestyle, linewidth=linewidth)
    plt.plot(2 * dataxlen + data[:, 0], data[:, 1], color='black', alpha=0.3, linestyle=linestyle,
             linewidth=linewidth)
    plt.plot(3 * dataxlen + data[:, 0], data[:, 1], color='black', alpha=0.3, linestyle=linestyle,
             linewidth=linewidth)

# plot_periods(data, '--', linewidth=0.5)
plot_periods(input_path, '-', linewidth=linewidth)
# plot_periods(projection_centers, '-', linewidth=1)

for shift in dataxlen * np.arange(3):
    plt.scatter(shift + input_path[-1, 0], input_path[-1, 1], s=35, color='black')
# plt.scatter(dataxlen + input_path[-1, 0], input_path[-1, 1], s=35, color='black')
plt.axis('equal')
if savetofile:
    figtraj.savefig(f'{savetofile}.png', dpi=300)
    figtraj.savefig(f'{savetofile}.eps')
plt.show()


# plot_three_path_periods(input_path, plot_midpoints=False, savetofile=target_folder + '/input_path')


# ## Make cut meshes for trajectoid
# compute_shape(input_path, kx=1, ky=1,
#               folder_for_path=target_folder + '/folder_for_path',
#               folder_for_meshes=target_folder + '/cut_meshes',
#               core_radius=1, cut_size = 10)