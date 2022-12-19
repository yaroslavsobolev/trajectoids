from compute_trajectoid import *

target_folder = 'examples/random_unclosed_1'
input_path_0 = make_random_path(seed=0, make_ends_horizontal=False, start_from_zero=True, end_with_zero=True, amplitude=3)

input_path_0 = upsample_path(input_path_0, by_factor=2, kind='linear')


#### Bridge the path
do_plot = True

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

# plot_three_path_periods(input_path, plot_midpoints=False, savetofile=target_folder + '/input_path')


# ## Make cut meshes for trajectoid
# compute_shape(input_path, kx=1, ky=1,
#               folder_for_path=target_folder + '/folder_for_path',
#               folder_for_meshes=target_folder + '/cut_meshes',
#               core_radius=1, cut_size = 10)