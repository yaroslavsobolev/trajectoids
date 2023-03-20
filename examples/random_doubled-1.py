import numpy as np

from compute_trajectoid import *
#
target_folder='examples/random_doubled_1'

# input_path_single_section = make_random_path(seed=1, amplitude=3, make_ends_horizontal='both', end_with_zero=True)
input_path_single_section = make_random_path(seed=0, make_ends_horizontal=False, start_from_zero=True, end_with_zero=True, amplitude=3)

input_path_0 = double_the_path(input_path_single_section, do_plot=True)

do_plot = True
npoints = 30

best_scale = minimize_mismatch_by_scaling(input_path_0, scale_range=(0.5, 0.7))
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

plot_three_path_periods(input_path, plot_midpoints=True, savetofile=target_folder + '/input_path')

np.save(target_folder + '/folder_for_path/path_data.npy', input_path)
np.savetxt(target_folder + '/folder_for_path/best_scale.txt', np.array([best_scale]))

## Make cut meshes for trajectoid
input_path = np.load(target_folder + '/folder_for_path/path_data.npy')
compute_shape(input_path, kx=1, ky=1,
              folder_for_path=target_folder + '/folder_for_path',
              folder_for_meshes=target_folder + '/cut_meshes',
              core_radius=1, cut_size = 10)
