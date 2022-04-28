import numpy as np

from compute_trajectoid import *
#
# target_folder='examples/little-prince'

input_path_single_section = make_random_path(seed=1, factor=3, make_ends_horizontal='both', end_with_zero=True)

input_path_0 = double_the_path(input_path_single_section, do_plot=True)


#### Bridge the path
do_plot = True
npoints = 30
min_curvature_radius = 0.4

# optimal_netscale = get_scale_that_minimizes_end_to_end(input_path_0)
# input_path = optimal_netscale * input_path_0

optimal_netscale = 0.64
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
    l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(0, 0, 1),
                    tube_radius=tube_radius)
    l = mlab.plot3d(sphere_trace_single_section[:, 0],
                    sphere_trace_single_section[:, 1],
                    sphere_trace_single_section[:, 2], color=(0, 1, 0),
                    tube_radius=tube_radius)
    mlab.show()


plot_mismatch_map_for_scale_tweaking(input_path_0, kx_range=(0.8, 1.1), ky_range=(0.7, 1.1), vmax=1)

#
# # ### SINGLE ANGLE TEST
# # declination_angle = np.pi/4 # -1.3156036312041406
# # path_with_bridge, is_successful = make_smooth_bridge_candidate(declination_angle, input_path, npoints=npoints, do_plot=True,
# #                                                                make_animation=False, mlab_show=True)
# # print(f'Is successful: {is_successful}')
# # plot_bridged_path(path_with_bridge, savetofilename=False)
#
# ## BEST ANGLE TEST
#
# for netscale in np.linspace(optimal_netscale, 0, 20):
#     print(f'Netscale={netscale}')
#     input_path = netscale * input_path_0
#     best_declination_angle = find_best_smooth_bridge(input_path, npoints=npoints, max_declination=np.pi/180*80,
#                                                      min_curvature_radius=min_curvature_radius)
#     if best_declination_angle:
#         path_with_bridge, is_successful = make_smooth_bridge_candidate(best_declination_angle, input_path, npoints=npoints, do_plot=True,
#                                                                        mlab_show = True, make_animation=True,
#                                                                        min_curvature_radius=min_curvature_radius)
#         print(f'Scale = {netscale}')
#         print(f'Declination = {best_declination_angle}')
#         # plot results
#         plot_bridged_path(path_with_bridge, savetofilename=f'tests/figures/2d-path_netscale{netscale}.png', npoints=npoints)
#         np.save(target_folder + '/folder_for_path/path_data.npy', path_with_bridge)
#         np.savetxt(target_folder + '/folder_for_path/best_scale.txt', np.array([netscale]))
#         break
#     else:
#         print('No solution found.')
#
#
# # ## Make cut meshes for trajectoid
# # path_with_bridge = np.load(target_folder + '/folder_for_path/path_data.npy')
# # compute_shape(path_with_bridge, kx=1, ky=1,
# #               folder_for_path=target_folder + '/folder_for_path',
# #               folder_for_meshes=target_folder + '/cut_meshes',
# #               core_radius=1, cut_size = 10)
