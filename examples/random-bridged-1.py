from compute_trajectoid import *

target_folder = 'examples/random_bridged_1'
input_path_0 = make_random_path(seed=0, make_ends_horizontal=False, start_from_zero=True, end_with_zero=True, amplitude=3)

#### Bridge the path
do_plot = True
npoints = 30

# optimal_netscale = get_scale_that_minimizes_end_to_end(input_path)
# input_path = optimal_netscale * input_path_0

optimal_netscale = 0.93
input_path = optimal_netscale * input_path_0

sphere_trace = trace_on_sphere(input_path, kx=1, ky=1)

# if do_plot:
#     core_radius = 1
#     mlab.figure(size=(1024, 768), \
#                 bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
#     tube_radius = 0.01
#     plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
#     l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(0, 0, 1),
#                     tube_radius=tube_radius)
#     mlab.show()


# ### SINGLE ANGLE TEST
# declination_angle = np.pi/4 # -1.3156036312041406
# path_with_bridge, is_successful = make_smooth_bridge_candidate(declination_angle, input_path, npoints=npoints, do_plot=True,
#                                                                make_animation=False, mlab_show=True)
# print(f'Is successful: {is_successful}')
# plot_bridged_path(path_with_bridge, savetofilename=False)

# ## BEST ANGLE TEST
# for netscale in np.linspace(optimal_netscale, 0, 20):
#     print(f'Netscale={netscale}')
#     input_path = netscale * input_path_0
#     best_declination_angle = find_best_smooth_bridge(input_path, npoints=npoints, max_declination=np.pi/180*80)
#     if best_declination_angle:
#         path_with_bridge, is_successful = make_smooth_bridge_candidate(best_declination_angle, input_path, npoints=npoints, do_plot=True,
#                                                                        mlab_show = True, make_animation=True)
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
# ## Make cut meshes for trajectoid
# path_with_bridge = np.load(target_folder + '/folder_for_path/path_data.npy')
# compute_shape(path_with_bridge, kx=1, ky=1,
#               folder_for_path=target_folder + '/folder_for_path',
#               folder_for_meshes=target_folder + '/cut_meshes',
#               core_radius=1, cut_size = 10)

# path_with_bridge = np.load(target_folder + '/folder_for_path/path_data.npy')
# plot_three_path_periods(input_path, plot_midpoints=True, savetofile=target_folder + '/input_path')
# plot_bridged_path(path_with_bridge, savetofilename=f'examples/random_bridged_1/input_path.png', npoints=npoints)

# For illustration in the paper
path_with_bridge = np.load(target_folder + '/folder_for_path/path_data.npy')
input_path_length = input_path.shape[0]
core_radius = 1
mfig = mlab.figure(size=(1024, 768), bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
tube_radius = 0.01
arccolor = tuple(np.array([44, 160, 44]) / 255)
arccolor = (1, 0, 0)
plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4, sphere_opacity=0.4)
tube_radius = tube_radius * 3

sphere_trace = trace_on_sphere(input_path, kx=1, ky=1) * (1 + tube_radius)
l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=tuple(np.array([130, 130, 130])/255),
                tube_radius=tube_radius)

sphere_trace = trace_on_sphere(path_with_bridge, kx=1, ky=1) * (1 + tube_radius)
forward_arc_points, forward_straight_section_points, main_arc_points, backward_straight_section_points, backward_arc_points =\
            tuple(sphere_trace[input_path_length + npoints * n - 1
                  :input_path_length + npoints * (n + 1), :] for n in range(5))
for piece_of_bridge in [forward_arc_points, backward_arc_points]:
    p = mlab.plot3d(piece_of_bridge[:, 0], piece_of_bridge[:, 1], piece_of_bridge[:, 2], color=arccolor,
                    tube_radius=tube_radius)
for piece_of_bridge in [forward_straight_section_points, backward_straight_section_points]:
    p = mlab.plot3d(piece_of_bridge[:, 0], piece_of_bridge[:, 1], piece_of_bridge[:, 2],
                    color=tuple(np.array([255, 127, 14]) / 255),
                    tube_radius=tube_radius)
for piece_of_bridge in [main_arc_points]:
    p = mlab.plot3d(piece_of_bridge[:, 0], piece_of_bridge[:, 1], piece_of_bridge[:, 2], color=arccolor,
                    tube_radius=tube_radius)
# for point_here in [backward_arc_points[0]]:
#     mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.05, color=(1, 1, 0))
mlab.view(elevation=120, azimuth=180, roll=-90)
# mlab.savefig('tests/figures/{0:.2f}.png'.format(input_declination_angle))
mlab.show()

# Flat path for first illustration
path_with_bridge = np.load(target_folder + '/folder_for_path/path_data.npy')
input_path_length = input_path.shape[0]
path = path_with_bridge
savetofilename='examples/random_bridged_1/input_path_illustration.png'
npoints=30
netscale=1
linewidth=3*0.75
fig, ax = plt.subplots(figsize=(12, 2))
alphabridge = 0.8
bridgelen = npoints * 5 - 5
dxs = [0,
       - path[-1, 0],
       path[-1, 0],
       2 * path[-1, 0]]
dys = [0,
       - path[-1, 1] + path[0, 1],
       - path[0, 1] + path[-1, 1],
       - path[0, 1] + 2 * path[-1, 1]]
for k in range(len(dxs)):
    dx = dxs[k]
    dy = dys[k]
    if k == 2:
        alpha = 1
    else:
        alpha = 0.3
    plt.plot(path[:input_path_length, 0] + dx,
             path[:input_path_length, 1] + dy, '-', alpha=alpha, color='black', linewidth=3*0.75, zorder=10)
    plt.plot(path[-(bridgelen):, 0] + dx,
             path[-(bridgelen):, 1] + dy, '-', alpha=alphabridge, color='C1', linewidth=linewidth)
    plt.plot(path[-(bridgelen):-(bridgelen) + npoints - 1, 0] + dx,
             path[-(bridgelen):-(bridgelen) + npoints - 1, 1] + dy, '-', alpha=alphabridge, color='red',
             linewidth=linewidth)
    plt.plot(path[-(bridgelen) + npoints * 2 - 2:-(bridgelen) + npoints * 3 - 3, 0] + dx,
             path[-(bridgelen) + npoints * 2 - 2:-(bridgelen) + npoints * 3 - 3, 1] + dy, '-', alpha=alphabridge,
             color='red',
             linewidth=linewidth)
    plt.plot(path[-(npoints - 1):, 0] + dx,
             path[-(npoints - 1):, 1] + dy, '-', alpha=alphabridge, color='red',
             linewidth=linewidth)
plt.scatter([path[0, 0], path[-1, 0], path[0, 0] + 2 * path[-1, 0]], [path[0, 1], path[-1, 1], 2 * path[-1, 1]], s=35,
            alpha=0.8, color='black', zorder=100)
# plt.scatter([path[0, 0], path[-1, 0]], [path[0, 1], path[-1, 1]], s=35, alpha=0.8, color='black', zorder=100)

plt.axis('equal')
plt.xlim(-8, -8 + 35 * netscale)
ax.axis('off')
if savetofilename:
    fig.savefig(savetofilename, dpi=300)
plt.show()
