import matplotlib.pyplot as plt
import numpy as np

from compute_trajectoid import *


target_folder = 'examples/random_warped_1'
input_path_0 = make_random_path(seed=0, make_ends_horizontal=False, start_from_zero=True, end_with_zero=True, amplitude=3)

input_path_0 = upsample_path(input_path_0, by_factor=2, kind='linear')

xlength = input_path_0[-1, 0] - input_path_0[0, 0]

# plt.plot(input_path_0[:, 0], input_path_0[:, 1])
# plt.axis('equal')
# plt.show()

# make arc
# for R in np.linspace(7, 10, 20):

################################ MAPPING

# N=20
# M=20
# kx_range=(-0.6, -0.2)
# ky_range=(3.1, 4.5)
# # vmin=0
# # vmax=np.pi
# signed_angle=False
# # sweeping parameter space for optimal match of the starting and ending orientation
# mismatchesmap = np.zeros(shape=(N, M))
# xs = np.zeros_like(mismatchesmap)
# ys = np.zeros_like(mismatchesmap)
# for i, kx in enumerate(np.linspace(kx_range[0], kx_range[1], N)):
#     print(i)
#     for j, ky in enumerate(np.linspace(ky_range[0], ky_range[1], M)):
#         input_path = np.copy(input_path_0)
#         # input_path[:, 1] += kx * np.sin(input_path[:, 0] / xlength * 2 * np.pi * 1.5)
#         input_path[:, 0] += kx * np.sin(input_path[:, 0] / xlength * 2 * np.pi * 1)
#         # input_path[:, 1] += ky * np.sin(input_path[:, 0] / xlength * 2 * np.pi * 0.5)
#         # input_path[:, 1] = ky * np.sin(input_path[:, 0] / xlength * 2 * np.pi * 0.5)
#
#         R = ky
#         y_center = -1 * np.sqrt(R**2 - (xlength/2)**2)
#         angle = np.arcsin((xlength/2)/R)
#         circle_ys = y_center + np.sqrt(R**2 - (xlength/2 - input_path_0[:, 0])**2)
#         input_path[:, 1] = input_path[:, 1] + circle_ys
#
#         xs[i, j] = kx
#         ys[i, j] = ky
#         scales = np.linspace(0.8, 1.5, 20)
#         mismatches = np.array([mismatch_angle_for_path(input_path * scale) for scale in scales])
#         mismatchesmap[i, j] = np.min(np.abs(mismatches))
#
# print('Min angle = {0}'.format(np.min(np.abs(mismatchesmap))))
# f3 = plt.figure(3)
# plt.pcolormesh(xs, ys, mismatchesmap, cmap='viridis', vmin=0)
# plt.colorbar()
# plt.show()


# for A in np.linspace(-0.35, -0.45, 10):
#     R = 4.3
#     xlength = input_path_0[-1, 0] - input_path_0[0, 0]
#     xs = np.linspace(input_path_0[0, 0], input_path_0[-1, 0], input_path_0.shape[0])
#     y_center = -1 * np.sqrt(R**2 - (xlength/2)**2)
#     angle = np.arcsin((xlength/2)/R)
#     ys = y_center + np.sqrt(R**2 - (xlength/2 - xs)**2)
#     # plt.plot(xs, ys)
#
#     input_path = np.copy(input_path_0)
#     # input_path[:, 1] = input_path_0[:, 1] + ys
#
#     input_path[:, 1] += A * np.sin(xs/xlength*2*np.pi * 1.5)
#
#     scales = np.linspace(0.9, 1.5, 30)
#     mismatches = [mismatch_angle_for_path(input_path * scale) for scale in scales]
#     plt.plot(scales, mismatches, 'o-', label=f'{A}')
#
# plt.legend()
# # plt.plot(input_path_0[:, 0], input_path_0[:, 1])
#
#
# # input_path_0[:, 1] *= 0.4
# # ys = ys * 0.6
#
# # plt.show()
#
#
# plt.show()




## VERSION ONE

# best_scale = minimize_mismatch_by_scaling(input_path_0, scale_range=(0.5, 2))
# print(best_scale)

# kx = -0.494
# ky = 0.691
# input_path = np.copy(input_path_0)
# input_path[:, 0] += kx * np.sin(input_path[:, 0] / xlength * 2 * np.pi * 1)
# input_path[:, 1] += ky * np.sin(input_path[:, 0] / xlength * 2 * np.pi * 0.5)
#
# plt.plot(input_path[:, 0], input_path[:, 1])
# plt.axis('equal')
# plt.show()
#
# scales = np.linspace(0.8, 1.5, 20)
# mismatches = [mismatch_angle_for_path(input_path * scale) for scale in scales]
# plt.plot(scales, mismatches, 'o-')
# plt.show()
#
# # best_scale = minimize_mismatch_by_scaling(input_path, scale_range=(1.2, 1.35))
# # print(best_scale)
#
#
# do_plot = True
#
# # optimal_netscale = get_scale_that_minimizes_end_to_end(input_path)
# # input_path = optimal_netscale * input_path_0
#
# optimal_netscale = 1.28319
# input_path_scaled = optimal_netscale * input_path
# # input_path_doubled = double_the_path(input_path)
# tube_radius = 0.01
# sphere_trace = trace_on_sphere(input_path_scaled, kx=1, ky=1, do_plot=False) * (1 + tube_radius*3)
# # sphere_trace_doubled = trace_on_sphere(input_path_doubled, kx=1, ky=1)
#
# if do_plot:
#     core_radius = 1
#     mlab.figure(size=(1024, 768), \
#                 bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
#     plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4, sphere_opacity=0.4)
#     l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=tuple(np.array([130, 130, 130])/255),# color=tuple(np.array([31,119,180])/255),
#                     tube_radius=tube_radius*3)#, opacity=0.4)
#     # first = 298
#     # last = 330
#     # l = mlab.plot3d(sphere_trace_doubled[first:last, 0], sphere_trace_doubled[first:last, 1], sphere_trace_doubled[first:last, 2], color=tuple(np.array([31,119,180])/255),
#     #                 tube_radius=tube_radius, opacity=0.1)
#     mlab.show()
#
#     figtraj = plt.figure(10, figsize=(10, 5))
#     dataxlen = np.max(input_path[:, 0])
#
# savetofile=target_folder + '/input_path_illustration'
# linewidth=3
# def plot_periods(data, linestyle, linewidth):
#     plt.plot(data[:, 0], data[:, 1], color='black', alpha=0.3, linestyle=linestyle, linewidth=linewidth)
#     plt.plot(dataxlen + data[:, 0], data[:, 1], color='black', alpha=1, linestyle=linestyle, linewidth=linewidth)
#     plt.plot(2 * dataxlen + data[:, 0], data[:, 1], color='black', alpha=0.3, linestyle=linestyle,
#              linewidth=linewidth)
#     plt.plot(3 * dataxlen + data[:, 0], data[:, 1], color='black', alpha=0.3, linestyle=linestyle,
#              linewidth=linewidth)
#
# # plot_periods(data, '--', linewidth=0.5)
# plot_periods(input_path, '-', linewidth=linewidth)
# # plot_periods(projection_centers, '-', linewidth=1)
#
# for shift in dataxlen * np.arange(3):
#     plt.scatter(shift + input_path[-1, 0], input_path[-1, 1], s=35, color='black')
# # plt.scatter(dataxlen + input_path[-1, 0], input_path[-1, 1], s=35, color='black')
# plt.axis('equal')
# if savetofile:
#     figtraj.savefig(f'{savetofile}.png', dpi=300)
#     figtraj.savefig(f'{savetofile}.eps')
# plt.show()


############# VERSION TWO

best_scale = minimize_mismatch_by_scaling(input_path_0, scale_range=(0.5, 2))
print(best_scale)

kx = -0.4
ky = 3.2
input_path = np.copy(input_path_0)
input_path[:, 0] += kx * np.sin(input_path[:, 0] / xlength * 2 * np.pi * 1)
R = ky
y_center = -1 * np.sqrt(R ** 2 - (xlength / 2) ** 2)
angle = np.arcsin((xlength / 2) / R)
circle_ys = y_center + np.sqrt(R ** 2 - (xlength / 2 - input_path_0[:, 0]) ** 2)
input_path[:, 1] = input_path[:, 1] + circle_ys

#
#
# node_interval = 15
# node_ids = np.arange(0, input_path_0.shape[0], node_interval)
# for node_id in node_ids:
#     startpoint = input_path_0[node_id, :]
#     endpoint = input_path[node_id, :]
#     plt.plot([startpoint[0], endpoint[0]], [startpoint[1], endpoint[1]], color='black', alpha=0.5)
#
# plt.plot(input_path_0[:, 0], input_path_0[:, 1])
# plt.plot(input_path[:, 0], input_path[:, 1])
# plt.axis('equal')
# plt.show()
#
#
# figtraj = plt.figure(10, figsize=(10, 5))
# dataxlen = np.max(input_path[:, 0])
# savetofile=target_folder + '/input_path_illustration'
# linewidth=3
# color = 'C0'
# def plot_periods(data, linestyle, linewidth, secondalpha=0.5):
#     plt.plot(data[:, 0], data[:, 1], color=color, alpha=secondalpha, linestyle=linestyle, linewidth=linewidth)
#     plt.plot(dataxlen + data[:, 0], data[:, 1], color=color, alpha=1, linestyle=linestyle, linewidth=linewidth)
#     plt.plot(2 * dataxlen + data[:, 0], data[:, 1], color=color, alpha=secondalpha, linestyle=linestyle,
#              linewidth=linewidth)
#     plt.plot(3 * dataxlen + data[:, 0], data[:, 1], color=color, alpha=secondalpha, linestyle=linestyle,
#              linewidth=linewidth)
#
# # plot_periods(data, '--', linewidth=0.5)
# plot_periods(input_path, '-', linewidth=linewidth)
# # plot_periods(projection_centers, '-', linewidth=1)
#
# node_interval = 15
# node_ids = np.arange(0, input_path_0.shape[0], node_interval)
# for node_id in node_ids:
#     startpoint = input_path_0[node_id, :]
#     endpoint = input_path[node_id, :]
#     plt.plot([startpoint[0] + dataxlen, endpoint[0] + dataxlen], [startpoint[1], endpoint[1]], color='C2', alpha=0.5)
# plt.plot(input_path_0[:, 0] + dataxlen, input_path_0[:, 1], color='black', alpha=1, linestyle='-', linewidth=linewidth)
#
# for shift in dataxlen * np.arange(3):
#     plt.scatter(shift + input_path[-1, 0], input_path[-1, 1], s=35, color='black')
# # plt.scatter(dataxlen + input_path[-1, 0], input_path[-1, 1], s=35, color=color)
# plt.axis('equal')
# if savetofile:
#     figtraj.savefig(f'{savetofile}.png', dpi=300)
#     figtraj.savefig(f'{savetofile}.eps')
# plt.show()
#
#
# scales = np.linspace(0.8, 1.5, 20)
# mismatches = [mismatch_angle_for_path(input_path * scale) for scale in scales]
# plt.plot(scales, mismatches, 'o-')
# plt.show()



# best_scale = minimize_mismatch_by_scaling(input_path, scale_range=(1.2, 1.35))
# print(best_scale)

# optimal_netscale = get_scale_that_minimizes_end_to_end(input_path)
# input_path = optimal_netscale * input_path_0

optimal_netscale = 1.28319
input_path_scaled = optimal_netscale * input_path
# input_path_doubled = double_the_path(input_path)
tube_radius = 0.01
sphere_trace = trace_on_sphere(input_path_scaled, kx=1, ky=1, do_plot=False) * (1 + tube_radius*3)
# sphere_trace_doubled = trace_on_sphere(input_path_doubled, kx=1, ky=1)

core_radius = 1
mlab.figure(size=(1024, 768), \
            bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4, sphere_opacity=0.4)
l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=tuple(np.array([31,119,180])/255), # tuple(np.array([130, 130, 130])/255)
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