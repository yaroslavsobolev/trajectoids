import matplotlib.pyplot as plt
import numpy as np
from compute_trajectoid import *

def add_interval(startpoint, angle, length, Ns = 10):
    xs = startpoint[0] + np.linspace(0, length * np.cos(angle), Ns)
    ys = startpoint[1] + np.linspace(0, length * np.sin(angle), Ns)
    return np.stack((xs, ys)).T

def make_zigzag(a, Ns=10):
    angles = [np.pi/4, -np.pi/4]#, -np.pi/4, np.pi/4, -np.pi/4, np.pi/4]
    # angles = [0, np.pi]
    lengths = [a, np.pi]#, np.pi/2]#, a, np.pi/2, np.pi/2]
    input_path = np.array([[0, 0]])
    tips = [[0, 0]]
    for i, angle in enumerate(angles):
        startpoint = input_path[-1,:]
        new_section = add_interval(startpoint, angle, lengths[i], Ns=Ns)
        input_path = np.concatenate((input_path, new_section[1:]), axis=0)
        tips.append(new_section[-1])
    return input_path, np.array(tips)


def make_zigzag_tapered(zigzag_edge_length_without_taper=np.pi / 2,
                        zigzag_corner_angle=np.pi / 4,
                        taper_ratio=0.3, Ns=3):
    distance_from_taper_start_to_default_corner = taper_ratio * zigzag_edge_length_without_taper / 2
    taper_length = 2 * distance_from_taper_start_to_default_corner * np.sin(zigzag_corner_angle / 2)
    input_path = np.array([[0, 0]])
    angles = [0,
              -1 * (np.pi / 2 - zigzag_corner_angle / 2),
              0,
              np.pi / 2 - zigzag_corner_angle / 2,
              0]  # , -np.pi/4, np.pi/4, -np.pi/4, np.pi/4]
    lengths = [taper_length / 2,
               zigzag_edge_length_without_taper - 2 * distance_from_taper_start_to_default_corner,
               taper_length,
               zigzag_edge_length_without_taper - 2 * distance_from_taper_start_to_default_corner,
               taper_length / 2]
    tips = [[0, 0]]
    for i, angle in enumerate(angles):
        startpoint = input_path[-1, :]
        new_section = add_interval(startpoint, angle, lengths[i], Ns=Ns)
        input_path = np.concatenate((input_path, new_section[1:]), axis=0)
        tips.append(new_section[-1])
    input_path[:, 1] = input_path[:, 1] - input_path[0, 1]
    return input_path, np.array(tips)


path_parameter=0.2464
input_path, tips = make_zigzag_tapered(taper_ratio=path_parameter)
# scale = 2.6
# area = get_gb_area(scale * input_path, do_plot=True)
minscale = 0.1
maxscale = 15
nframes = 500
sweeped_scales, gb_areas = gb_areas_for_all_scales(input_path, minscale=minscale, maxscale=maxscale,
                                                   nframes=nframes, adaptive_sampling=True)
plt.plot(sweeped_scales, gb_areas, 'o-', alpha=0.5)
plt.show()

# test_gb_area_1()

# core_radius = 1
# fig = mlab.figure(size=(1024, 768), \
#             bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
# better_mayavi_lights(fig)
# tube_radius = 0.01
# plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)


# for scale_factor in np.linspace(0.8, 1.05, 10):
#     sphere_trace = trace_on_sphere(input_path, kx=scale_factor, ky=scale_factor)
#     l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=(0, 0, 1),
#                     tube_radius=tube_radius, opacity=0.3)
#     colors = [(1, 0, 0), (0, 1, 0)]
#     for i, point_here in enumerate([sphere_trace[0], sphere_trace[-1]]):
#         mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.05, color=colors[i])

# # data = make_path(xlen=9.5, r=2.82, Npath=150)
# data = make_path(2*np.pi, 0.5, Npath=150)
# # # data = make_path(xlen=9.5, r=2.5)
# trace_on_sphere(data, kx=0.5, ky=0.5, core_radius=1, do_plot=True)
