import matplotlib.pyplot as plt
import numpy as np
from compute_trajectoid import *

def add_interval(startpoint, angle, length, Ns = 10):
    xs = startpoint[0] + np.linspace(0, length * np.cos(angle), Ns)
    ys = startpoint[1] + np.linspace(0, length * np.sin(angle), Ns)
    return np.stack((xs, ys)).T

def make_zigzag(a):
    angles = [np.pi/4, -np.pi/4]#, -np.pi/4, np.pi/4, -np.pi/4, np.pi/4]
    # angles = [0, np.pi]
    lengths = [a, np.pi]#, np.pi/2]#, a, np.pi/2, np.pi/2]
    input_path = np.array([[0, 0]])
    tips = [[0, 0]]
    for i, angle in enumerate(angles):
        startpoint = input_path[-1,:]
        new_section = add_interval(startpoint, angle, lengths[i])
        input_path = np.concatenate((input_path, new_section[1:]), axis=0)
        tips.append(new_section[-1])
    return input_path, np.array(tips)

input_path, tips = make_zigzag(np.pi*1.0000001)
fig = plt.figure(dpi=300)
plt.plot(input_path[:, 0], input_path[:, 1], '-o', color='black', alpha=0.5)
plt.plot(tips[:, 0], tips[:, 1], 'o', color='black', alpha=0.5)
plt.axis('equal')
plt.show()

area = get_gb_area(input_path)
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
