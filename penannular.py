import matplotlib.pyplot as plt
import numpy as np

from compute_trajectoid import *

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

def make_path(xlen, r, Npath = 400, do_double=True):
    # first linear section
    step_size = xlen/Npath
    overall_xs = np.linspace(0, xlen/2 - r, int(round(xlen/2 - r)/step_size))
    overall_ys = np.zeros_like(overall_xs)

    # semicirle section
    nsteps_in_theta = int(round(np.pi*r/step_size))
    thetas = np.linspace(np.pi, 0, nsteps_in_theta)
    xs = r*np.cos(thetas) + xlen/2
    ys = r*np.sin(thetas)
    overall_xs = np.concatenate((overall_xs[:-1], xs))
    overall_ys = np.concatenate((overall_ys[:-1], ys))

    # second linear section
    xs = np.linspace(xlen/2 + r, xlen, int(round(xlen/2 - r)/step_size))
    ys = np.zeros_like(xs)
    overall_xs = np.concatenate((overall_xs, xs[1:]))
    overall_ys = np.concatenate((overall_ys, ys[1:]))

    input_path = np.stack((overall_xs, overall_ys)).T

    if do_double:
        input_path = double_the_path(input_path)

    return input_path

def plot_mismatch_map_for_penannular(N=60, M=60, kx_range=(0.1, 5*np.pi), kr_range=(0.01, 1.5*np.pi)):
    # sweeping parameter space for optimal match of the starting and ending orientation
    angles = np.zeros(shape=(N, M))
    xs = np.zeros_like(angles)
    ys = np.zeros_like(angles)
    for i, kx in enumerate(np.linspace(kx_range[0], kx_range[1], N)):
        print(i)
        for j, r in enumerate(np.linspace(kr_range[0], kr_range[1], M)):
            xs[i, j] = kx
            ys[i, j] = r
            if kx<2*r:
                angles[i, j] = np.nan
            else:
                data = make_path(xlen=kx, r=r)
                rotation_of_entire_traj = trimesh.transformations.rotation_from_matrix(rotation_to_origin(data.shape[0]-1, data))
                angle = rotation_of_entire_traj[0]
                angles[i, j] = angle

    print('Min angle = {0}'.format(np.min(np.abs(angles))))
    f3 = plt.figure(3)
    plt.pcolormesh(xs, ys, np.abs(angles), cmap='viridis')
    plt.colorbar()
    plt.ylabel('radius')
    plt.xlabel('total length')
    plt.show()

path = make_path(2*np.pi, 0.5)
plt.scatter(path[:, 0], path[:, 1], alpha=0.5, color='C0')
plt.axis('equal')
plt.show()

# plot_mismatch_map_for_penannular(N=20,
#                                  M=20,
#                                  kx_range=(0.1, 11.5),
#                                  kr_range=(0.01, 1.5*np.pi))


# plot_mismatch_map_for_penannular(N=20,
#                                  M=20,
#                                  kx_range=(3.2, 4),
#                                  kr_range=(0.5, 2))

# # data = make_path(xlen=9.5, r=2.82, Npath=150)
# data = make_path(2*np.pi, 0.5, Npath=150)
# # # data = make_path(xlen=9.5, r=2.5)
# trace_on_sphere(data, kx=0.5, ky=0.5, core_radius=1, do_plot=True)
input_path = make_path(xlen=3.81, r=1.23, Npath=150)
input_path_single_section = make_path(xlen=3.81, r=1.23, Npath=150, do_double=False)
_ = double_the_path(input_path_single_section, do_plot=True)

best_scale = minimize_mismatch_by_scaling(input_path, scale_range=(0.9, 1.1))
input_path = input_path * best_scale
input_path_single_section = input_path_single_section * best_scale


# data = make_path(2*np.pi, 0.5, Npath=150)
# # # data = make_path(xlen=9.5, r=2.5)
# trace_on_sphere(data, kx=1, ky=1, core_radius=1, do_plot=True)
sphere_trace = trace_on_sphere(input_path, kx=1, ky=1)
sphere_trace_single_section = trace_on_sphere(input_path_single_section, kx=1, ky=1)
core_radius = 1
mlab.figure(size=(1024, 768), \
            bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
tube_radius = 0.01
plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
last_index = sphere_trace.shape[0] // 2
l = mlab.plot3d(sphere_trace[last_index:, 0], sphere_trace[last_index:, 1], sphere_trace[last_index:, 2], color=(0, 0, 1),
                tube_radius=tube_radius)
l = mlab.plot3d(sphere_trace_single_section[:, 0],
                sphere_trace_single_section[:, 1],
                sphere_trace_single_section[:, 2], color=(0, 1, 0),
                tube_radius=tube_radius)

# make_orbit_animation(folder_for_frames='examples/penannular_1/orbit_frames', elevation=60)

# for point_here in [sphere_trace_single_section[-1], sphe]:
#     mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.05, color=(1, 1, 0))
mlab.show()