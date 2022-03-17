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

def make_path(alpha, Npath = 400):
    beta = alpha/(2*np.pi)
    # first linear section
    h = 2 - 1 / (1 - beta)
    A = np.pi/2 + np.arcsin(1-h)
    step_size = A/Npath

    # straight section
    xlen = A * np.cos(np.pi/2 - alpha/2)
    overall_xs = np.linspace(0, xlen, int(round(A/step_size)))
    overall_ys = overall_xs * np.tan(np.pi/2 - alpha/2)

    # semicirle section
    rg = 1/beta * np.sqrt(1 - 2 * beta)
    circle_center_x = (A + rg) * np.cos(np.pi / 2 - alpha / 2)
    circle_center_y = (A + rg) * np.sin(np.pi / 2 - alpha / 2)
    # plt.scatter(x=circle_center_x, y=circle_center_y, color='r')
    nsteps_in_theta = int(round((rg * alpha/2)/step_size))
    thetas = np.linspace(-np.pi/2 - alpha/2, -np.pi/2, nsteps_in_theta)
    xs = rg*np.cos(thetas) + circle_center_x
    ys = rg*np.sin(thetas) + circle_center_y
    overall_xs = np.concatenate((overall_xs[:-1], xs))
    overall_ys = np.concatenate((overall_ys[:-1], ys))


    overall_xs = np.concatenate((overall_xs[:-1], overall_xs[-1]*2 - np.flip(overall_xs)))
    overall_ys = np.concatenate((overall_ys[:-1], np.flip(overall_ys)))

    # # second linear section
    # xs = np.linspace(xlen/2 + r, xlen, int(round(xlen/2 - r)/step_size))
    # ys = np.zeros_like(xs)
    # overall_xs = np.concatenate((overall_xs, xs[1:]))
    # overall_ys = np.concatenate((overall_ys, ys[1:]))

    input_path = np.stack((overall_xs, overall_ys)).T
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

# path = make_path(np.pi/2, 100)
# plt.plot(path[:, 0], path[:, 1], '-', alpha=1, color='black', linewidth=2)
# plt.plot(path[:, 0]-path[-1, 0], path[:, 1], '-', alpha=0.3, color='black', linewidth=2)
# plt.plot(path[:, 0]+path[-1, 0], path[:, 1], '-', alpha=0.3, color='black', linewidth=2)
# plt.scatter([path[0, 0], path[-1, 0]], [0, 0], alpha=1, color='black')
# plt.axis('equal')
# plt.show()

for alpha in np.linspace(0.2, np.pi*0.99, 10):
    path = make_path(alpha, 100)
    fig, ax = plt.subplots(figsize=(8,2))
    plt.plot(path[:, 0], path[:, 1], '-', alpha=1, color='black', linewidth=2)
    plt.plot(path[:, 0]-path[-1, 0], path[:, 1], '-', alpha=0.3, color='black', linewidth=2)
    plt.plot(path[:, 0]+path[-1, 0], path[:, 1], '-', alpha=0.3, color='black', linewidth=2)
    plt.scatter([path[0, 0], path[-1, 0]], [0, 0], alpha=1, color='black')
    plt.axis('equal')
    ax.axis('off')
    fig.savefig('examples/penannular_2/allowed_paths/{0:.3f}.png'.format(alpha))
    plt.close(fig)
    # plt.show()

# plot_mismatch_map_for_penannular(N=60,
#                                  M=60,
#                                  kx_range=(0.1, 11.5),
#                                  kr_range=(0.01, 1.5*np.pi))


# data = make_path(xlen=9.5, r=2.82)
# # data = make_path(xlen=9.5, r=2.5)
# trace_on_sphere(data, kx=1, ky=1, core_radius=1, do_plot=True)