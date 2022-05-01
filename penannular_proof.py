import matplotlib.pyplot as plt
import numpy as np
import mayavi

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

def make_path(xlen, r, shift=0.25, Npath = 400, do_double=True):
    # first linear section
    step_size = xlen/Npath
    overall_xs = np.linspace(0, xlen/2 - r - shift, int(round((xlen/2 - r - shift)/step_size)))
    overall_ys = np.zeros_like(overall_xs)

    # semicirle section
    nsteps_in_theta = int(round(np.pi*r/step_size))
    thetas = np.linspace(np.pi, 0, nsteps_in_theta)
    xs = r*np.cos(thetas) + xlen/2 - shift
    ys = r*np.sin(thetas)
    overall_xs = np.concatenate((overall_xs[:-1], xs))
    overall_ys = np.concatenate((overall_ys[:-1], ys))

    # second linear section
    xs = np.linspace(xlen/2 + r - shift, xlen, int(round((xlen/2 - r + shift)/step_size)))
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

def align_view(scene):
    scene.scene.camera.position = [0.8338505129447937, -4.514338837405451, -4.8515133455799955]
    scene.scene.camera.focal_point = [0.00025303918109570445, 0.0, -0.007502121504843806]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [0.04727430775215174, -0.7266741891255103, 0.6853537500337594]
    scene.scene.camera.clipping_range = [3.573025799931812, 10.592960367304393]
    scene.scene.camera.compute_view_plane_normal()

# input_path_symm = make_path(xlen=3.81, r=1.23, Npath=150, shift=0, do_double=False) * best_scale
# plt.plot(input_path_symm[:, 0], input_path_symm[:, 1], '--', color='C2', label='Symmetric')
# plt.scatter(input_path_symm[0, 0], input_path_symm[0, 1], color='black')
# plt.scatter(input_path_symm[-1, 0], input_path_symm[-1, 1], color='black')
# _ = double_the_path(input_path_single_section, do_plot=True)
tube_radius = 0.01
core_radius = 1

# best_scale = minimize_mismatch_by_scaling(input_path, scale_range=(0.9, 1.1))
best_scale = 1.006534368463883
input_path = make_path(xlen=3.81, r=1.23, Npath=150) * best_scale
input_path_single_section = make_path(xlen=3.81, r=1.23, Npath=150, do_double=False) * best_scale
sphere_trace = trace_on_sphere(input_path, kx=1, ky=1)
sphere_trace_single_section = trace_on_sphere(input_path_single_section, kx=1, ky=1)
mfig = mlab.figure(size=(1024, 768), \
            bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))

plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
last_index = sphere_trace.shape[0] // 2
l = mlab.plot3d(sphere_trace[last_index:, 0], sphere_trace[last_index:, 1], sphere_trace[last_index:, 2], color=(0, 0, 1),
                tube_radius=tube_radius)
l = mlab.plot3d(sphere_trace_single_section[:, 0],
                sphere_trace_single_section[:, 1],
                sphere_trace_single_section[:, 2], color=(0, 1, 0),
                tube_radius=tube_radius)
align_view(mfig)
mlab.savefig('examples/penannular_proof/full_trace.png')
mlab.show()

def make_animation_of_scaling_sweep():
    tube_radius = 0.01
    core_radius = 1

    best_scale = 1.006534368463883
    input_path = make_path(xlen=3.81, r=1.23, Npath=150) * best_scale
    input_path_single_section = make_path(xlen=3.81, r=1.23, Npath=150, do_double=False) * best_scale

    mfig = mlab.figure(size=(1024, 768), \
                bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
    align_view(mfig)
    nframes = 100
    for frame_id, scale in enumerate(np.linspace(0.1*best_scale, best_scale, nframes)):
        input_path_single_section = make_path(xlen=3.81, r=1.23, Npath=150, do_double=False) * scale
        sphere_trace_single_section = trace_on_sphere(input_path_single_section, kx=1, ky=1)
        object1 = mlab.plot3d(sphere_trace_single_section[:, 0],
                        sphere_trace_single_section[:, 1],
                        sphere_trace_single_section[:, 2], color=(0, 1, 0),
                        tube_radius=tube_radius)
        arc_here = bridge_two_points_by_arc(sphere_trace_single_section[0, :], sphere_trace_single_section[-1, :], npoints=30)
        object2 = mlab.plot3d(arc_here[:, 0],
                        arc_here[:, 1],
                        arc_here[:, 2], color=(1, 0, 0),
                        tube_radius=tube_radius)
        points_list = []
        for point_here in [sphere_trace_single_section[0, :], sphere_trace_single_section[-1, :]]:
            points_list.append(mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.05, color=(0, 0, 0)))
        mlab.savefig('examples/penannular_proof/scale-sweep-frames/{0:08d}.png'.format(frame_id))
        for x in [object1, object2]:
            x.remove()
        for x in points_list:
            x.remove()



def make_animation_of_rotational_symmetry():
    tube_radius = 0.01
    core_radius = 1

    best_scale = 1.006534368463883
    input_path = make_path(xlen=3.81, r=1.23, Npath=150) * best_scale
    input_path_single_section = make_path(xlen=3.81, r=1.23, Npath=150, do_double=False) * best_scale

    # ====== Make animation of the initial scaling sweep
    mfig = mlab.figure(size=(1024, 768), \
                bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4)
    align_view(mfig)

    sphere_trace_single_section = trace_on_sphere(input_path_single_section, kx=1, ky=1)
    mlab.plot3d(sphere_trace_single_section[:, 0],
                          sphere_trace_single_section[:, 1],
                          sphere_trace_single_section[:, 2], color=(0, 1, 0),
                          tube_radius=tube_radius)
    arc_here = bridge_two_points_by_arc(sphere_trace_single_section[0, :], sphere_trace_single_section[-1, :], npoints=30)
    mlab.plot3d(arc_here[:, 0],
                          arc_here[:, 1],
                          arc_here[:, 2], color=(1, 0, 0),
                          tube_radius=tube_radius)
    for point_here in [sphere_trace_single_section[0, :], sphere_trace_single_section[-1, :]]:
        mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.05, color=(0, 0, 0))

    # draw axis
    axis_of_symmetry = sphere_trace_single_section[0, :] + sphere_trace_single_section[-1, :]
    axis_of_symmetry = axis_of_symmetry/np.linalg.norm(axis_of_symmetry)
    for point_here in [[0,0,0], axis_of_symmetry]:
        mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.05, color=(0, 0, 0))
    axis_len = 1
    mlab.plot3d([0, axis_of_symmetry[0] * axis_len],
                          [0, axis_of_symmetry[1] * axis_len],
                          [0, axis_of_symmetry[2] * axis_len], color=(0, 0, 0),
                          tube_radius=tube_radius)
    nframes = 100
    for frame_id, angle in enumerate(np.linspace(0, np.pi, nframes)):
        if frame_id > 0:
            trace_copy = np.copy(sphere_trace_single_section)
            for i in range(trace_copy.shape[0]):
                trace_copy[i, :] = rotate_3d_vector(trace_copy[i, :], axis_of_rotation=axis_of_symmetry, angle=angle)
            object1 = mlab.plot3d(trace_copy[:, 0],
                              trace_copy[:, 1],
                              trace_copy[:, 2], color=(0, 0, 1),
                              tube_radius=tube_radius)
        mlab.savefig('examples/penannular_proof/symmetry_rotation_frames/{0:08d}.png'.format(frame_id))
        if frame_id > 0:
            for x in [object1]:
                x.remove()

# make_animation_of_rotational_symmetry()

# make_orbit_animation(folder_for_frames='examples/penannular_1/orbit_frames', elevation=60)

# for point_here in [sphere_trace_single_section[-1], sphe]:
#     mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=0.05, color=(1, 1, 0))
