import numpy as np
from scipy.interpolate import UnivariateSpline
from compute_trajectoid import *
from tqdm import tqdm

def add_interval(startpoint, angle, length, Ns = 2):
    xs = startpoint[0] + np.linspace(0, length * np.cos(angle), Ns)
    ys = startpoint[1] + np.linspace(0, length * np.sin(angle), Ns)
    return np.stack((xs, ys)).T

def make_zigzag(a=np.pi/2):
    angles = [-np.pi/3, np.pi/4, -np.pi/8, np.pi/4, 0]
    lengths = [np.sqrt(2) * np.pi/2, np.sqrt(3) * np.pi/2, np.sqrt(np.pi) * np.pi/2, np.sqrt(1.231) * np.pi/2, np.sqrt(3.421) * np.pi/2, np.pi/2]
    input_path = np.array([[0, 0]])
    tips = [[0, 0]]
    for i, angle in enumerate(angles):
        startpoint = input_path[-1,:]
        new_section = add_interval(startpoint, angle, lengths[i])
        input_path = np.concatenate((input_path, new_section[1:]), axis=0)
        tips.append(new_section[-1])
    return input_path, np.array(tips)

input_path_single_section, tips = make_zigzag()
input_path_single_section[:, 1] -= np.linspace(input_path_single_section[0, 1], input_path_single_section[-1, 1], input_path_single_section.shape[0])
input_path_0 = double_the_path(input_path_single_section, do_plot=False)
spherical_trace_upsample_factor = 1
mlab.options.offscreen = True

def plot_spherical_one_period(input_path, scale, plotting_upsample_factor=100,
                                                    sphere_opacity=.8, plot_endpoints=False, endpoint_radius=0.1):
    length_from_start_to_here = cumsum_full_length_along_the_path(input_path)
    sphere_trace = trace_on_sphere(upsample_path(scale * input_path,
                                                 by_factor=plotting_upsample_factor), kx=1, ky=1)
    length_from_start_to_here = np.repeat(length_from_start_to_here, repeats=plotting_upsample_factor)[:sphere_trace.shape[0]]
    logging.debug('Mlab plot begins...')
    core_radius = 1
    tube_radius = 0.01
    mfig = mlab.figure(size=(1024, 1024), \
                bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4, sphere_opacity=sphere_opacity)
    mlab.plot3d(sphere_trace[:, 0],
                sphere_trace[:, 1],
                sphere_trace[:, 2],
                length_from_start_to_here, colormap='viridis',
                tube_radius=tube_radius)
    colors = [(1, 0, 0), (0, 0, 0)]
    for i, point_here in enumerate([sphere_trace[-1], sphere_trace[0]]):
        mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=endpoint_radius, color=colors[i])
    return mfig

scale_to_plot = 0.25
mfig = plot_spherical_one_period(
    input_path_single_section,
    scale_to_plot,
    sphere_opacity=0.6)

f = mlab.gcf()
f.scene._lift()
mlab.view(azimuth=45, elevation=60, distance=5, focalpoint=(0, 0, 0))
mlab.savefig(f'misc/figures/tpt-theorem-video/one-period.png', magnification=2)
mlab.close(all=True)

# scale_to_plot = 0.001
# mfig = plot_spherical_one_period(
#     input_path_single_section,
#     scale_to_plot,
#     sphere_opacity=0.6)
# f = mlab.gcf()
# f.scene._lift()
# mlab.view(azimuth=45, elevation=60, distance=5, focalpoint=(0, 0, 0))
# # mlab.show()
# mlab.savefig(f'misc/figures/tpt-theorem-video/null_sphere.png', magnification=2)
# mlab.close(all=True)
#
# def plot_spherical_two_period_zigzag(input_path, input_path_doubled_upsampled, scale, plotting_upsample_factor=100,
#                                                     sphere_opacity=.8, plot_endpoints=False, endpoint_radius=0.1):
#     length_from_start_to_here = cumsum_half_length_along_the_path(input_path)
#     sphere_trace = trace_on_sphere(scale * input_path_doubled_upsampled, kx=1, ky=1)
#     length_from_start_to_here = length_from_start_to_here[:-1]
#     length_from_start_to_here = np.repeat(length_from_start_to_here, repeats=plotting_upsample_factor)
#     length_from_start_to_here = np.insert(length_from_start_to_here, 0, 0)
#     logging.debug('Mlab plot begins...')
#     core_radius = 1
#     tube_radius = 0.01
#     mfig = mlab.figure(size=(1024, 1024), \
#                 bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
#     plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4, sphere_opacity=sphere_opacity)
#     mlab.plot3d(sphere_trace[:, 0],
#                 sphere_trace[:, 1],
#                 sphere_trace[:, 2],
#                 length_from_start_to_here, colormap='viridis',
#                 tube_radius=tube_radius)
#     colors = [(1, 0, 0), (0, 0, 0)]
#     for i, point_here in enumerate([sphere_trace[-1], sphere_trace[0]]):
#         mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=endpoint_radius, color=colors[i])
#     return mfig
#
#
# nframes = 100
# path_doubled_upsampled = double_the_path(upsample_path(input_path_single_section, by_factor=100), do_plot=False)
# # best_scale = minimize_mismatch_by_scaling(path_doubled_upsampled, scale_range=(0.27, 0.35))
# # print(f'best_scale = {best_scale}')
# best_scale = 0.3282796374942105
# scale_list = np.linspace(0.25, best_scale, nframes)
# for frame_id, scale_to_plot in enumerate(tqdm(scale_list)):
#     mfig = plot_spherical_two_period_zigzag(
#         input_path_0,
#         input_path_doubled_upsampled=path_doubled_upsampled,
#         scale=scale_to_plot,
#         sphere_opacity=0.6)
#     f = mlab.gcf()
#     f.scene._lift()
#     mlab.view(azimuth=45, elevation=60, distance=5, focalpoint=(0, 0, 0))
#     mlab.savefig(f'misc/figures/tpt-theorem-video/doubled-tuning-one-point-frames/frame{frame_id:08d}.png', magnification=2)
#     mlab.close(all=True)



# def plot_spherical_two_period_zigzag_noncontact_points(input_path, input_path_doubled_upsampled, scale, plotting_upsample_factor=100,
#                                                     sphere_opacity=.8, plot_endpoints=False, endpoint_radius=0.1):
#     startpoints = ([0, 0, -1], [0, 1, 0], [0, -1, 0], [0, 0, 1], [1/np.sqrt(2), 1/np.sqrt(2), 0], [-1/np.sqrt(2), -1/np.sqrt(2), 0])
#     length_from_start_to_here = cumsum_half_length_along_the_path(input_path)
#     length_from_start_to_here = length_from_start_to_here[:-1]
#     length_from_start_to_here = np.repeat(length_from_start_to_here, repeats=plotting_upsample_factor)
#     length_from_start_to_here = np.insert(length_from_start_to_here, 0, 0)
#     logging.debug('Mlab plot begins...')
#     core_radius = 1
#     tube_radius = 0.01
#     mfig = mlab.figure(size=(1024, 1024), \
#                 bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
#     plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4, sphere_opacity=sphere_opacity)
#
#     for startpoint in startpoints:
#         sphere_trace = trace_on_sphere_nonocontact_point(scale * input_path_doubled_upsampled, kx=1, ky=1, startpoint=startpoint)
#         mlab.plot3d(sphere_trace[:, 0],
#                     sphere_trace[:, 1],
#                     sphere_trace[:, 2],
#                     length_from_start_to_here, colormap='viridis',
#                     tube_radius=tube_radius)
#         colors = [(1, 0, 0), (0, 0, 0)]
#         for i, point_here in enumerate([sphere_trace[-1], sphere_trace[0]]):
#             mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=endpoint_radius, color=colors[i])
#     return mfig
#
#
# nframes = 100
# path_doubled_upsampled = double_the_path(upsample_path(input_path_single_section, by_factor=100), do_plot=False)
# # best_scale = minimize_mismatch_by_scaling(path_doubled_upsampled, scale_range=(0.27, 0.35))
# # print(f'best_scale = {best_scale}')
# best_scale = 0.3282796374942105
# scale_list = np.linspace(0.01, best_scale, nframes)
# for frame_id, scale_to_plot in enumerate(tqdm(scale_list)):
#     mfig = plot_spherical_two_period_zigzag_noncontact_points(
#         input_path_0,
#         input_path_doubled_upsampled=path_doubled_upsampled,
#         scale=scale_to_plot,
#         sphere_opacity=0.6)
#     f = mlab.gcf()
#     f.scene._lift()
#     mlab.view(azimuth=45, elevation=60, distance=5, focalpoint=(0, 0, 0))
#     mlab.savefig(f'misc/figures/tpt-theorem-video/doubled-tuning-many-point-frames/frame{frame_id:08d}.png', magnification=2)
#     mlab.close(all=True)