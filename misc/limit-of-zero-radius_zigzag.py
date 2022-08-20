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

# plt.plot(input_path_single_section[:, 0], input_path_single_section[:, 1], 'o-', alpha=0.33)
# plt.axis('equal')
# plt.show()

# input_path_0 = double_the_path(input_path_single_section, do_plot=True)
spherical_trace_upsample_factor = 1


mlab.options.offscreen = True

# scale_to_plot = 2.2
real_frame_id = 0
power_law = 3
scale_list = np.linspace(0.01**(1/power_law), (100)**(1/power_law), 400) ** power_law
# scale_list = [100]

def plot_spherical_one_period(input_path, scale, plotting_upsample_factor=1000,
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
    for point_here in [sphere_trace[-1], sphere_trace[0]]:
        mlab.points3d(point_here[0], point_here[1], point_here[2], scale_factor=endpoint_radius, color=(0, 0, 0))
    return mfig

# for frame_id, scale_to_plot in enumerate(tqdm(scale_list)):
#     # if frame_id > 100 and frame_id % 2:
#     # Plot the 3D and show it in the matplotlib subplot
#     mfig = plot_spherical_one_period(
#         input_path_single_section,
#         scale_to_plot,
#         sphere_opacity=0.6)
#     # plot certain points
#     # sphere_trace_single_section = trace_on_sphere(scale_to_plot * input_path_single_section, kx=1, ky=1)
#     # colors_of_trace_points = [(0, 0, 1), (0, 1, 0)]
#     # sphere_trace_full = trace_on_sphere(scale_to_plot * input_path_0, kx=1, ky=1)
#     f = mlab.gcf()
#     f.scene._lift()
#     # cam = mfig.scene.camera
#     # cam.zoom(1.5)
#     mlab.view(azimuth=45, elevation=60, distance=5, focalpoint=(0, 0, 0))
#     # mlab.show()
#     mlab.savefig(f'examples/small-radius-limit-zigzag/scale-sweep-frames/frame{frame_id:08d}.png', magnification=2)
#     mlab.close(all=True)

input_path_0 = double_the_path(input_path_single_section, do_plot=True)
x = input_path_0[:, 0]
y = input_path_0[:, 1]
pathlength = np.sum( np.sqrt((np.diff(x)**2 + np.diff(y)**2)) )/2
print(f'Pathlength: {pathlength}')

path_linewidth=2
path_alpha=1
plot_single_period=True
trace_upsample_factor = 1
circle_center=[5, 1]
circlealpha=1
path_for_figs = 'examples/small-radius-limit-zigzag/scale_sweep_flat_path_frames'
for frame_id, scale_to_plot in enumerate(tqdm(scale_list)):
    fig, axs = plt.subplots()
    # plot_flat_path_with_color(input_path_0, input_path_single_section, axs)
    # plot_flat_path_with_color(upsample_path(input_path_0, by_factor=trace_upsample_factor),
    #                           upsample_path(input_path_single_section, by_factor=trace_upsample_factor),
    #                           axs, linewidth=path_linewidth, alpha=path_alpha,
    #                           plot_single_period=plot_single_period)
    linewidth = path_linewidth
    alpha = path_alpha
    x = input_path_single_section[:, 0]
    y = input_path_single_section[:, 1]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Coloring the curve
    length_from_start_to_here = cumsum_full_length_along_the_path(input_path_single_section)
    norm = plt.Normalize(length_from_start_to_here.min(), length_from_start_to_here.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(length_from_start_to_here)
    lc.set_linewidth(linewidth)
    lc.set_alpha(alpha)
    line = axs.add_collection(lc)

    # black dots at middle and ends of the path
    for point in [input_path_single_section[0], input_path_single_section[-1]]:
        axs.scatter(point[0], point[1], color='black', s=10)

    plt.axis('equal')

    plt.title(f'$\sigma = L/(2 \pi r)$: {pathlength/(2*np.pi)*scale_to_plot:.3f}')
    # plot circle showing relative diameter of the sphere
    circle_rad = 1 / scale_to_plot
    circle1 = plt.Circle((circle_center[0], circle_center[1]),
                         circle_rad, fill=False, linewidth=2, edgecolor='C1', alpha=circlealpha)
    axs.add_patch(circle1)
    plt.axis('off')
    fig.savefig(path_for_figs + f'/{frame_id:08d}.png', dpi=300)
    plt.close(fig)