import numpy as np
from scipy.interpolate import UnivariateSpline
from compute_trajectoid import *
from tqdm import tqdm

def sigmoid(x,mi, mx):
    return mi + (mx-mi)*(lambda t: (1+200**(-t+0.5))**(-1) )( (x-mi)/(mx-mi) )


def smoothclamp(x, mi, mx):
    return mi + (mx-mi)*(lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )


def window_function(x, windowlen):
    mask = smoothclamp(np.arange(x.shape[0]), mi=0, mx=windowlen)
    mask = mask/windowlen
    # plt.plot(mask, 'o-')
    # plt.show()
    return mask

#
# target_folder='examples/random_doubled_3'

# input_path = np.load(target_folder + '/folder_for_path/path_data.npy')
# plot_three_path_periods(input_path, plot_midpoints=True, savetofile=target_folder + '/input_path')

# input_path_single_section = make_random_path(seed=1, amplitude=3, make_ends_horizontal='both', end_with_zero=True)
input_path_single_section = make_random_path(seed=1, make_ends_horizontal=False, start_from_zero=True, end_with_zero=True, amplitude=3.5,
                                             savgom_window_1=41, savgol_window_2=5)

spl = UnivariateSpline(input_path_single_section[:,0], input_path_single_section[:,1])
spl.set_smoothing_factor(0.7)
plt.plot(input_path_single_section[:,0], input_path_single_section[:,1], 'o')
plt.plot(input_path_single_section[:,0], spl(input_path_single_section[:,0]), 'g', lw=3)
# plt.plot(xs, spl(xs), 'b', lw=3)
plt.axis('equal')
plt.show()
upsample_by = 50
xs = np.linspace(np.min(input_path_single_section[:, 0]), np.max(input_path_single_section[:, 0]), input_path_single_section.shape[0] * upsample_by)
ys = spl(xs)

input_path_single_section = np.stack((xs, ys)).T

window_for_ends = 25 * upsample_by
plt.plot(input_path_single_section[:,0], input_path_single_section[:,1], 'o-')
input_path_single_section[:,1] = input_path_single_section[:,1] * window_function(input_path_single_section[:,0], windowlen=window_for_ends)
# plt.plot(input_path_single_section[:,0], input_path_single_section[:,1], 'o-')
input_path_single_section[:,1] = input_path_single_section[:,1] * np.flip(window_function(input_path_single_section[:,0], windowlen=window_for_ends))
plt.plot(input_path_single_section[:,0], input_path_single_section[:,1], 'o-')
plt.axis('equal')
plt.show()

input_path_0 = double_the_path(input_path_single_section, do_plot=True)
spherical_trace_upsample_factor = 1


mlab.options.offscreen = True

# scale_to_plot = 2.2
real_frame_id = 0
power_law = 3
scale_list = np.linspace(0.01**(1/power_law), (100)**(1/power_law), 400) ** power_law

# for frame_id, scale_to_plot in enumerate(tqdm(scale_list)):
#     # if frame_id > 100 and frame_id % 2:
#     # Plot the 3D and show it in the matplotlib subplot
#     mfig = plot_spherical_trace_with_color_along_the_trace(
#         input_path_0,
#         input_path_single_section, scale_to_plot,
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
#     mlab.savefig(f'examples/random_doubled_3/scale-sweep-frames/frame{frame_id:08d}.png', magnification=2)
#     mlab.close(all=True)
x = input_path_0[:, 0]
y = input_path_0[:, 1]
pathlength = np.sum( np.sqrt((np.diff(x)**2 + np.diff(y)**2)) )/2
print(f'Pathlength: {pathlength}')

path_linewidth=2
path_alpha=1
plot_single_period=False
trace_upsample_factor = 1
circle_center=[5, 1]
circlealpha=1
path_for_figs = 'examples/random_doubled_3/scale_sweep_flat_path_frames'
for frame_id, scale_to_plot in enumerate(tqdm(scale_list)):
    fig, axs = plt.subplots()
    # plot_flat_path_with_color(input_path_0, input_path_single_section, axs)
    plot_flat_path_with_color(upsample_path(input_path_0, by_factor=trace_upsample_factor),
                              upsample_path(input_path_single_section, by_factor=trace_upsample_factor),
                              axs, linewidth=path_linewidth, alpha=path_alpha,
                              plot_single_period=plot_single_period)
    plt.title(f'$T/(2 \pi r)$: {pathlength/(2*np.pi)*scale_to_plot:.3f}')
    # plot circle showing relative diameter of the sphere
    circle_rad = 1 / scale_to_plot
    circle1 = plt.Circle((circle_center[0], circle_center[1]),
                         circle_rad, fill=False, linewidth=2, edgecolor='C1', alpha=circlealpha)
    axs.add_patch(circle1)
    plt.axis('off')
    fig.savefig(path_for_figs + f'/{frame_id:08d}.png', dpi=300)
    plt.close(fig)