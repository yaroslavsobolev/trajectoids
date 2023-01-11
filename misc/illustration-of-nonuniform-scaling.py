from compute_trajectoid import *

data0 = get_trajectory_from_raster_image('examples/ibs-v5/ibs_v5-01.png')
# compute_shape(data0, kx=1.0678, ky=0.8009,
#               folder_for_path='examples/ibs-v5/folder_for_path',
#               folder_for_meshes='examples/ibs-v5/cut_meshes'
#               )

bscale = 1.2

# kx=1.0678
# ky=0.8009

kx=1.04
ky=0.60

core_radius=1
do_plot=False

# data = np.copy(data0)
# data[:, 0] = data[:, 0] * kx
# data[:, 1] = data[:, 1] * ky  # +  kx * np.sin(data0[:, 0]/2)

core_radius = 1
mfig = mlab.figure(size=(1024, 768), bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
tube_radius = 0.01
arccolor = tuple(np.array([44, 160, 44]) / 255)
arccolor = (1, 0, 0)
plot_sphere(r0=core_radius - tube_radius, line_radius=tube_radius / 4, sphere_opacity=0.4)
tube_radius = tube_radius * 3

sphere_trace = trace_on_sphere(data0, kx=1 * bscale, ky=1.1 * bscale) * (1 + tube_radius)
l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=tuple(np.array([130, 130, 130])/255),
                tube_radius=tube_radius)

sphere_trace = trace_on_sphere(data0, kx=kx, ky=ky) * (1 + tube_radius)
l = mlab.plot3d(sphere_trace[:, 0], sphere_trace[:, 1], sphere_trace[:, 2], color=tuple(np.array([31,119,180])/255),
                tube_radius=tube_radius)

mlab.show()




# Flat illustration

figtraj = plt.figure(10, figsize=(10, 5))
savetofile='misc/figures/scaling_ibs_illustration'
linewidth=2

kx=1 * bscale
ky=1.1 * bscale
ky = ky / kx
kx = 1

input_path = np.copy(data0)
input_path[:, 0] = input_path[:, 0] * kx
input_path[:, 1] = input_path[:, 1] * ky  # +  kx * np.sin(data0[:, 0]/2)
dataxlen = np.max(input_path[:, 0])


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
    
    
kx=1.04
ky=0.60

ky = ky / kx
kx = 1
input_path = np.copy(data0)
input_path[:, 0] = input_path[:, 0] * kx
input_path[:, 1] = input_path[:, 1] * ky  # +  kx * np.sin(data0[:, 0]/2)
dataxlen = np.max(input_path[:, 0])

color = 'C0'

def plot_periods(data, linestyle, linewidth):
    plt.plot(data[:, 0], data[:, 1], color=color, alpha=0.3, linestyle=linestyle, linewidth=linewidth)
    plt.plot(dataxlen + data[:, 0], data[:, 1], color=color, alpha=1, linestyle=linestyle, linewidth=linewidth)
    plt.plot(2 * dataxlen + data[:, 0], data[:, 1], color=color, alpha=0.3, linestyle=linestyle,
             linewidth=linewidth)
    plt.plot(3 * dataxlen + data[:, 0], data[:, 1], color=color, alpha=0.3, linestyle=linestyle,
             linewidth=linewidth)

# plot_periods(data, '--', linewidth=0.5)
plot_periods(input_path, '-', linewidth=linewidth)
# plot_periods(projection_centers, '-', linewidth=1)

# for shift in dataxlen * np.arange(3):
#     plt.scatter(shift + input_path[-1, 0], input_path[-1, 1], s=35, color=color)
    
    
# plt.scatter(dataxlen + input_path[-1, 0], input_path[-1, 1], s=35, color='black')
plt.axis('equal')
if savetofile:
    figtraj.savefig(f'{savetofile}.png', dpi=300)
    figtraj.savefig(f'{savetofile}.eps')
plt.show()

