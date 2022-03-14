from compute_trajectoid import *

Npath = 400
factor = 0.2
xs = np.linspace(0, 2*np.pi, Npath)
ys = xs*factor
middle = int(round(Npath/2))
ys[middle:] = factor*np.flip(xs)[middle:]
# plt.scatter(xs, ys)
# plt.show()
data0 = np.stack((xs, ys)).T

sphere_trace = trace_on_sphere(data0, kx=1, ky=1)
path_from_trace(sphere_trace, core_radius=1)