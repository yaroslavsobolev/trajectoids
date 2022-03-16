from compute_trajectoid import *

def test_path_from_trace():
    # we convert path to spherical trace and back, then make sure we got the same path back

    # make a path that is just a single zigzag like so: /\
    # TODO: Test more complex input paths as well for completeness
    Npath = 400
    factor = 0.2
    xs = np.linspace(0, 2*np.pi, Npath)
    ys = xs*factor
    middle = int(round(Npath/2))
    ys[middle:] = factor*np.flip(xs)[middle:]
    input_path = np.stack((xs, ys)).T
    # compute a spherical trace corresponding to flat path
    sphere_trace = trace_on_sphere(input_path, kx=1, ky=1)
    # compute a flat path corresponding to spherical trace
    path_reconstructed_from_trace = path_from_trace(sphere_trace, core_radius=1)
    # compare to the input path
    assert np.isclose(path_reconstructed_from_trace, input_path).all()

    ## optional plotting to debug if something goes wrong
    # plt.scatter(xs, ys, alpha=0.5, color='C0')
    # plt.plot(path_reconstructed_from_trace[:, 0], path_reconstructed_from_trace[:, 1], '-', alpha=0.5, color='C1')
    # plt.show()

