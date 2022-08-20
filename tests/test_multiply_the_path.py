from compute_trajectoid import *

def make_zigzag_path(Npath, factor, factor2):
    '''Make a path that is just a single V-shaped zigzag'''
    xs = np.linspace(0, 2 * np.pi * factor2, Npath)
    ys = xs * factor
    middle = int(round(Npath / 2))
    ys[middle:] = factor * np.flip(xs)[middle:]
    return np.stack((xs, ys)).T

def test_the_tripling_functio():
    input_path = make_zigzag_path(Npath=400, factor=0.2, factor2=1)
    mpath = multiply_the_path(input_path, m=3)
    assert np.isclose(mpath[-1, 0], input_path[-1, 0] * 3)