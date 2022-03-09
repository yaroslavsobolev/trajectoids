from compute_trajectoid import *

data0 = get_trajectory_from_raster_image('examples/ibs-v5/ibs_v5-01.png')
compute_shape(data0, kx=1.0678, ky=0.8009,
              folder_for_path='examples/ibs-v5/folder_for_path',
              folder_for_meshes='examples/ibs-v5/cut_meshes'
              )
