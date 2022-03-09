# Trajectoids
Given a path, computes the 3D shape ("trajectoid") that would follow this path when rolling down a slope

## Installation

Clone (checkout) this repository after installing these dependencies:

`Python 3.7+`

`trimesh`(the only hard dependency)

`matplotlib` (if plotting is needed)

`scikit-image` (for loading an image as input path)

## Running

Example usage with loading input path from an image:
```
from compute-trajectoids import *
input_path = get_trajectory_from_raster_image('vector_images/ibs_particle/ibs_v5_current_good.png')
compute_shape(input_path, 
              kx=1.0678, 
              ky=0.8009, 
              folder_for_path='trajectory_project_1',
              folder_for_meshes='cutting_boxes')
```

This will compute the same number of "cutting boxes" as the number of points in the `input_path` and save 
these boxes as `.obj` files into the `folder_for_meshes` directory. For obtaining the final trajetroid mesh,
these boxes must be subtracted from a sphere of radius `R` greater than the core radius `r=1`.

## Citation
If you use this code, please cite our paper:
```
@article{2021trajectoids,
  title={Solid-body trajectoids shaped to roll along desired pathways: downwards, upwards, and in loops},
  author={Sobolev, Yaroslav I. and Dong, Ruoyu and Granick, Steve and Grzybowski, Bartosz A.},
  journal={XXX},
  year={2022}
}
```